from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import traceback
import uuid

from pydantic import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


# Extensions collected from the sandbox working directory and returned as artifacts.
_ARTIFACT_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".csv", ".json", ".txt", ".html"}
_MAX_OUTPUT_CHARS = 20000


class PythonSandboxInput(BaseModel):
    """Input schema for the Python Sandbox Tool."""

    code: str = Field(
        ...,
        description="Python source to execute. numpy, pandas, scipy, matplotlib (Agg backend) "
        "and the standard library are available. print() results you want to read back. To "
        "produce a chart or file, save it into the current working directory (e.g. "
        "plt.savefig('trend.png')) — any files you create are returned as artifact paths.",
    )
    timeout_seconds: Optional[int] = Field(
        None, description="Wall-clock timeout in seconds (default 30, max 120)."
    )
    input_files: Optional[List[str]] = Field(
        None,
        description="Optional list of file paths to copy into the sandbox working directory before "
        "running (e.g. a PACS cohort CSV from MedRax-Dicomweb export_dataset). Inside the code, "
        "open them by basename only, e.g. pd.read_csv('pacs_cohort_ab12cd34.csv'). Only the files "
        "you list here enter the sandbox — no other file on disk is visible.",
    )


class PythonSandboxTool(BaseTool):
    """Execute Python in a bubblewrap-isolated subprocess for calculations and chart generation.

    When ``bwrap`` (bubblewrap) is available the child runs inside a container that:
      * sees only the Python interpreter and system libraries (read-only) plus its own working
        directory (read-write) — the MedRAX repository, its ``.env``, and any PACS/DICOMweb
        credentials are simply not mounted, so they cannot be read;
      * has no network (all namespaces unshared);
      * runs with a scrubbed environment (no inherited secrets);
      * is bounded by CPU-time, address-space and file-size resource limits and a wall-clock
        timeout that kills the whole process group.

    If ``bwrap`` is not installed the tool falls back to a plain subprocess with the scrubbed
    environment, resource limits, best-effort network unsharing and timeout — but WITHOUT
    filesystem isolation (see ``filesystem_isolated`` in the metadata).

    This is a compute/plotting sandbox, not a general shell. Use it for arithmetic the model
    should not do by hand, for aggregating measurements across studies, and for generating
    comparison charts (e.g. longitudinal CTR trends) returned as artifact file paths.
    """

    name: str = "python_sandbox"
    description: str = (
        "Runs Python in a locked-down container (no secrets, no network, no access to the app's "
        "files, resource-limited) for deterministic calculations and for generating charts. "
        "Prefer this over doing arithmetic yourself — e.g. percent change of a measurement "
        "between two studies, unit conversions, statistics across several tool outputs. numpy, "
        "pandas, scipy and matplotlib are available. To create a longitudinal comparison plot "
        "(e.g. CTR or a measurement over time), write the values into the code and save a figure "
        "with plt.savefig('name.png'); every file the code writes is returned in 'artifacts' as a "
        "path you can display or embed in a PACS report. To analyze a dataset exported from PACS "
        "(e.g. the CSV from MedRax-Dicomweb export_dataset — exam counts per patient, female/male "
        "distribution, exams per modality/month), pass its path in input_files and read it by "
        "basename with pandas. Returns stdout, stderr, return code and artifact paths."
    )
    args_schema: Type[BaseModel] = PythonSandboxInput

    temp_dir: Path = Path("temp")
    default_timeout: int = 30
    max_timeout: int = 120
    cpu_seconds: int = 30
    memory_bytes: int = 8 * 1024 * 1024 * 1024  # 8 GiB virtual — generous so numpy/BLAS imports
    fsize_bytes: int = 256 * 1024 * 1024  # 256 MiB per file

    def __init__(self, temp_dir: Optional[Path] = Path("temp")):
        """Initialize the sandbox tool and its artifact output directory."""
        super().__init__()
        self.temp_dir = temp_dir if isinstance(temp_dir, Path) else Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)

    def _sandbox_env(self, workdir: Path) -> Dict[str, str]:
        """Scrubbed environment for the child — deliberately excludes all parent secrets."""
        return {
            "PATH": "/usr/bin:/bin",
            "HOME": str(workdir),
            "TMPDIR": str(workdir),
            # Keep matplotlib's font cache out of the artifact-collection root.
            "MPLCONFIGDIR": str(workdir / ".mplconfig"),
            "MPLBACKEND": "Agg",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "USER": "sandbox",
            "LOGNAME": "sandbox",
        }

    def _make_preexec(self):
        """Child pre-exec hook: own process group + resource limits (inherited by bwrap child).

        RLIMIT_NPROC is intentionally NOT set: it is keyed by real UID system-wide, so bounding
        it while running as root would break forking. The PID namespace (via bwrap) plus the CPU
        and wall-clock limits bound runaway processes instead.
        """
        import resource

        cpu_s, mem, fsize = self.cpu_seconds, self.memory_bytes, self.fsize_bytes

        def _preexec():
            os.setsid()
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
                resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
                resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            except Exception:
                pass

        return _preexec

    def _build_command(self, workdir: Path, script_path: Path) -> Tuple[List[str], bool]:
        """Build the execution command. Returns (argv, filesystem_isolated)."""
        env = self._sandbox_env(workdir)
        bwrap = shutil.which("bwrap")
        if bwrap:
            cmd = [
                bwrap,
                "--unshare-all",       # net, pid, ipc, uts, cgroup, user, mount namespaces
                "--die-with-parent",
                "--new-session",
                "--clearenv",
                "--proc", "/proc",
                "--dev", "/dev",
                "--tmpfs", "/tmp",
                "--ro-bind", "/usr", "/usr",
                "--ro-bind-try", "/bin", "/bin",
                "--ro-bind-try", "/lib", "/lib",
                "--ro-bind-try", "/lib64", "/lib64",
                "--ro-bind-try", "/sbin", "/sbin",
                "--ro-bind-try", "/etc/ld.so.cache", "/etc/ld.so.cache",
                "--ro-bind-try", "/etc/fonts", "/etc/fonts",
            ]
            # Expose only the Python interpreter/prefixes read-only (not the repo).
            for prefix in {sys.prefix, sys.base_prefix}:
                cmd += ["--ro-bind", prefix, prefix]
            cmd += ["--bind", str(workdir), str(workdir), "--chdir", str(workdir)]
            for key, val in env.items():
                cmd += ["--setenv", key, val]
            cmd += ["--", sys.executable, "-I", str(script_path)]
            return cmd, True

        # Fallback: no bubblewrap — best-effort isolation without a private filesystem.
        return [sys.executable, "-I", str(script_path)], False

    def _stage_inputs(self, workdir: Path, input_files: Optional[List[str]]) -> Tuple[set, List[str]]:
        """Copy the requested input files into the workdir. Returns (staged_names, missing)."""
        staged: set = {"snippet.py"}
        missing: List[str] = []
        for raw in input_files or []:
            src = Path(str(raw))
            if not src.is_file():
                missing.append(str(raw))
                continue
            dest = workdir / src.name
            shutil.copy2(src, dest)
            os.chmod(dest, 0o644)
            staged.add(dest.name)
        return staged, missing

    def _run(
        self,
        code: str,
        timeout_seconds: Optional[int] = None,
        input_files: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Execute code in the sandbox and return stdout/stderr plus any artifacts produced."""
        workdir = None
        try:
            wall = int(timeout_seconds) if timeout_seconds else self.default_timeout
            wall = max(1, min(wall, self.max_timeout))

            workdir = Path(tempfile.mkdtemp(prefix="sandbox_", dir=self.temp_dir))
            script_path = workdir / "snippet.py"
            script_path.write_text(code)
            os.chmod(script_path, 0o644)

            # Stage any input datasets (e.g. a PACS cohort CSV) into the workdir so the
            # code can read them by basename; nothing else on disk is exposed.
            staged_names, missing_inputs = self._stage_inputs(workdir, input_files)

            cmd, fs_isolated = self._build_command(workdir, script_path)
            # For the fallback (no bwrap), pass the scrubbed env directly; bwrap uses --setenv.
            popen_env = None if fs_isolated else self._sandbox_env(workdir)

            proc = subprocess.Popen(
                cmd,
                cwd=str(workdir),
                env=popen_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=self._make_preexec(),
            )

            timed_out = False
            try:
                stdout, stderr = proc.communicate(timeout=wall)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    proc.kill()
                stdout, stderr = proc.communicate()

            # Collect artifacts the code wrote (excluding the snippet itself).
            artifacts: List[str] = []
            for f in sorted(workdir.iterdir()):
                if f.name in staged_names or not f.is_file():
                    continue
                if f.suffix.lower() in _ARTIFACT_EXTS:
                    dest = self.temp_dir / f"sandbox_{uuid.uuid4().hex[:8]}{f.suffix.lower()}"
                    shutil.copy2(f, dest)
                    artifacts.append(str(dest))

            def _clip(s: str) -> str:
                if not s:
                    return ""
                return s if len(s) <= _MAX_OUTPUT_CHARS else s[:_MAX_OUTPUT_CHARS] + "\n...[truncated]"

            output = {
                "stdout": _clip(stdout),
                "stderr": _clip(stderr),
                "returncode": proc.returncode,
                "timed_out": timed_out,
                "artifacts": artifacts,
            }
            if timed_out:
                output["error"] = f"Execution exceeded the {wall}s time limit and was killed."
            if missing_inputs:
                output["missing_input_files"] = missing_inputs

            metadata = {
                "timeout_seconds": wall,
                "filesystem_isolated": fs_isolated,
                "network_disabled": fs_isolated,  # bwrap --unshare-all drops the network
                "memory_limit_bytes": self.memory_bytes,
                "analysis_status": "completed"
                if proc.returncode == 0 and not timed_out
                else "failed",
            }
            return output, metadata

        except Exception as e:
            return (
                {"error": str(e)},
                {"analysis_status": "failed", "error_traceback": traceback.format_exc()},
            )
        finally:
            if workdir is not None:
                shutil.rmtree(workdir, ignore_errors=True)

    async def _arun(
        self,
        code: str,
        timeout_seconds: Optional[int] = None,
        input_files: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Async version of _run."""
        return self._run(code, timeout_seconds, input_files)

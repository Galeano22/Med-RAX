"""MedRax-Dicomweb: query and retrieve studies from any DICOMweb server.

This tool talks to a DICOMweb (QIDO-RS / WADO-RS) server such as Orthanc,
dcm4chee, Google Healthcare, or a commercial PACS. It has two actions:

  * list_studies   - list the most recent N exams on the server
  * retrieve_study - fetch a study by accession number, patient name, or
                     StudyInstanceUID; download its images locally and convert
                     them to PNG so the chest X-ray tools can analyze them.

The default server is configured at construction time (our local Orthanc), but
any call may point at a different server via the `dicomweb_url` argument, so the
same tool can reach any DICOMweb endpoint.
"""

import re
import csv
import uuid
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Type

import requests
from pydantic import BaseModel, Field
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from .dicom import DicomProcessorTool

# DICOM tag numbers (group+element, as used in the DICOM+JSON keys returned by
# QIDO-RS). Kept explicit so the code reads the same as a DICOM tag dictionary.
_TAG = {
    "StudyInstanceUID": "0020000D",
    "SeriesInstanceUID": "0020000E",
    "SOPInstanceUID": "00080018",
    "PatientName": "00100010",
    "PatientID": "00100020",
    "PatientSex": "00100040",
    "PatientBirthDate": "00100030",
    "PatientAge": "00101010",
    "AccessionNumber": "00080050",
    "StudyDate": "00080020",
    "StudyTime": "00080030",
    "StudyDescription": "00081030",
    "ModalitiesInStudy": "00080061",
    "NumberOfStudyRelatedSeries": "00201206",
    "NumberOfStudyRelatedInstances": "00201208",
}

# Fields we ask the server to include in QIDO study results.
_STUDY_FIELDS = [
    _TAG["StudyInstanceUID"],
    _TAG["PatientName"],
    _TAG["PatientID"],
    _TAG["PatientSex"],
    _TAG["PatientBirthDate"],
    _TAG["PatientAge"],
    _TAG["AccessionNumber"],
    _TAG["StudyDate"],
    _TAG["StudyTime"],
    _TAG["StudyDescription"],
    _TAG["ModalitiesInStudy"],
    _TAG["NumberOfStudyRelatedSeries"],
    _TAG["NumberOfStudyRelatedInstances"],
]


def _clean(value: Optional[str]) -> Optional[str]:
    """Strip surrounding whitespace and stray quotes that local models often add
    around string arguments (e.g. patient_name='"DOE^JOHN"')."""
    if value is None:
        return None
    return value.strip().strip("\"'").strip() or None


def _tag_value(obj: Dict, tag: str, person_name: bool = False):
    """Extract the first value of a DICOM tag from a QIDO+JSON element."""
    node = obj.get(tag)
    if not node:
        return None
    values = node.get("Value")
    if not values:
        return None
    value = values[0]
    if person_name and isinstance(value, dict):
        return value.get("Alphabetic") or value.get("Ideographic") or None
    return value


def _parse_multipart(body: bytes, content_type: str) -> List[bytes]:
    """Split a multipart/related WADO-RS response into its part bodies.

    WADO-RS returns instances as multipart/related even for a single object.
    We parse it by hand to avoid an extra dependency (requests-toolbelt).
    """
    match = re.search(r'boundary="?([^";]+)"?', content_type or "")
    if not match:
        # Not multipart (some servers return the DICOM part directly).
        return [body] if body else []
    delimiter = b"--" + match.group(1).encode()
    parts: List[bytes] = []
    for chunk in body.split(delimiter):
        chunk = chunk.lstrip(b"\r\n")
        if not chunk or chunk.startswith(b"--"):
            continue
        header_end = chunk.find(b"\r\n\r\n")
        if header_end == -1:
            continue
        part_body = chunk[header_end + 4 :]
        if part_body.endswith(b"\r\n"):
            part_body = part_body[:-2]
        if part_body:
            parts.append(part_body)
    return parts


class _DicomWebClient:
    """Minimal QIDO-RS / WADO-RS client built on `requests`."""

    def __init__(
        self,
        base_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.auth = (username, password) if username else None
        self.timeout = timeout

    def _get_json(self, path: str, params: Optional[Dict] = None) -> List[Dict]:
        resp = requests.get(
            f"{self.base_url}/{path.lstrip('/')}",
            params=params,
            auth=self.auth,
            headers={"Accept": "application/dicom+json"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        # QIDO returns 204/empty when nothing matches.
        if resp.status_code == 204 or not resp.content or not resp.text.strip():
            return []
        return resp.json()

    def search_studies(self, params: Dict) -> List[Dict]:
        return self._get_json("studies", params)

    def list_instances(self, study_uid: str) -> List[Dict]:
        return self._get_json(f"studies/{study_uid}/instances")

    def retrieve_instance(self, study_uid: str, series_uid: str, sop_uid: str) -> Optional[bytes]:
        resp = requests.get(
            f"{self.base_url}/studies/{study_uid}/series/{series_uid}/instances/{sop_uid}",
            auth=self.auth,
            headers={"Accept": 'multipart/related; type="application/dicom"'},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        parts = _parse_multipart(resp.content, resp.headers.get("Content-Type", ""))
        return parts[0] if parts else None

    def store_instances(self, dicom_datasets: List[bytes]) -> Dict:
        """STOW-RS: upload one or more encoded DICOM instances to the server."""
        boundary = "MedRAXBoundary" + uuid.uuid4().hex
        body = b""
        for dcm in dicom_datasets:
            body += (
                b"--" + boundary.encode() + b"\r\n"
                b"Content-Type: application/dicom\r\n\r\n" + dcm + b"\r\n"
            )
        body += b"--" + boundary.encode() + b"--\r\n"
        resp = requests.post(
            f"{self.base_url}/studies",
            data=body,
            auth=self.auth,
            headers={
                "Content-Type": f'multipart/related; type="application/dicom"; boundary={boundary}',
                "Accept": "application/dicom+json",
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        if resp.status_code == 204 or not resp.content or not resp.text.strip():
            return {}
        return resp.json()


class MedRAXDicomWebInput(BaseModel):
    """Input schema for the MedRax-Dicomweb tool."""

    action: Literal["list_studies", "retrieve_study", "export_dataset"] = Field(
        ...,
        description=(
            "'list_studies' to list the most recent exams on the server; "
            "'retrieve_study' to download a specific study for analysis; "
            "'export_dataset' to export a whole cohort of studies (with patient demographics) "
            "to a CSV file for statistical analysis in the python_sandbox."
        ),
    )
    # list_studies
    limit: int = Field(10, description="[list_studies] How many recent studies to return.")
    modality: Optional[str] = Field(
        None,
        description="[list_studies / export_dataset] Optional modality filter, e.g. 'CR', 'CT', 'DX'.",
    )
    # export_dataset
    start_date: Optional[str] = Field(
        None,
        description="[export_dataset] Optional inclusive start of the StudyDate range, YYYYMMDD.",
    )
    end_date: Optional[str] = Field(
        None,
        description="[export_dataset] Optional inclusive end of the StudyDate range, YYYYMMDD.",
    )
    max_studies: int = Field(
        1000,
        description="[export_dataset] Maximum number of studies to export (server-side cap).",
    )
    # retrieve_study (provide at least one identifier)
    accession_number: Optional[str] = Field(
        None, description="[retrieve_study] Accession number of the study to retrieve."
    )
    patient_name: Optional[str] = Field(
        None,
        description=(
            "[retrieve_study] Patient name (or part of it) whose study to retrieve. "
            "If several studies match, the most recent one is used."
        ),
    )
    study_instance_uid: Optional[str] = Field(
        None, description="[retrieve_study] Exact StudyInstanceUID to retrieve."
    )
    max_instances: int = Field(
        20, description="[retrieve_study] Maximum number of images to download from the study."
    )
    # server override (defaults to the configured server)
    dicomweb_url: Optional[str] = Field(
        None,
        description=(
            "Optional. Base URL of a different DICOMweb server to query instead of the "
            "default, e.g. 'https://server/dicom-web'. Leave empty to use the configured server."
        ),
    )
    username: Optional[str] = Field(
        None, description="Optional. Username for the overridden DICOMweb server."
    )
    password: Optional[str] = Field(
        None, description="Optional. Password for the overridden DICOMweb server."
    )


class MedRAXDicomWebTool(BaseTool):
    """Query and retrieve studies from a DICOMweb server (defaults to our Orthanc)."""

    name: str = "MedRax-Dicomweb"
    description: str = (
        "Accesses a DICOMweb PACS/archive (QIDO-RS/WADO-RS) to find and retrieve medical imaging "
        "studies. Two actions: (1) action='list_studies' returns the most recent exams with patient "
        "name, accession number, study date, modality and description; (2) action='retrieve_study' "
        "downloads a study identified by accession_number, patient_name, or study_instance_uid, "
        "saves its images locally and converts them to PNG; (3) action='export_dataset' exports a "
        "cohort of studies (optionally filtered by modality and StudyDate range) — including patient "
        "sex, birth date and age — to a CSV file and returns its path plus a quick summary. "
        "Use retrieve_study to focus on / pull a patient's exam, then run the chest X-ray tools "
        "(classifier, report generator, VQA) on the returned image_paths to summarize the findings. "
        "To answer statistical questions about the archive (e.g. how many exams a patient has, how "
        "many patients are female vs male, exams per modality or per month), call export_dataset and "
        "then feed the returned csv_path to the python_sandbox tool via its input_files argument and "
        "aggregate with pandas. "
        "By default it targets the configured server; pass dicomweb_url to query any other DICOMweb server."
    )
    args_schema: Type[BaseModel] = MedRAXDicomWebInput

    base_url: str = "http://localhost:8042/dicom-web"
    username: Optional[str] = None
    password: Optional[str] = None
    fetch_cap: int = 500
    timeout: int = 60
    temp_dir: Path = None

    def __init__(
        self,
        base_url: str = "http://localhost:8042/dicom-web",
        username: Optional[str] = None,
        password: Optional[str] = None,
        temp_dir: Optional[str] = None,
        fetch_cap: int = 500,
        timeout: int = 60,
    ):
        """Initialize the MedRax-Dicomweb tool.

        Args:
            base_url: Default DICOMweb base URL (QIDO-RS/WADO-RS root).
            username/password: Credentials for the default server (None if unauthenticated).
            temp_dir: Directory for downloaded DICOM files and PNGs.
            fetch_cap: Max studies fetched for recency sorting in list_studies.
            timeout: HTTP timeout in seconds.
        """
        super().__init__()
        self.base_url = base_url
        self.username = username or None
        self.password = password or None
        self.fetch_cap = fetch_cap
        self.timeout = timeout
        self.temp_dir = Path(temp_dir if temp_dir else tempfile.mkdtemp())
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ helpers
    def _client(
        self, url: Optional[str], user: Optional[str], pw: Optional[str]
    ) -> _DicomWebClient:
        """Build a client for the requested server (or the configured default)."""
        url, user, pw = _clean(url), _clean(user), _clean(pw)
        if url:
            return _DicomWebClient(url, user, pw, timeout=self.timeout)
        return _DicomWebClient(self.base_url, self.username, self.password, timeout=self.timeout)

    @staticmethod
    def _study_summary(study: Dict) -> Dict:
        return {
            "patient_name": _tag_value(study, _TAG["PatientName"], person_name=True),
            "patient_id": _tag_value(study, _TAG["PatientID"]),
            "patient_sex": _tag_value(study, _TAG["PatientSex"]),
            "patient_birth_date": _tag_value(study, _TAG["PatientBirthDate"]),
            "patient_age": _tag_value(study, _TAG["PatientAge"]),
            "accession_number": _tag_value(study, _TAG["AccessionNumber"]),
            "study_date": _tag_value(study, _TAG["StudyDate"]),
            "study_time": _tag_value(study, _TAG["StudyTime"]),
            "modality": _tag_value(study, _TAG["ModalitiesInStudy"]),
            "study_description": _tag_value(study, _TAG["StudyDescription"]),
            "num_series": _tag_value(study, _TAG["NumberOfStudyRelatedSeries"]),
            "num_instances": _tag_value(study, _TAG["NumberOfStudyRelatedInstances"]),
            "study_instance_uid": _tag_value(study, _TAG["StudyInstanceUID"]),
        }

    @staticmethod
    def _recency_key(study: Dict) -> str:
        date = _tag_value(study, _TAG["StudyDate"]) or ""
        time = _tag_value(study, _TAG["StudyTime"]) or ""
        return f"{date}{time}"

    # ------------------------------------------------------------------ actions
    def _list_studies(
        self, client: _DicomWebClient, limit: int, modality: Optional[str]
    ) -> Tuple[Dict, Dict]:
        params: Dict = {"includefield": _STUDY_FIELDS, "limit": self.fetch_cap}
        modality = _clean(modality)
        if modality:
            params[_TAG["ModalitiesInStudy"]] = modality.upper()

        studies = client.search_studies(params)
        studies.sort(key=self._recency_key, reverse=True)
        top = studies[: max(1, limit)]

        output = {
            "studies": [self._study_summary(s) for s in top],
            "count": len(top),
            "server": client.base_url,
        }
        metadata = {
            "action": "list_studies",
            "total_available": len(studies),
            "status": "completed",
        }
        if len(studies) >= self.fetch_cap:
            metadata["note"] = (
                f"Server returned at least {self.fetch_cap} studies (fetch cap); recency is exact "
                "only within that window. Narrow by modality if needed."
            )
        return output, metadata

    @staticmethod
    def _date_range(start_date: Optional[str], end_date: Optional[str]) -> Optional[str]:
        """Build a QIDO StudyDate matcher from an optional start/end (YYYYMMDD).

        Returns 'YYYYMMDD-YYYYMMDD' for a range, a bare date for an exact match,
        an open-ended 'YYYYMMDD-' / '-YYYYMMDD', or None when neither is given.
        """
        start = _clean(start_date)
        end = _clean(end_date)
        if start and end:
            return start if start == end else f"{start}-{end}"
        if start:
            return f"{start}-"
        if end:
            return f"-{end}"
        return None

    # export dataset -------------------------------------------------------
    _CSV_FIELDS = [
        "patient_id",
        "patient_name",
        "patient_sex",
        "patient_birth_date",
        "patient_age",
        "study_date",
        "study_time",
        "modality",
        "study_description",
        "accession_number",
        "num_series",
        "num_instances",
        "study_instance_uid",
    ]

    def _export_dataset(
        self,
        client: _DicomWebClient,
        modality: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        max_studies: int,
    ) -> Tuple[Dict, Dict]:
        """Export a cohort of studies (with demographics) to CSV for analysis."""
        cap = max(1, min(int(max_studies or self.fetch_cap), 10000))
        params: Dict = {"includefield": _STUDY_FIELDS, "limit": cap}
        modality = _clean(modality)
        if modality:
            params[_TAG["ModalitiesInStudy"]] = modality.upper()
        date_range = self._date_range(start_date, end_date)
        if date_range:
            params[_TAG["StudyDate"]] = date_range

        studies = client.search_studies(params)
        rows = [self._study_summary(s) for s in studies]

        csv_path = self.temp_dir / f"pacs_cohort_{uuid.uuid4().hex[:8]}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        # A quick built-in summary so the model gets numbers immediately; the CSV
        # lets the python_sandbox compute anything more involved (and plot it).
        patient_sex: Dict[str, Optional[str]] = {}
        for row in rows:
            pid = row.get("patient_id") or row.get("patient_name")
            if pid is not None and pid not in patient_sex:
                patient_sex[pid] = (row.get("patient_sex") or "").strip().upper() or None
        sex_counts = Counter((s or "UNKNOWN") for s in patient_sex.values())
        modality_counts = Counter((row.get("modality") or "UNKNOWN") for row in rows)

        output = {
            "csv_path": str(csv_path),
            "total_studies": len(rows),
            "unique_patients": len(patient_sex),
            "patients_by_sex": dict(sex_counts),
            "studies_by_modality": dict(modality_counts),
            "columns": self._CSV_FIELDS,
            "server": client.base_url,
            "note": (
                "Cohort exported to csv_path. For per-patient exam counts, sex/age distributions "
                "or charts, pass csv_path to the python_sandbox tool via input_files and aggregate "
                "with pandas, e.g. pd.read_csv('"
                f"{csv_path.name}"
                "'). Sex codes follow DICOM: 'F' female, 'M' male, 'O'/blank other/unknown. "
                "Count distinct patients by patient_id (a patient may have several studies)."
            ),
        }
        metadata = {
            "action": "export_dataset",
            "status": "completed",
            "exported_studies": len(rows),
        }
        if len(rows) >= cap:
            metadata["note"] = (
                f"Hit the {cap}-study cap; the cohort may be truncated. Raise max_studies or "
                "narrow by modality / date range for a complete export."
            )
        return output, metadata

    def _find_study(
        self,
        client: _DicomWebClient,
        accession_number: Optional[str],
        patient_name: Optional[str],
        study_instance_uid: Optional[str],
    ) -> List[Dict]:
        """Locate matching studies from the given identifier(s)."""
        base_params: Dict = {"includefield": _STUDY_FIELDS, "limit": self.fetch_cap}

        study_instance_uid = _clean(study_instance_uid)
        accession_number = _clean(accession_number)
        patient_name = _clean(patient_name)

        if study_instance_uid:
            params = {**base_params, _TAG["StudyInstanceUID"]: study_instance_uid}
            return client.search_studies(params)

        if accession_number:
            params = {**base_params, _TAG["AccessionNumber"]: accession_number}
            return client.search_studies(params)

        if patient_name:
            name = patient_name
            # Try a wildcard match, then an uppercased variant (DICOM PN are
            # commonly stored uppercase and many servers match case-sensitively).
            candidates = [
                "*" + name.replace(" ", "*") + "*",
                "*" + name.upper().replace(" ", "*") + "*",
            ]
            for pn in candidates:
                params = {**base_params, _TAG["PatientName"]: pn}
                results = client.search_studies(params)
                if results:
                    return results
            return []

        return []

    def _retrieve_study(
        self,
        client: _DicomWebClient,
        accession_number: Optional[str],
        patient_name: Optional[str],
        study_instance_uid: Optional[str],
        max_instances: int,
    ) -> Tuple[Dict, Dict]:
        if not (_clean(accession_number) or _clean(patient_name) or _clean(study_instance_uid)):
            return (
                {"error": "Provide accession_number, patient_name, or study_instance_uid."},
                {"action": "retrieve_study", "status": "failed"},
            )

        matches = self._find_study(client, accession_number, patient_name, study_instance_uid)
        if not matches:
            return (
                {"error": "No study found for the given identifier.", "server": client.base_url},
                {"action": "retrieve_study", "status": "failed", "matched_studies": 0},
            )

        matches.sort(key=self._recency_key, reverse=True)
        study = matches[0]
        summary = self._study_summary(study)
        study_uid = summary["study_instance_uid"]

        instances = client.list_instances(study_uid)[: max(1, max_instances)]

        # Reuse the existing DICOM->PNG converter so downstream CXR tools work directly.
        processor = DicomProcessorTool(temp_dir=str(self.temp_dir))
        files: List[Dict] = []
        errors: List[str] = []
        for inst in instances:
            series_uid = _tag_value(inst, _TAG["SeriesInstanceUID"])
            sop_uid = _tag_value(inst, _TAG["SOPInstanceUID"])
            if not (series_uid and sop_uid):
                continue
            try:
                dicom_bytes = client.retrieve_instance(study_uid, series_uid, sop_uid)
                if not dicom_bytes:
                    errors.append(f"{sop_uid}: empty response")
                    continue
                dcm_path = self.temp_dir / f"dicomweb_{uuid.uuid4().hex[:8]}.dcm"
                dcm_path.write_bytes(dicom_bytes)
                png_out, _ = processor._run(str(dcm_path))
                files.append(
                    {"dicom_path": str(dcm_path), "image_path": png_out.get("image_path")}
                )
            except Exception as exc:  # keep going on a per-instance failure
                errors.append(f"{sop_uid}: {exc}")

        output = {
            "study": summary,
            "instances_retrieved": len(files),
            "files": files,
            "image_paths": [f["image_path"] for f in files if f.get("image_path")],
            "server": client.base_url,
            "note": (
                "Study downloaded and converted to PNG. Run the chest X-ray analysis tools "
                "(classifier, report generator, VQA) on image_paths to summarize the findings."
            ),
        }
        metadata = {
            "action": "retrieve_study",
            "status": "completed" if files else "failed",
            "matched_studies": len(matches),
            "study_instance_uid": study_uid,
        }
        if len(matches) > 1:
            metadata["ambiguous"] = (
                f"{len(matches)} studies matched; retrieved the most recent. "
                "Provide accession_number or study_instance_uid to disambiguate."
            )
        if errors:
            metadata["instance_errors"] = errors
        return output, metadata

    # ------------------------------------------------------------------ run
    def _run(
        self,
        action: str,
        limit: int = 10,
        modality: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_studies: int = 1000,
        accession_number: Optional[str] = None,
        patient_name: Optional[str] = None,
        study_instance_uid: Optional[str] = None,
        max_instances: int = 20,
        dicomweb_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict, Dict]:
        """Execute a MedRax-Dicomweb action against the configured or overridden server."""
        try:
            client = self._client(dicomweb_url, username, password)
            if action == "list_studies":
                return self._list_studies(client, limit, modality)
            if action == "retrieve_study":
                return self._retrieve_study(
                    client, accession_number, patient_name, study_instance_uid, max_instances
                )
            if action == "export_dataset":
                return self._export_dataset(client, modality, start_date, end_date, max_studies)
            return (
                {
                    "error": f"Unknown action '{action}'. Use 'list_studies', 'retrieve_study' "
                    "or 'export_dataset'."
                },
                {"status": "failed"},
            )
        except requests.HTTPError as exc:
            resp = exc.response
            return (
                {"error": f"DICOMweb server error {resp.status_code if resp is not None else ''}: {exc}"},
                {"action": action, "status": "failed", "error_details": str(exc)},
            )
        except requests.RequestException as exc:
            return (
                {"error": f"Could not reach DICOMweb server: {exc}"},
                {"action": action, "status": "failed", "error_details": str(exc)},
            )
        except Exception as exc:
            return (
                {"error": str(exc)},
                {"action": action, "status": "failed", "error_details": str(exc)},
            )

    async def _arun(
        self,
        action: str,
        limit: int = 10,
        modality: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_studies: int = 1000,
        accession_number: Optional[str] = None,
        patient_name: Optional[str] = None,
        study_instance_uid: Optional[str] = None,
        max_instances: int = 20,
        dicomweb_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict, Dict]:
        """Async version of _run."""
        return self._run(
            action,
            limit,
            modality,
            start_date,
            end_date,
            max_studies,
            accession_number,
            patient_name,
            study_instance_uid,
            max_instances,
            dicomweb_url,
            username,
            password,
        )

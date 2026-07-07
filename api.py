"""
FastAPI wrapper around the MedRAX agent.

Exposes the same agent that `main.py` serves through Gradio, but as a plain HTTP
API so it can be driven from OHIF (or any other frontend).

Run:
    cd /mnt/dados2/MedRAX
    conda activate MedRax
    uvicorn api:app --host 0.0.0.0 --port 8686

Endpoints
    GET  /health                      -> liveness + loaded tools
    POST /api/upload   (multipart)    -> store an image / DICOM, return an image_id
    POST /api/chat     (json)         -> ask the agent, get the full answer back
    POST /api/chat/stream (json)      -> same, streamed as Server-Sent Events
    GET  /api/image/{name}            -> download an image produced by a tool
    POST /api/reset    (json)         -> start a fresh conversation thread
"""

import os
import re
import json
import time
import uuid
import base64
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from main import initialize_agent

# Load .env before reading any OPENAI_* / MEDRAX_* variables below.
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("medrax.api")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (mirrors main.py; override via env)
# ──────────────────────────────────────────────────────────────────────────────

MODEL_DIR = os.getenv("MEDRAX_MODEL_DIR", "/mnt/dados2/MedRAX/downloadable_content/model_weights")
TEMP_DIR = Path(os.getenv("MEDRAX_TEMP_DIR", "temp"))
DEVICE = os.getenv("MEDRAX_DEVICE", "cuda")
MODEL = os.getenv("MEDRAX_LLM", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
PROMPT_FILE = os.getenv("MEDRAX_PROMPT_FILE", "medrax/docs/system_prompts.txt")

# The chat LLM endpoint the agent talks to (an OpenAI-compatible server:
# llama.cpp / vLLM / Ollama / etc.). If this is unreachable, every /api/chat
# fails with APIConnectionError — surfaced clearly by the error handling below.
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1 (default)")

SELECTED_TOOLS = [
    "ImageVisualizerTool",
    "DicomProcessorTool",
    "MedRAXDicomWebTool",
    "MedRAXReportToPacsTool",
    "ChestXRayClassifierTool",
    "ChestXRaySegmentationTool",
    "CardiothoracicRatioTool",
    "ChestXRayReportGeneratorTool",
    "XRayVQATool",
    "LlavaMedTool",
    "XRayPhraseGroundingTool",
    "WebSearchTool",
    "PythonSandboxTool",
    # "MedGemmaTool",          # disabled: no spare GPU memory for it
]

TEMP_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# App + agent bootstrap
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="MedRAX API", version="1.0")

# OHIF runs in the browser, so it calls us cross-origin. Lock this down in prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("MEDRAX_CORS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AGENT = None
TOOLS = None
# The models sit on a single GPU and langgraph state is not concurrency-safe,
# so we serialize every agent invocation behind one lock.
_AGENT_LOCK = asyncio.Lock()


@app.on_event("startup")
def _load_agent() -> None:
    global AGENT, TOOLS
    openai_kwargs = {}
    if api_key := os.getenv("OPENAI_API_KEY"):
        openai_kwargs["api_key"] = api_key
    if base_url := os.getenv("OPENAI_BASE_URL"):
        openai_kwargs["base_url"] = base_url

    AGENT, TOOLS = initialize_agent(
        PROMPT_FILE,
        tools_to_use=SELECTED_TOOLS,
        model_dir=MODEL_DIR,
        temp_dir=str(TEMP_DIR),
        device=DEVICE,
        model=MODEL,
        temperature=0.7,
        top_p=0.95,
        prompt_key="MEDICAL_ASSISTANT",
        openai_kwargs=openai_kwargs,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Request / response models
# ──────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    image_id: Optional[str] = None        # returned by /api/upload
    image_b64: Optional[str] = None       # or send pixels inline (e.g. OHIF canvas PNG)
    thread_id: Optional[str] = None       # keep it to continue the conversation


class UploadResponse(BaseModel):
    image_id: str          # path handed to the agent as `image_path:`
    display_url: str       # viewable PNG URL (for showing back in OHIF)
    is_dicom: bool


class ChatResponse(BaseModel):
    thread_id: str
    reply: str
    images: List[str]      # URLs of images produced by tools (seg masks, grounding, ...)
    tool_calls: List[Dict[str, Any]]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_IMG_RE = re.compile(r"[\w./\-]+\.(?:png|jpe?g)", re.IGNORECASE)


def _save_bytes(data: bytes, suffix: str) -> Path:
    dest = TEMP_DIR / f"upload_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}{suffix}"
    dest.write_bytes(data)
    return dest


def _ingest(raw: bytes, suffix: str) -> UploadResponse:
    """Persist an incoming image/DICOM and return an id + a viewable PNG url.

    DICOM is converted to PNG via the agent's own DicomProcessorTool so the
    vision model and the browser both get something they can render.
    """
    saved = _save_bytes(raw, suffix.lower())
    is_dicom = suffix.lower() == ".dcm"

    if is_dicom:
        output, meta = TOOLS["DicomProcessorTool"]._run(str(saved))
        if "image_path" not in output:
            raise HTTPException(
                422, f"Could not process DICOM: {output.get('error', 'unknown error')}"
            )
        display = Path(output["image_path"])
    else:
        display = saved

    # We hand the agent the *viewable* file for both tools and vision. Most
    # MedRAX CXR tools expect a standard image, and the vision LLM can only read
    # a real image (not raw DICOM bytes).
    return UploadResponse(
        image_id=str(display),
        display_url=f"/api/image/{display.name}",
        is_dicom=is_dicom,
    )


def _build_messages(req: ChatRequest) -> List[dict]:
    """Assemble the single multimodal user turn the agent expects."""
    content: List[dict] = []
    image_path = None

    if req.image_id:
        image_path = req.image_id
    elif req.image_b64:
        b64 = req.image_b64.split(",", 1)[-1]  # tolerate a data: prefix
        saved = _save_bytes(base64.b64decode(b64), ".png")
        image_path = str(saved)

    if image_path:
        # Text part → lets the tools open the file on disk.
        content.append({"type": "text", "text": f"image_path: {image_path}"})
        # Image part → lets the vision LLM actually see it.
        with open(image_path, "rb") as f:
            enc = base64.b64encode(f.read()).decode()
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{enc}"}}
        )

    content.append({"type": "text", "text": req.message})
    return [{"role": "user", "content": content}]


def _extract_images(tool_content: str) -> List[str]:
    """Pull any image file paths out of a tool result string."""
    paths: List[str] = []
    try:  # tool results are stringified python, sometimes containing numpy
        obj = eval(tool_content, {"np": np, "array": np.array})  # noqa: S307
        stack = [obj]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                if "image_path" in cur and cur["image_path"]:
                    paths.append(str(cur["image_path"]))
                stack.extend(cur.values())
            elif isinstance(cur, (list, tuple)):
                stack.extend(cur)
    except Exception:
        paths = _IMG_RE.findall(tool_content or "")
    # de-dupe, keep only files that exist, return as URLs
    seen, urls = set(), []
    for p in paths:
        name = Path(p).name
        if name not in seen and (TEMP_DIR / name).exists():
            seen.add(name)
            urls.append(f"/api/image/{name}")
    return urls


def _run_agent(messages: List[dict], thread_id: str):
    """Blocking generator over agent events (assistant text + tool calls)."""
    config = {"configurable": {"thread_id": thread_id}}
    for event in AGENT.workflow.stream({"messages": messages}, config):
        if not isinstance(event, dict):
            continue
        if "process" in event:
            msg = event["process"]["messages"][-1]
            text = re.sub(r"temp/[^\s]*", "", msg.content or "").strip()
            if text:
                yield {"type": "assistant", "text": text}
        elif "execute" in event:
            for m in event["execute"]["messages"]:
                yield {
                    "type": "tool",
                    "name": m.name,
                    "args": m.args,
                    "images": _extract_images(m.content),
                }


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "tools": list(TOOLS.keys()) if TOOLS else []}


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix or ".png"
    raw = await file.read()
    return _ingest(raw, suffix)


@app.get("/api/image/{name}")
def get_image(name: str):
    path = TEMP_DIR / Path(name).name  # basename only — no path traversal
    if not path.exists():
        raise HTTPException(404, "image not found")
    return FileResponse(path)


@app.post("/api/reset")
def reset():
    return {"thread_id": uuid.uuid4().hex}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    thread_id = req.thread_id or uuid.uuid4().hex
    messages = _build_messages(req)

    reply_parts: List[str] = []
    images: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    async with _AGENT_LOCK:
        loop = asyncio.get_event_loop()
        try:
            # Drain the blocking generator in a worker thread.
            events = await loop.run_in_executor(
                None, lambda: list(_run_agent(messages, thread_id))
            )
        except Exception as exc:  # LLM timeout, context overflow, server error, ...
            log.exception("agent run failed for thread %s", thread_id)
            return ChatResponse(
                thread_id=thread_id,
                reply=f"The request could not be completed: {exc}",
                images=[],
                tool_calls=[],
            )

    for ev in events:
        if ev["type"] == "assistant":
            reply_parts.append(ev["text"])
        else:
            tool_calls.append({"name": ev["name"], "args": ev["args"]})
            images.extend(ev["images"])

    return ChatResponse(
        thread_id=thread_id,
        reply="\n\n".join(reply_parts).strip() or "(no answer)",
        images=list(dict.fromkeys(images)),
        tool_calls=tool_calls,
    )


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """Server-Sent Events: emit each assistant chunk / tool call as it happens."""
    import json

    thread_id = req.thread_id or uuid.uuid4().hex
    messages = _build_messages(req)

    async def gen():
        yield f"data: {json.dumps({'type': 'thread', 'thread_id': thread_id})}\n\n"
        async with _AGENT_LOCK:
            loop = asyncio.get_event_loop()
            q: asyncio.Queue = asyncio.Queue()

            def produce():
                # Always deliver the None sentinel — even on error — or the consumer
                # below would await q.get() forever and never release the lock.
                try:
                    for ev in _run_agent(messages, thread_id):
                        loop.call_soon_threadsafe(q.put_nowait, ev)
                except Exception as exc:  # LLM timeout, context overflow, server error, ...
                    log.exception("agent stream failed for thread %s", thread_id)
                    loop.call_soon_threadsafe(
                        q.put_nowait,
                        {"type": "error", "message": f"The request could not be completed: {exc}"},
                    )
                finally:
                    loop.call_soon_threadsafe(q.put_nowait, None)

            loop.run_in_executor(None, produce)
            while True:
                ev = await q.get()
                if ev is None:
                    break
                yield f"data: {json.dumps(ev)}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ──────────────────────────────────────────────────────────────────────────────
# Option A: also serve the original Gradio UI from THIS process, so the UI and
# the REST API (OHIF) share one in-memory agent / one GPU copy of the models.
# Enable with MEDRAX_MOUNT_GRADIO=1. Then the UI is at http://host:8686/gradio
# ──────────────────────────────────────────────────────────────────────────────

if os.getenv("MEDRAX_MOUNT_GRADIO") == "1":
    import gradio as gr
    from interface import create_demo

    @app.on_event("startup")
    def _mount_gradio() -> None:
        demo = create_demo(AGENT, TOOLS)  # same AGENT/TOOLS built at startup
        gr.mount_gradio_app(app, demo, path="/gradio")

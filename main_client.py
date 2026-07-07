"""
Thin Gradio client for MedRAX.

Unlike main.py, this does NOT load any models — it is a lightweight front end
that talks to the FastAPI backend (api.py) over HTTP. Run api.py once (it owns
the GPU / models); then OHIF and this Gradio UI both hit that single backend, so
they share the same resources.

Run:
    # 1) start the backend (loads the models, one GPU copy)
    uvicorn api:app --host 0.0.0.0 --port 8686
    # 2) start this UI (no GPU needed; can even run on another machine)
    MEDRAX_API=http://localhost:8686 python main_client.py
"""

import os
import time
import tempfile
from pathlib import Path

import requests
import gradio as gr
from gradio import ChatMessage

API = os.getenv("MEDRAX_API", "http://localhost:8686").rstrip("/")
CACHE = Path(tempfile.gettempdir()) / "medrax_client"
CACHE.mkdir(exist_ok=True)


class APIClient:
    """Holds per-session state and forwards everything to the REST API."""

    def __init__(self):
        self.thread_id = None
        self.image_id = None          # server-side path returned by /api/upload
        self.display_path = None      # local copy of the viewable image (for the UI)
        self.pending_upload = None    # image the user JUST uploaded, to show inline once

    def _download(self, url_path: str) -> str:
        """Fetch an image the backend produced and cache it locally for display."""
        r = requests.get(f"{API}{url_path}", timeout=60)
        r.raise_for_status()
        local = CACHE / f"{int(time.time()*1000)}_{Path(url_path).name}"
        local.write_bytes(r.content)
        return str(local)

    def handle_upload(self, file_path: str):
        if not file_path:
            return None
        with open(file_path, "rb") as f:
            resp = requests.post(
                f"{API}/api/upload",
                files={"file": (Path(file_path).name, f)},
                timeout=120,
            ).json()
        self.image_id = resp["image_id"]
        self.display_path = self._download(resp["display_url"])
        self.pending_upload = self.display_path  # show this upload inline on the next message
        return self.display_path

    def add_message(self, message, display_image, history):
        history = history or []
        # Attach an image to the user's message ONLY if they just uploaded one this
        # turn. Do NOT attach self.display_path: that persists across turns and holds
        # the last image the agent produced (e.g. a previous patient's result), which
        # is what caused a stale image to be stapled onto every new question. The
        # current image is always visible in the image panel on the right anyway.
        if self.pending_upload:
            history.append(ChatMessage(role="user", content={"path": self.pending_upload}))
            self.pending_upload = None
        if message:
            history.append(ChatMessage(role="user", content=message))
        return history, gr.Textbox(value=message, interactive=False)

    def process_message(self, message, display_image, history):
        history = history or []
        try:
            resp = requests.post(
                f"{API}/api/chat",
                json={
                    "message": message,
                    "image_id": self.image_id,
                    "thread_id": self.thread_id,
                },
                timeout=600,
            ).json()

            self.thread_id = resp.get("thread_id", self.thread_id)

            for name in {tc["name"] for tc in resp.get("tool_calls", [])}:
                history.append(
                    ChatMessage(
                        role="assistant",
                        content=f"🔧 used tool: {name}",
                        metadata={"title": f"Tool: {name}"},
                    )
                )

            if resp.get("reply"):
                history.append(ChatMessage(role="assistant", content=resp["reply"]))

            new_display = self.display_path
            for img_url in resp.get("images", []):
                local = self._download(img_url)
                new_display = local
                history.append(ChatMessage(role="assistant", content={"path": local}))

            self.display_path = new_display
            yield history, new_display, ""
        except Exception as e:
            history.append(
                ChatMessage(role="assistant", content=f"❌ Error: {e}", metadata={"title": "Error"})
            )
            yield history, self.display_path, ""


def create_client_demo():
    client = APIClient()

    with gr.Blocks() as demo:
        gr.Markdown("# 🏥 MedRAX (API client)\nMedical Reasoning Agent for Chest X-ray")

        with gr.Row():
            with gr.Column(scale=3):
                # Gradio 6 dropped the `type` arg (messages is the only format);
                # Gradio 5 needs type="messages" to accept ChatMessage objects.
                try:
                    chatbot = gr.Chatbot([], height=800, type="messages", label="Agent")
                except TypeError:
                    chatbot = gr.Chatbot([], height=800, label="Agent")
                txt = gr.Textbox(
                    show_label=False, placeholder="Ask about the X-ray...", container=False
                )
            with gr.Column(scale=3):
                image_display = gr.Image(label="Image", type="filepath", height=700)
                with gr.Row():
                    upload_button = gr.UploadButton("📎 Upload X-Ray", file_types=["image"])
                    dicom_upload = gr.UploadButton("📄 Upload DICOM", file_types=["file"])
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat")
                    new_thread_btn = gr.Button("New Thread")

        def clear_chat():
            client.image_id = None
            client.display_path = None
            client.pending_upload = None
            return [], None

        def new_thread():
            resp = requests.post(f"{API}/api/reset", timeout=30).json()
            client.thread_id = resp["thread_id"]
            return [], client.display_path

        chat_msg = txt.submit(
            client.add_message, inputs=[txt, image_display, chatbot], outputs=[chatbot, txt]
        )
        bot_msg = chat_msg.then(
            client.process_message,
            inputs=[txt, image_display, chatbot],
            outputs=[chatbot, image_display, txt],
        )
        bot_msg.then(lambda: gr.Textbox(interactive=True), None, [txt])

        upload_button.upload(lambda f: client.handle_upload(f.name), upload_button, image_display)
        dicom_upload.upload(lambda f: client.handle_upload(f.name), dicom_upload, image_display)
        clear_btn.click(clear_chat, outputs=[chatbot, image_display])
        new_thread_btn.click(new_thread, outputs=[chatbot, image_display])

    return demo


if __name__ == "__main__":
    print(f"Connecting to MedRAX API at {API}")
    # share=True downloads the frpc tunnel binary and triggers the api_info
    # schema builder (buggy in some gradio_client versions); keep it opt-in.
    kwargs = dict(
        server_name="0.0.0.0",
        server_port=8585,
        share=os.getenv("MEDRAX_SHARE", "0") == "1",
    )
    demo = create_client_demo()
    try:
        demo.launch(show_api=False, **kwargs)  # Gradio 5 accepts show_api
    except TypeError:
        demo.launch(**kwargs)                  # Gradio 6 dropped it

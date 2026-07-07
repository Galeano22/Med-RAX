import os
import base64
import mimetypes
from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
from pydantic import BaseModel, Field

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from openai import OpenAI


class MedGemmaToolInput(BaseModel):
    """Input schema for the MedGemma multimodal tool."""

    image_paths: List[str] = Field(
        ..., description="List of paths to medical images to analyze"
    )
    prompt: str = Field(
        ..., description="Question or instruction about the medical images"
    )
    max_new_tokens: int = Field(
        512, description="Maximum number of tokens to generate in the response"
    )


class MedGemmaTool(BaseTool):
    """Tool that queries MedGemma, a multimodal medical model, over an OpenAI-compatible endpoint.

    MedGemma can directly interpret medical images (e.g. chest X-rays) and answer
    questions about them. It complements the specialist tools by providing a second,
    vision-grounded opinion. Treat its output as a hypothesis to corroborate against
    the validated specialist tools, not as a final diagnosis.
    """

    name: str = "medgemma_image_opinion"
    description: str = (
        "Ask MedGemma, a multimodal medical AI, to directly look at one or more medical "
        "images and give its opinion. Useful for a vision-grounded second opinion, "
        "open-ended visual questions, describing findings, or sanity-checking other tools' "
        "outputs against the actual image. Input is image paths and a natural language prompt. "
        "Its reading should be corroborated by the specialized diagnostic tools."
    )
    args_schema: Type[BaseModel] = MedGemmaToolInput
    return_direct: bool = True

    client: Optional[Any] = None
    model: Optional[str] = None
    temperature: float = 0.2
    system_prompt: Optional[str] = None

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        system_prompt: Optional[str] = (
            "You are MedGemma, an expert medical AI that analyzes medical images. "
            "Describe what you actually observe in the image, answer the question, and be "
            "explicit about uncertainty. Do not invent findings you cannot see."
        ),
        **kwargs: Any,
    ) -> None:
        """Initialize the MedGemmaTool.

        Args:
            model: MedGemma model name as exposed by the serving endpoint.
                Falls back to the MEDGEMMA_MODEL env var.
            base_url: Base URL of the OpenAI-compatible endpoint serving MedGemma.
                Falls back to the MEDGEMMA_BASE_URL env var.
            api_key: API key for the endpoint. Falls back to the MEDGEMMA_API_KEY env
                var, then to "EMPTY" (common for local servers that ignore the key).
            temperature: Sampling temperature. Kept low for deterministic medical output.
            system_prompt: System instruction sent with every request.
            **kwargs: Additional arguments passed to BaseTool.
        """
        super().__init__(**kwargs)

        self.model = model or os.getenv("MEDGEMMA_MODEL", "google/medgemma-4b-it")
        self.temperature = temperature
        self.system_prompt = system_prompt

        resolved_base_url = base_url or os.getenv("MEDGEMMA_BASE_URL")
        resolved_api_key = api_key or os.getenv("MEDGEMMA_API_KEY") or "EMPTY"

        self.client = OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)

    def _encode_image(self, path: str) -> str:
        """Read an image from disk and return a base64 data URI."""
        mime, _ = mimetypes.guess_type(path)
        if mime is None or not mime.startswith("image"):
            mime = "image/png"
        with open(path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"

    def _generate_response(
        self, image_paths: List[str], prompt: str, max_new_tokens: int
    ) -> str:
        """Query MedGemma with the given images and prompt.

        Args:
            image_paths: List of paths to medical images.
            prompt: Question or instruction about the images.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            str: MedGemma's textual response.
        """
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            content.append(
                {"type": "image_url", "image_url": {"url": self._encode_image(path)}}
            )

        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": content})

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_new_tokens,
        )
        return completion.choices[0].message.content

    def _run(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int = 512,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Execute the MedGemma image analysis.

        Args:
            image_paths: List of paths to medical images.
            prompt: Question or instruction about the images.
            max_new_tokens: Maximum number of tokens to generate.
            run_manager: Optional callback manager.

        Returns:
            Tuple[Dict[str, Any], Dict]: Output dictionary and metadata dictionary.
        """
        try:
            # Local models (e.g. Qwen GGUF) sometimes emit path arguments wrapped in
            # literal quotes, which breaks file access. Strip surrounding quotes/space.
            image_paths = [p.strip().strip("\"'").strip() for p in image_paths]

            for path in image_paths:
                if not Path(path).is_file():
                    raise FileNotFoundError(f"Image file not found: {path}")

            response = self._generate_response(image_paths, prompt, max_new_tokens)

            output = {"response": response}
            metadata = {
                "image_paths": image_paths,
                "prompt": prompt,
                "model": self.model,
                "max_new_tokens": max_new_tokens,
                "analysis_status": "completed",
            }
            return output, metadata

        except Exception as e:
            output = {"error": str(e)}
            metadata = {
                "image_paths": image_paths,
                "prompt": prompt,
                "model": self.model,
                "max_new_tokens": max_new_tokens,
                "analysis_status": "failed",
                "error_details": str(e),
            }
            return output, metadata

    async def _arun(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int = 512,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Async version of _run."""
        return self._run(image_paths, prompt, max_new_tokens)

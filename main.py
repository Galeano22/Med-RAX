import os
import warnings
from typing import *
from dotenv import load_dotenv
from transformers import logging

from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from interface import create_demo
from medrax.agent import *
from medrax.tools import *
from medrax.utils import *

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
_ = load_dotenv()


def initialize_agent(
    prompt_file,
    tools_to_use=None,
    model_dir="/mnt/dados2/MedRAX/downloadable_content/model_weights",
    temp_dir="/mnt/dados2/MedRAX/downloadable_content/temp",
    device="cuda",
    model="chatgpt-4o-latest",
    temperature=0.7,
    top_p=0.95,
    prompt_key="MEDICAL_ASSISTANT",
    openai_kwargs={}
):
    """Initialize the MedRAX agent with specified tools and configuration.

    Args:
        prompt_file (str): Path to file containing system prompts
        tools_to_use (List[str], optional): List of tool names to initialize. If None, all tools are initialized.
        model_dir (str, optional): Directory containing model weights. Defaults to "/model-weights".
        temp_dir (str, optional): Directory for temporary files. Defaults to "temp".
        device (str, optional): Device to run models on. Defaults to "cuda".
        model (str, optional): Model to use. Defaults to "chatgpt-4o-latest".
        temperature (float, optional): Temperature for the model. Defaults to 0.7.
        top_p (float, optional): Top P for the model. Defaults to 0.95.
        prompt_key (str, optional): Which system prompt section to load. Defaults to "MEDICAL_ASSISTANT".
        openai_kwargs (dict, optional): Additional keyword arguments for OpenAI API, such as API key and base URL.

    Returns:
        Tuple[Agent, Dict[str, BaseTool]]: Initialized agent and dictionary of tool instances
    """
    prompts = load_prompts_from_file(prompt_file)
    prompt = prompts[prompt_key]

    all_tools = {
        "ChestXRayClassifierTool": lambda: ChestXRayClassifierTool(device=device),
        "ChestXRaySegmentationTool": lambda: ChestXRaySegmentationTool(device=device),
        "CardiothoracicRatioTool": lambda: CardiothoracicRatioTool(device=device, temp_dir=temp_dir),
        "LlavaMedTool": lambda: LlavaMedTool(cache_dir=model_dir, device=device, load_in_8bit=True),
        "XRayVQATool": lambda: XRayVQATool(cache_dir=model_dir, device=device),
        "ChestXRayReportGeneratorTool": lambda: ChestXRayReportGeneratorTool(
            cache_dir=model_dir, device=device
        ),
        "XRayPhraseGroundingTool": lambda: XRayPhraseGroundingTool(
            cache_dir=model_dir, temp_dir=temp_dir, load_in_8bit=True, device=device
        ),
        "ChestXRayGeneratorTool": lambda: ChestXRayGeneratorTool(
            model_path=f"{model_dir}/roentgen", temp_dir=temp_dir, device=device
        ),
        "ImageVisualizerTool": lambda: ImageVisualizerTool(),
        "DicomProcessorTool": lambda: DicomProcessorTool(temp_dir=temp_dir),
        "MedRAXDicomWebTool": lambda: MedRAXDicomWebTool(
            base_url=os.getenv("DICOMWEB_URL", "http://localhost:8042/dicom-web"),
            username=os.getenv("DICOMWEB_USERNAME"),
            password=os.getenv("DICOMWEB_PASSWORD"),
            temp_dir=temp_dir,
        ),
        "MedRAXReportToPacsTool": lambda: MedRAXReportToPacsTool(
            base_url=os.getenv("DICOMWEB_URL", "http://localhost:8042/dicom-web"),
            username=os.getenv("DICOMWEB_USERNAME"),
            password=os.getenv("DICOMWEB_PASSWORD"),
        ),
        "MedGemmaTool": lambda: MedGemmaTool(
            model=os.getenv("MEDGEMMA_MODEL"),
            base_url=os.getenv("MEDGEMMA_BASE_URL"),
            api_key=os.getenv("MEDGEMMA_API_KEY"),
        ),
        "WebSearchTool": lambda: WebSearchTool(),
        "PythonSandboxTool": lambda: PythonSandboxTool(temp_dir=temp_dir),
    }

    # Initialize only selected tools or all if none specified
    tools_dict = {}
    tools_to_use = tools_to_use or all_tools.keys()
    for tool_name in tools_to_use:
        if tool_name in all_tools:
            tools_dict[tool_name] = all_tools[tool_name]()

    # Inject the classifier into the CTR tool for an independent Cardiomegaly cross-check,
    # reusing the already-loaded model instead of loading a second copy onto the GPU.
    if "CardiothoracicRatioTool" in tools_dict and "ChestXRayClassifierTool" in tools_dict:
        tools_dict["CardiothoracicRatioTool"].classifier = tools_dict["ChestXRayClassifierTool"]

    checkpointer = MemorySaver()
    # A finite timeout + bounded retries so a stalled LLM call fails cleanly instead
    # of hanging forever (an indefinite hang holds the API's global agent lock and
    # blocks every other session). Override via MEDRAX_LLM_TIMEOUT / MEDRAX_LLM_RETRIES.
    llm_timeout = float(os.getenv("MEDRAX_LLM_TIMEOUT", "300"))
    llm_retries = int(os.getenv("MEDRAX_LLM_RETRIES", "2"))
    model = ChatOpenAI(
        model=model,
        temperature=temperature,
        top_p=top_p,
        timeout=llm_timeout,
        max_retries=llm_retries,
        **openai_kwargs,
    )
    agent = Agent(
        model,
        tools=list(tools_dict.values()),
        log_tools=True,
        log_dir="logs",
        system_prompt=prompt,
        checkpointer=checkpointer,
        max_context_tokens=int(os.getenv("MEDRAX_MAX_CONTEXT_TOKENS", "65536")),
    )

    print("Agent initialized")
    return agent, tools_dict


if __name__ == "__main__":
    """
    This is the main entry point for the MedRAX application.
    It initializes the agent with the selected tools and creates the demo.
    """
    print("Starting server...")

    # Example: initialize with only specific tools
    # Here three tools are commented out, you can uncomment them to use them
    selected_tools = [
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
        # "ChestXRayGeneratorTool",
    ]

    # Collect the ENV variables
    openai_kwargs = {}
    if api_key := os.getenv("OPENAI_API_KEY"):
        openai_kwargs["api_key"] = api_key

    if base_url := os.getenv("OPENAI_BASE_URL"):
        openai_kwargs["base_url"] = base_url

    agent, tools_dict = initialize_agent(
        "medrax/docs/system_prompts.txt",
        tools_to_use=selected_tools,
        model_dir="/mnt/dados2/MedRAX/downloadable_content/model_weights",
        temp_dir="temp",  # Change this to the path of the temporary directory
        device="cuda",  # Change this to the device you want to use
        model="Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        temperature=0.7,
        top_p=0.95,
        prompt_key="MEDICAL_ASSISTANT",
        openai_kwargs=openai_kwargs
    )
    demo = create_demo(agent, tools_dict)

    demo.launch(server_name="0.0.0.0", server_port=8585, share=True)

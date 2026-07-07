import json
import operator
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, TypedDict, Annotated, Optional

from langgraph.graph import StateGraph, END
from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

_ = load_dotenv()

# Rough token estimate: ~3.5 chars/token is conservative for English + code/JSON
# tool output (it slightly over-counts, so we trim a little early rather than
# overflow the model's context window). Inline base64 images are counted at a
# flat cost instead of by their (enormous) character length.
_CHARS_PER_TOKEN = 3.5
_IMAGE_TOKEN_COST = 1500


def _estimate_message_tokens(message: BaseMessage) -> int:
    """Estimate the token cost of a single message, handling multimodal content.

    Text content is counted by length; image parts (base64 data URIs) are counted
    at a flat ``_IMAGE_TOKEN_COST`` so a single inline image does not appear to
    consume the entire context window.
    """
    content = message.content
    chars = 0
    images = 0
    if isinstance(content, str):
        chars += len(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "image_url" or "image_url" in part:
                    images += 1
                else:
                    chars += len(str(part.get("text", part)))
            else:
                chars += len(str(part))
    else:
        chars += len(str(content))

    # Tool calls carry their arguments as JSON that also costs tokens.
    for call in getattr(message, "tool_calls", None) or []:
        chars += len(str(call.get("args", "")))

    # A small per-message overhead for role/formatting tokens.
    return int(chars / _CHARS_PER_TOKEN) + images * _IMAGE_TOKEN_COST + 4


class ToolCallLog(TypedDict):
    """
    A TypedDict representing a log entry for a tool call.

    Attributes:
        timestamp (str): The timestamp of when the tool call was made.
        tool_call_id (str): The unique identifier for the tool call.
        name (str): The name of the tool that was called.
        args (Any): The arguments passed to the tool.
        content (str): The content or result of the tool call.
    """

    timestamp: str
    tool_call_id: str
    name: str
    args: Any
    content: str


class AgentState(TypedDict):
    """
    A TypedDict representing the state of an agent.

    Attributes:
        messages (Annotated[List[AnyMessage], operator.add]): A list of messages
            representing the conversation history. The operator.add annotation
            indicates that new messages should be appended to this list.
    """

    messages: Annotated[List[AnyMessage], operator.add]


class Agent:
    """
    A class representing an agent that processes requests and executes tools based on
    language model responses.

    Attributes:
        model (BaseLanguageModel): The language model used for processing.
        tools (Dict[str, BaseTool]): A dictionary of available tools.
        checkpointer (Any): Manages and persists the agent's state.
        system_prompt (str): The system instructions for the agent.
        workflow (StateGraph): The compiled workflow for the agent's processing.
        log_tools (bool): Whether to log tool calls.
        log_path (Path): Path to save tool call logs.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: List[BaseTool],
        checkpointer: Any = None,
        system_prompt: str = "",
        log_tools: bool = True,
        log_dir: Optional[str] = "logs",
        max_context_tokens: int = 65536,
        reserve_tokens: int = 8192,
    ):
        """
        Initialize the Agent.

        Args:
            model (BaseLanguageModel): The language model to use.
            tools (List[BaseTool]): A list of available tools.
            checkpointer (Any, optional): State persistence manager. Defaults to None.
            system_prompt (str, optional): System instructions. Defaults to "".
            log_tools (bool, optional): Whether to log tool calls. Defaults to True.
            log_dir (str, optional): Directory to save logs. Defaults to 'logs'.
            max_context_tokens (int, optional): The model's context window in tokens.
                The conversation history is trimmed to stay within this budget so a
                long session never overflows the window (which would otherwise hang
                the local LLM server). Defaults to 65536.
            reserve_tokens (int, optional): Tokens held back from the window for the
                model's response and the tool-schema definitions bound to every
                request. Defaults to 8192.
        """
        self.system_prompt = system_prompt
        self.log_tools = log_tools
        self.max_context_tokens = max_context_tokens
        self.reserve_tokens = reserve_tokens

        if self.log_tools:
            self.log_path = Path(log_dir or "logs")
            self.log_path.mkdir(exist_ok=True)

        # Define the agent workflow
        workflow = StateGraph(AgentState)
        workflow.add_node("process", self.process_request)
        workflow.add_node("execute", self.execute_tools)
        workflow.add_conditional_edges(
            "process", self.has_tool_calls, {True: "execute", False: END}
        )
        workflow.add_edge("execute", "process")
        workflow.set_entry_point("process")

        self.workflow = workflow.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def _trim_history(self, messages: List[AnyMessage]) -> List[AnyMessage]:
        """Keep the most recent messages that fit the context budget.

        Trims from the oldest end so the model never receives more than the window
        can hold. The result is guaranteed not to begin with a ``ToolMessage`` (an
        orphaned tool result with no preceding tool call is rejected by
        OpenAI-compatible servers), and the newest message is always kept.
        """
        system_tokens = (
            _estimate_message_tokens(SystemMessage(content=self.system_prompt))
            if self.system_prompt
            else 0
        )
        budget = max(1024, self.max_context_tokens - self.reserve_tokens - system_tokens)

        kept: List[AnyMessage] = []
        used = 0
        # Walk from newest to oldest, accumulating until the budget is exhausted.
        for message in reversed(messages):
            cost = _estimate_message_tokens(message)
            if kept and used + cost > budget:
                break
            kept.append(message)
            used += cost
        kept.reverse()

        # Never start on a tool result whose originating tool call was trimmed away.
        while kept and isinstance(kept[0], ToolMessage):
            kept.pop(0)

        # Fall back to the single newest message if trimming removed everything.
        return kept or messages[-1:]

    def process_request(self, state: AgentState) -> Dict[str, List[AnyMessage]]:
        """
        Process the request using the language model.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            Dict[str, List[AnyMessage]]: A dictionary containing the model's response.
        """
        messages = self._trim_history(state["messages"])
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        response = self.model.invoke(messages)
        return {"messages": [response]}

    def has_tool_calls(self, state: AgentState) -> bool:
        """
        Check if the response contains any tool calls.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            bool: True if tool calls exist, False otherwise.
        """
        response = state["messages"][-1]
        return len(response.tool_calls) > 0

    def execute_tools(self, state: AgentState) -> Dict[str, List[ToolMessage]]:
        """
        Execute tool calls from the model's response.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            Dict[str, List[ToolMessage]]: A dictionary containing tool execution results.
        """
        tool_calls = state["messages"][-1].tool_calls
        results = []

        for call in tool_calls:
            print(f"Executing tool: {call}")
            if call["name"] not in self.tools:
                print("\n....invalid tool....")
                result = "invalid tool, please retry"
            else:
                result = self.tools[call["name"]].invoke(call["args"])

            results.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    name=call["name"],
                    args=call["args"],
                    content=str(result),
                )
            )

        self._save_tool_calls(results)
        print("Returning to model processing!")

        return {"messages": results}

    def _save_tool_calls(self, tool_calls: List[ToolMessage]) -> None:
        """
        Save tool calls to a JSON file with timestamp-based naming.

        Args:
            tool_calls (List[ToolMessage]): List of tool calls to save.
        """
        if not self.log_tools:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_path / f"tool_calls_{timestamp}.json"

        logs: List[ToolCallLog] = []
        for call in tool_calls:
            log_entry = {
                "tool_call_id": call.tool_call_id,
                "name": call.name,
                "args": call.args,
                "content": call.content,
                "timestamp": datetime.now().isoformat(),
            }
            logs.append(log_entry)

        with open(filename, "w") as f:
            json.dump(logs, f, indent=4)

from abc import ABC, abstractmethod
from typing import Any, Optional
from pydantic import BaseModel, Field
from langfuse import observe
from loguru import logger
import json

from llm.base import LLMClient
from tools.registry import ToolRegistry


class BaseAgentState(BaseModel):
    """
    Holds the evolving state of an agent's execution.
    """
    messages: list[dict] = Field(default_factory=list)
    is_finished: bool = False
    iteration: int = 0

    def add_message(self, role: str, content: str, **extra):
        msg = {"role": role, "content": content}
        # extra for example tool_id
        msg.update(extra)
        self.messages.append(msg)
        

class Agent(ABC):
    def __init__( self, llm: LLMClient, tool_registry: ToolRegistry, max_iterations: int = 100):
        self.llm = llm
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations
        # initial state to start with
        self.inital_state = BaseAgentState()

    @abstractmethod
    def start_point(self, *args, **kwargs) -> BaseAgentState:
        """ Start Point of State for example start of user query or anything """
        raise NotImplementedError()
    
    @abstractmethod
    def run(self, state: BaseAgentState) -> BaseAgentState:
        """ Run 1 Step/Iteration """
        raise NotImplementedError()

    def iterate(self, *args, **kwargs) -> BaseAgentState:
        state = self.start_point(*args, **kwargs)
        # TODO: iteration schema alwayse same regarding `self.run(state)``
        # TODO: istate.is_finished or state.iteration > self.max_iteration
        # TODO: while: run()
        while not state.is_finished:
            raise NotImplementedError()

        return state

    # LLM WRAPPER
    @observe(name="llm-call", as_type="generation")
    def llm_generate(self, state: BaseAgentState):
        # TODO: wrap code to generate from llm -- self.tool_registry.to_client_tools to convert to client format
        raise NotImplementedError()
    # TOOL EXECUTION WRAPPER
    @observe(name="tool-call", as_type="tool")
    def call_tool(self, tool_call):
        # TODO: wrap code to execute function with error pron and tracing here
        raise NotImplementedError()
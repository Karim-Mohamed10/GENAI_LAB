from ..base import Agent, BaseAgentState, LLMClient, ToolRegistry
from tools.toolkit.builtin import code_tools, file_tools, json_tools
from pathlib import Path
from loguru import logger

class ScratchpadUnitTesterAgent(Agent):
    def __init__(self, llm: LLMClient,  max_iterations: int = 100):
        # TODO: create tool_registery with what you want
        raise NotImplementedError()
        super().__init__(llm, tool_registry, max_iterations)
        
        # TODO: inialize state
        # TODO 1: read prompts/unit_tester_v2.txt
        # TODO 2: format system prompt with tools
        # TODO 3: add system message to inital_state
        raise NotImplementedError()
    
    def start_point(self, user_query) -> BaseAgentState:
        """ Start Point of State for example start of user query or anything """
        state = self.inital_state
        state.add_message(role="user", content=user_query)

        return state
    
    def run(self, state: BaseAgentState) -> BaseAgentState:
        # TODO 1) Call LLM
        raise NotImplementedError()
        
        # TODO 2) **PRUNE PREVIOUSE OUTPUTs (what should I keep?) ** + Add assistant response
        raise NotImplementedError()

        # TODO 3) Set Stop condition
        raise NotImplementedError()
        
        # TODO 4) Execute tool calls if any
        raise NotImplementedError()
        # TODO 5) return what ?
        raise NotImplementedError()

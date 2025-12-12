import importlib
from types import ModuleType
from typing import Dict, List
from loguru import logger
from .base import Tool
from llm.config import LLMProvider

class ToolRegistry:
    """
    Registry for all tools in the system.
    (optional to register all tool under session)
    """

    def __init__(self, session_id: str = None):
        self._tools: Dict[str, Tool] = {}
        self._session_id = session_id
        
    def register(self, tool: Tool):
        """
        Register a single Tool instance & inject session_id.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        logger.debug(f"register new tool {tool.name} and inject session `{self._session_id}`")
        tool.session_id = self._session_id
        self._tools[tool.name] = tool

    def register_from_module(self, module: ModuleType):
        """
        Register all tools from a given module that have been decorated.
        """
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            # Only register Tool instances
            if isinstance(attr, Tool):
                self.register(attr)

    def get(self, name: str) -> Tool:
        """
        Retrieve a tool by name.
        """
        return self._tools.get(name)

    def list_tools(self) -> [str]:
        """
        List all registered tools with metadata.
        """
        return [ tool.to_string()
            for tool in self._tools.values()
        ]

    def to_client_tools(self, llm_provider: LLMProvider) -> List[dict]:
        """
        Convert registered tools to OpenAI-compatible tools schema:
        [
            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": {...}
                }
            }
        ]
        """
        return [tool.to_client_format(llm_provider) for tool in self._tools.values()]
    
    def to_string(self) -> [str]:
        """
        List all registered tools with metadata.
        """
        return "\n".join(self.list_tools())

    def load_module(self, module_path: str):
        """
        Dynamically import a module and register its tools.
        Example: registry.load_module("tools.builtin.math_tools")
        """
        module = importlib.import_module(module_path)
        self.register_from_module(module)

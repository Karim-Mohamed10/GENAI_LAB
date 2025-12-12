from typing import Callable
from loguru import logger
from llm.config import LLMProvider

class Tool:
    """
    A class representing a reusable piece of code (Tool).

    Attributes:
        name (str): Name of the tool.
        description (str): A textual description of what the tool does.
        func (callable): The function this tool wraps.
        arguments (list): A list of arguments.
        outputs (str or list): The return type(s) of the wrapped function.
        session_id (str): Optional id for session *advanced to use for playwright or code etc...*
    """
    def __init__(self,
                 name: str,
                 description: str,
                 func: Callable,
                 arguments: list,
                 outputs: str,
                 session_id: str = None):
        self.name = name
        self.description = description
        self.func = func
        self.arguments = arguments
        self.outputs = outputs
        self.session_id = session_id

    def to_string(self) -> str:
        """
        Return a string representation of the tool,
        """
        # TODO: complete function with proper string output
        # TODO: if there's session_id as argname skip it as we inject it
        raise NotImplementedError()
        args_str = None # how you concate args and skip session_id ?
        

        return (
            f"Tool Name: {self.name},"
            f" Description: {self.description},"
            f" Arguments: {args_str},"
            f" Outputs: {self.outputs}"
        )
    
    def to_openai_format(self) -> dict:
        """
        Return a OpenAI-compatible tool schema for chat completion calls.
        Converts argument list to JSON Schema format.
        """
        properties = {}
        required_args = []

        for arg_name, arg_type in self.arguments:
            if arg_name == "session_id":
                continue  # session_id is injected, not required in schema

            # map simple types to JSON Schema
            if arg_type.lower() in ["str", "string"]:
                schema_type = "string"
            elif arg_type.lower() in ["int", "integer"]:
                schema_type = "integer"
            elif arg_type.lower() in ["bool", "boolean"]:
                schema_type = "boolean"
            elif arg_type.lower() in ["list", "array"]:
                schema_type = "array"
            else:
                schema_type = "string"

            properties[arg_name] = {"type": schema_type}
            required_args.append(arg_name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_args,
                },
            },
        }

    def to_gemini_format(self) -> dict:
        """
        Return a Gemini-compatible tool schema for chat completion calls.
        Converts argument list to Gemini's JSON Schema format.
        """
        properties = {}
        required_args = []

        for arg_name, arg_type in self.arguments:
            if arg_name == "session_id":
                continue  # session_id is injected, not required in schema

            # Map basic types to JSON Schema
            if arg_type.lower() in ["str", "string"]:
                schema_type = "string"
            elif arg_type.lower() in ["int", "integer"]:
                schema_type = "integer"
            elif arg_type.lower() in ["bool", "boolean"]:
                schema_type = "boolean"
            elif arg_type.lower() in ["list", "array"]:
                schema_type = "array"
            else:
                schema_type = "string"

            properties[arg_name] = {
                "type": schema_type,
            }
            required_args.append(arg_name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required_args,
            },
        }

    def to_client_format(self, llm_provider: LLMProvider):
        if llm_provider in [LLMProvider.GROQ, LLMProvider.OPENAI] :
            return self.to_openai_format()
        elif llm_provider == LLMProvider.GEMINI:
            return self.to_gemini_format()
        
    def __call__(self, *args, **kwargs):
        """
        Invoke the underlying function (callable) with provided arguments.
        """
        # TODO complete the function + inject session_id=self.session_id
        raise NotImplementedError()
        logger.debug(f"calling tool {self.name} with {args} {kwargs}")
        return self.func(*args, **kwargs)
        
    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return f"<Tool {self.name}: {self.description[:50]}{'...' if len(self.description) > 50 else ''}>"
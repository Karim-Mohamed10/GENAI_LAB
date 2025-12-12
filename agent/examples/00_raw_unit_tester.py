"""
This is an Agent who given certain python files will write unit tests using pytest 
And will execute them and report result

Difference Between it And `01_raw_traced_unit_tester.py` no mlflow/langfuse 
"""
from tools.registry import ToolRegistry
import tools.toolkit.web_explorer as web_explorer_tools
from llm.groq_client import GroqClient, LLMConfig
from loguru import logger
import json

from pathlib import Path

# ================ 1. Initalization ================
# 1.1 setup llm client
config = LLMConfig(
    max_tokens=5000,
    model_name="openai/gpt-oss-120b",
    reasoning_effort="medium",
    temperature=1.0,
    top_p=1
)
client = GroqClient(config)

# 1.2 setup path
messages = [
    {
        "role": "system", "content": """
        You are a highly skilled QA Automation Agent with expertise in Python programming, unit testing (using Pytest), and modern GenAI tools. 
        Your role is to review, write, and execute comprehensive unit tests for the provided toolkit modules to ensure reliability and correctness. 
        Analyze each tool's behavior, suggest improvements if needed, and provide a clear, structured test report based on your findings.
        
        Output:
            - "finished": <boolean, indicate if the task is complete>
            - "message": <summary and coverage of tests>
        
        **Use only this tools:**
        {tools}
        you will be penalized if you use OTHER TOOLS
        """
    },
    {
        "role": "user", "content": """write unite tests for file: {files_under_test} code and run it ensure everything is okay then report it
        You are only allowed to change files in this directory {tests_output_directory_path}
        """ 
    }
]
# TODO: 1.3 create tool register and add tools/modules you need
raise NotImplementedError()
# TODO: 1.4 add tools to system_message use str.format method and  registery.to_string()
raise NotImplementedError()
# TODO: 1.5 set `files_under_test` and `tests_output_directory_path` in user_message like .format
raise NotImplementedError()

# ================ 2. Starts Iterations ================
max_iterations = 20
iteration = 0
while True:
    iteration += 1
    logger.info(f"Iteration {iteration}")
    # TODO 2.1 call client (provide tools in .generate) with registery.to_openai_tools()
    raise NotImplementedError()
    # TODO 2.2 append assistant message (role, content, **tool_calls**) *log it logger.info*
    raise NotImplementedError()

    # TODO get content and check if is finished
    # 2.3 Stop when one of the conditions happen
    # 2.3 'finished' in response['content'] -- handle response['content']=None case
    # 2.3 exceed max_iterations
    raise NotImplementedError()
    
    # 2.4 execute any function execturion inside `tool_calls` || handle if it's None or not passed
    tool_calls = response.get("tool_calls", []) or []
    for tool_call in tool_calls:
        if tool_call["type"] != "function":
            continue
        try:
            # TODO: extract func_name, args and call it -> set tool_message {content: result}
            func_name = 'dummy'      
            args_raw = 'dummy'
            if isinstance(args_raw, str):
                func_inputs = json.loads(args_raw)
            else:
                func_inputs = args_raw
                
            func_results = 'dummy'

            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": json.dumps(func_results),
            }
        except Exception as error:
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": json.dumps(error),
            }
            
        messages.append(tool_message)
        logger.info(f"tool response {json.dumps(tool_message, indent=2)}")
        

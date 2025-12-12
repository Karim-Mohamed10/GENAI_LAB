from agent.unit_tester.v1_simple import SimpleUnitTesterAgent
from llm.openai_client import OpenAIClient, LLMConfig
from llm.config import LLMProvider
from pathlib import Path

# 1. llm client
config = LLMConfig(
    provider=LLMProvider.OPENAI,
    max_tokens=5000,
    model_name="gpt-4.1",
    reasoning_effort="medium",
    temperature=1.0,
    top_p=1
)
client = OpenAIClient(config)

# 2. user query
tests_output_directory_path = Path("tools/llm_tests")
tests_output_directory_path.mkdir(exist_ok=True, parents=True)

files_under_test = ["tools/toolkit/web_explorer.py",]
user_query = f"""write unite tests for file: {files_under_test} code and run it ensure everything is okay then report it
        You are only allowed to change files in this directory {str(tests_output_directory_path)}
        """
        
# 3. run agent

agent = SimpleUnitTesterAgent(client)

state = agent.iterate(user_query = user_query)

print(state)
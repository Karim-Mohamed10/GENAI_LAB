import os
from typing import Iterator
from dotenv import load_dotenv
from groq import Groq
from .base import LLMClient
from .config import LLMConfig

# TODO 1: load dotenv
raise NotImplementedError()

class GroqClient(LLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        # TODO 2: create groq client and set api_key from .env
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    def generate(self, messages: [dict[str, str]], tools = None) -> dict:
        # TODO: write description for Returend Fields 
        """ Main Returned Dict Fields
            - content
            - [YOUR_ANSWER]
        """
        # TODO 3: call `client.chat.completions.create` with configurations in self.config
        # TODO 3: search difference between max_tokens and max_compeletion_tokens
        # TODO 3: now you can pass tools=tools but search about format later when move to tools sections
        raise NotImplementedError()
        response = None
        return response.choices[0].message.model_dump()
    
    def stream(self, messages: [dict[str, str]], tools = None) -> Iterator[dict]:
        # TODO 3: call `client.chat.completions.create` with stream options configurations in self.config
        raise NotImplementedError()
        stream = None
        for chunk in stream:
            yield chunk.choices[0].delta.model_dump()
                
                
if __name__ == "__main__":
    #TODO: initlaize configuraiton with reasoning model -- search for groq reasoning models 
    raise NotImplementedError()
    config = None
    client = GroqClient(config)
    
    #TODO: write messages with (1. system prompt on how the model is QA engineer and know python, playwright etc... provide in course) 
    #TODO: (2. Ask model to "write a plan to build a software autonomus like cursor but for testing") 
    messages = [
        {
            "role": "system", "content": """dummy""",
        },
        {
            "role": "user", "content": "write a plan to build a software autonomus like cursor but for testing"   
        }
    ]
    
    #TODO: test client.generate
    response = client.generate(messages)
    #TODO: test client.stream and mention what's difference and why we need it?
    # YOUR_ANSWER: ...
    for chunk in client.stream(messages):
        print(chunk)
    
    #TODO add the new answer to messages -> create multi-turn conversation (with same system message from above)
        # user: your name is CHATTAH tester
        # assisstant: ...
        # user: tell me what's your name and what are language you expert in it ?
    messages = [
        messages[0],
        {"role": "user", "content": "your name is CHATTAH tester"},        
    ]
    # TODO: first turn -> get answer -> print answer -> append answer to messages "i.e state"
    answer = None
    # TODO: second turn -> get answer -> print answer -> append answer to messages "i.e state"
    answer = None
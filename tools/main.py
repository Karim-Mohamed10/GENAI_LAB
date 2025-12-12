from sys import exception
from session import Session
from tools.registry import ToolRegistry
import tools.toolkit.web_explorer as web_explorer_tools
from tools.toolkit.builtin import code_tools, file_tools, json_tools
from llm.groq_client import GroqClient, LLMConfig
from loguru import logger
import json


def simple_test_browser_tools():
    with Session("starting") as session:
        registery = ToolRegistry(session.session_id)
        registery.register_from_module(web_explorer_tools)
        
        print(registery.list_tools)
        
        status = registery.get("goto_url")("https://www.google.com")
        logger.info(f"Navigat to Google status={status}")
        assert("HTTP Status: 200" in status)
        
        text = registery.get("get_page_content")(mode="text")
        logger.info(f"The content of page {text}")
        assert("Google" in text)
        
        status = registery.get("end_browsing_page")()
        logger.info(f"Closing Page status={status}")
        assert(status == "Page closed and session terminated.")
        

if __name__ == "__main__":
    simple_test_browser_tools()
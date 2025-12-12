from tools.decorator import tool
import json

@tool()
def json_is_valid(s: str) -> bool:
    """
    Check if the input string is valid JSON.

    Args:
        s: The string to check.

    Returns:
        True if the string is valid JSON, False otherwise.
    """
    #TODO: implement function with details and why llm need it ?
    # YOUR_ANSWER: ...
    raise NotImplementedError()

if __name__ == "__main__":
    print(json_is_valid.to_string())
from tools.decorator import tool

@tool(name="add", description="Adding Numbers")
def add(a: int|float, b: int|float) -> int|float:
    "Add two numbers"
    return a + b

@tool()
def subtract(a: int|float, b: int|float) -> int|float:
    "Subtract two numbers"
    return a - b

@tool()
def multiply(a: int|float, b: int|float) -> int|float:
    """Multiply two numbers."""
    return a * b

if __name__ == "__main__":
    print(subtract.to_string())
from tools.decorator import tool

@tool()
def string_length(s: str) -> int:
    """Return the length of a string."""
    return len(s)

@tool()
def to_uppercase(s: str) -> str:
    """Convert a string to uppercase."""
    return s.upper()

@tool()
def to_lowercase(s: str) -> str:
    """Convert a string to lowercase."""
    return s.lower()

@tool()
def split_string(s: str, separator: str = " ") -> list:
    """Split a string by the given separator."""
    return s.split(separator)

@tool()
def contains(sub: str, string: str) -> bool:
    """Check if a substring is in a string."""
    return sub in string


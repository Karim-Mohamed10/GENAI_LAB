import os
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    GROQ = "groq"
    GEMINI = "gemini"
    OPENAI = "openai"

class LLMConfig(BaseModel):
    """Configuration for LLM providers"""
    
    provider: LLMProvider = Field(default=LLMProvider.GROQ, description="The LLM provider to use")
    
    base_url: Optional[str] = Field( default=None, description="API Base URL (only needed for custom endpoints)")

    model_name: str = Field( default="llama-3.1-70b-versatile", description="model name")
    
    # Common configuration
    # TODO: add temperature field with validations (0.0 <= temperature <= 2.0) and make default = 0.7
    raise NotImplementedError()
    
    # TODO: add top_p field with validations (0.0 <= top_p <= 1.0) and make default = 0.7
    raise NotImplementedError()
    # TODO: add max_tokens field
    raise NotImplementedError()
    
    # TODO: search on reasoning_effort and what it do (write description to this field)
    raise NotImplementedError()
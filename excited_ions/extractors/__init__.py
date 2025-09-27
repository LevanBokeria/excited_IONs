"""Base classes for LLM-based extractors."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import openai
from ..models import PaperMetadata, PaperContent, ExtractedData, PaperSummary, ExperimentIdea
from ..config import Config


class BaseLLMExtractor(ABC):
    """Abstract base class for LLM-based extractors."""
    
    def __init__(self, config: Config):
        self.config = config
        if not config.api.openai_api_key:
            raise ValueError("OpenAI API key is required")
        self.client = openai.OpenAI(api_key=config.api.openai_api_key)
    
    async def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Make a call to the LLM."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.api.openai_model,
                messages=messages,
                max_tokens=self.config.api.max_tokens,
                temperature=self.config.api.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling LLM: {e}")
            raise
    
    @abstractmethod
    async def extract(self, paper: Dict[str, Any]) -> Any:
        """Extract information from a paper."""
        pass
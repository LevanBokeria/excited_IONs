"""Configuration management for the excited_ions pipeline."""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class APIConfig(BaseModel):
    """API configuration settings."""
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_model: str = Field(default="gpt-4")
    max_tokens: int = Field(default=4000)
    temperature: float = Field(default=0.7)


class FetcherConfig(BaseModel):
    """Paper fetcher configuration settings."""
    arxiv_base_url: str = Field(default="http://export.arxiv.org/api/query")
    pubmed_base_url: str = Field(default="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/")
    max_results_per_query: int = Field(default=50)
    delay_between_requests: float = Field(default=1.0)


class ProcessingConfig(BaseModel):
    """Processing pipeline configuration settings."""
    output_dir: str = Field(default="./output")
    cache_dir: str = Field(default="./cache")
    max_concurrent_papers: int = Field(default=5)
    enable_caching: bool = Field(default=True)


class Config(BaseModel):
    """Main configuration class for the excited_ions pipeline."""
    api: APIConfig = Field(default_factory=APIConfig)
    fetcher: FetcherConfig = Field(default_factory=FetcherConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    
    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """Load configuration from a JSON file."""
        import json
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return cls(**config_data)
    
    def to_file(self, config_path: str) -> None:
        """Save configuration to a JSON file."""
        import json
        with open(config_path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)
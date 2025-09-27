"""Base classes for paper fetchers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import asyncio
import aiohttp
from ..models import PaperMetadata, PaperContent
from ..config import Config


class BaseFetcher(ABC):
    """Abstract base class for paper fetchers."""
    
    def __init__(self, config: Config):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def search_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for papers based on a query."""
        pass
    
    @abstractmethod
    async def fetch_paper_metadata(self, paper_id: str) -> PaperMetadata:
        """Fetch metadata for a specific paper."""
        pass
    
    @abstractmethod
    async def fetch_paper_content(self, paper_id: str) -> Optional[PaperContent]:
        """Fetch full content for a specific paper."""
        pass
    
    async def rate_limit_delay(self) -> None:
        """Add delay between requests to respect rate limits."""
        await asyncio.sleep(self.config.fetcher.delay_between_requests)
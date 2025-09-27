"""ArXiv paper fetcher implementation."""

import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from datetime import datetime
import re
from urllib.parse import urlencode
import aiohttp
import asyncio

from . import BaseFetcher
from ..models import PaperMetadata, PaperContent, Author
from ..config import Config


class ArXivFetcher(BaseFetcher):
    """Fetcher for arXiv papers."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.base_url = config.fetcher.arxiv_base_url
        self.pdf_base_url = "https://arxiv.org/pdf/"
    
    async def search_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for papers on arXiv."""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        # Build arXiv API query
        params = {
            'search_query': query,
            'start': 0,
            'max_results': min(max_results, self.config.fetcher.max_results_per_query),
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        url = f"{self.base_url}?{urlencode(params)}"
        
        try:
            async with self.session.get(url) as response:
                response.raise_for_status()
                xml_content = await response.text()
                return self._parse_search_results(xml_content)
        except Exception as e:
            print(f"Error searching arXiv: {e}")
            return []
    
    def _parse_search_results(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse arXiv search results from XML."""
        results = []
        
        try:
            root = ET.fromstring(xml_content)
            # Define namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom',
                  'arxiv': 'http://arxiv.org/schemas/atom'}
            
            entries = root.findall('atom:entry', ns)
            
            for entry in entries:
                try:
                    # Extract ID from URL
                    id_elem = entry.find('atom:id', ns)
                    arxiv_url = id_elem.text if id_elem is not None else ""
                    arxiv_id = arxiv_url.split('/')[-1] if arxiv_url else ""
                    
                    # Extract basic info
                    title_elem = entry.find('atom:title', ns)
                    title = title_elem.text.strip() if title_elem is not None else ""
                    
                    summary_elem = entry.find('atom:summary', ns)
                    abstract = summary_elem.text.strip() if summary_elem is not None else ""
                    
                    # Extract publication date
                    published_elem = entry.find('atom:published', ns)
                    pub_date_str = published_elem.text if published_elem is not None else ""
                    pub_date = None
                    if pub_date_str:
                        try:
                            pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                        except:
                            pass
                    
                    # Extract authors
                    authors = []
                    author_elems = entry.findall('atom:author', ns)
                    for author_elem in author_elems:
                        name_elem = author_elem.find('atom:name', ns)
                        if name_elem is not None:
                            authors.append({"name": name_elem.text.strip()})
                    
                    # Extract categories/subjects
                    subjects = []
                    category_elems = entry.findall('atom:category', ns)
                    for cat_elem in category_elems:
                        term = cat_elem.get('term', '')
                        if term:
                            subjects.append(term)
                    
                    results.append({
                        'id': arxiv_id,
                        'source': 'arxiv',
                        'title': title,
                        'abstract': abstract,
                        'authors': authors,
                        'publication_date': pub_date,
                        'subjects': subjects,
                        'url': arxiv_url
                    })
                    
                except Exception as e:
                    print(f"Error parsing entry: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error parsing XML: {e}")
        
        return results
    
    async def fetch_paper_metadata(self, paper_id: str) -> PaperMetadata:
        """Fetch metadata for a specific arXiv paper."""
        # Search for the specific paper
        search_results = await self.search_papers(f"id:{paper_id}", max_results=1)
        
        if not search_results:
            raise ValueError(f"Paper {paper_id} not found")
        
        paper_data = search_results[0]
        
        # Convert to PaperMetadata model
        authors = [Author(name=a['name']) for a in paper_data.get('authors', [])]
        
        return PaperMetadata(
            title=paper_data['title'],
            authors=authors,
            abstract=paper_data['abstract'],
            arxiv_id=paper_id,
            publication_date=paper_data.get('publication_date'),
            subjects=paper_data.get('subjects', [])
        )
    
    async def fetch_paper_content(self, paper_id: str) -> Optional[PaperContent]:
        """Fetch full content for an arXiv paper."""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        # For arXiv, we can fetch the PDF, but parsing PDF to text requires additional tools
        # For now, we'll return None and focus on metadata and abstracts
        # In a full implementation, you'd use pdfplumber or similar to extract text
        
        pdf_url = f"{self.pdf_base_url}{paper_id}.pdf"
        
        try:
            async with self.session.get(pdf_url) as response:
                if response.status == 200:
                    # For now, just indicate PDF is available
                    return PaperContent(
                        raw_text=f"PDF available at: {pdf_url}",
                        sections={"pdf_url": pdf_url}
                    )
        except Exception as e:
            print(f"Error fetching PDF for {paper_id}: {e}")
        
        return None
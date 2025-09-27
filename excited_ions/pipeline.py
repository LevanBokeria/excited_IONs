"""Main pipeline orchestrator for the excited_ions system."""

import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from .config import Config
from .models import ProcessedPaper, PaperMetadata
from .fetchers.arxiv_fetcher import ArXivFetcher
from .extractors.metadata_extractor import MetadataExtractor
from .extractors.summary_generator import SummaryGenerator
from .extractors.ideation_engine import IdeationEngine


class Pipeline:
    """Main pipeline for processing scientific papers."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.ensure_directories()
        
        # Initialize components
        self.arxiv_fetcher = ArXivFetcher(self.config)
        self.metadata_extractor = MetadataExtractor(self.config)
        self.summary_generator = SummaryGenerator(self.config)
        self.ideation_engine = IdeationEngine(self.config)
        
        # Storage for processed papers
        self.processed_papers: List[ProcessedPaper] = []
    
    def ensure_directories(self) -> None:
        """Ensure output and cache directories exist."""
        Path(self.config.processing.output_dir).mkdir(parents=True, exist_ok=True)
        if self.config.processing.enable_caching:
            Path(self.config.processing.cache_dir).mkdir(parents=True, exist_ok=True)
    
    async def search_and_process_papers(self, 
                                      query: str, 
                                      max_papers: int = 10,
                                      enable_ideation: bool = True) -> List[ProcessedPaper]:
        """Search for papers and process them through the full pipeline."""
        
        print(f"Starting pipeline: searching for papers with query '{query}'")
        
        # Step 1: Fetch papers
        async with self.arxiv_fetcher as fetcher:
            search_results = await fetcher.search_papers(query, max_papers)
        
        print(f"Found {len(search_results)} papers")
        
        if not search_results:
            print("No papers found for the given query")
            return []
        
        # Step 2: Process papers concurrently (with limit)
        semaphore = asyncio.Semaphore(self.config.processing.max_concurrent_papers)
        tasks = [
            self._process_single_paper(paper_data, search_results if enable_ideation else [], semaphore)
            for paper_data in search_results
        ]
        
        processed_papers = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and add to storage
        successful_papers = []
        for result in processed_papers:
            if isinstance(result, ProcessedPaper):
                successful_papers.append(result)
                self.processed_papers.append(result)
            elif isinstance(result, Exception):
                print(f"Error processing paper: {result}")
        
        print(f"Successfully processed {len(successful_papers)} papers")
        
        # Step 3: Save results
        await self.save_results(successful_papers)
        
        return successful_papers
    
    async def _process_single_paper(self, 
                                   paper_data: Dict[str, Any], 
                                   all_papers: List[Dict[str, Any]],
                                   semaphore: asyncio.Semaphore) -> ProcessedPaper:
        """Process a single paper through the full pipeline."""
        
        async with semaphore:
            paper_id = paper_data.get('id', f"paper_{datetime.now().timestamp()}")
            
            try:
                print(f"Processing paper: {paper_data.get('title', paper_id)[:50]}...")
                
                # Create initial ProcessedPaper object
                processed_paper = ProcessedPaper(
                    paper_id=paper_id,
                    source_url=paper_data.get('url'),
                    metadata=self._convert_to_metadata(paper_data),
                    processing_status="processing"
                )
                
                # Step 1: Extract metadata and experimental data
                try:
                    extracted_data = await self.metadata_extractor.extract(paper_data)
                    processed_paper.extracted_data = extracted_data
                    print(f"  ✓ Extracted metadata for {paper_id}")
                except Exception as e:
                    error_msg = f"Metadata extraction failed: {e}"
                    processed_paper.processing_errors.append(error_msg)
                    print(f"  ✗ {error_msg}")
                
                # Step 2: Generate summary
                try:
                    # Add extracted data to paper context for better summaries
                    paper_with_extracted = {**paper_data}
                    if processed_paper.extracted_data:
                        paper_with_extracted['extracted_data'] = processed_paper.extracted_data.model_dump()
                    
                    summary = await self.summary_generator.extract(paper_with_extracted)
                    processed_paper.summary = summary
                    print(f"  ✓ Generated summary for {paper_id}")
                except Exception as e:
                    error_msg = f"Summary generation failed: {e}"
                    processed_paper.processing_errors.append(error_msg)
                    print(f"  ✗ {error_msg}")
                
                # Step 3: Generate experiment idea (with related papers context)
                try:
                    # Find related papers (exclude current paper)
                    related_papers = [p for p in all_papers if p.get('id') != paper_id][:3]
                    
                    # Add context from extracted data and summary
                    paper_with_all_data = {**paper_data}
                    if processed_paper.extracted_data:
                        paper_with_all_data['extracted_data'] = processed_paper.extracted_data.model_dump()
                    if processed_paper.summary:
                        paper_with_all_data['summary'] = processed_paper.summary.model_dump()
                    
                    experiment_idea = await self.ideation_engine.extract(paper_with_all_data, related_papers)
                    processed_paper.experiment_idea = experiment_idea
                    print(f"  ✓ Generated experiment idea for {paper_id}")
                except Exception as e:
                    error_msg = f"Experiment ideation failed: {e}"
                    processed_paper.processing_errors.append(error_msg)
                    print(f"  ✗ {error_msg}")
                
                # Update status
                processed_paper.processing_status = "completed" if not processed_paper.processing_errors else "partial"
                processed_paper.processed_at = datetime.now()
                
                return processed_paper
                
            except Exception as e:
                # Handle any unexpected errors
                error_msg = f"Unexpected error processing {paper_id}: {e}"
                print(f"  ✗ {error_msg}")
                
                return ProcessedPaper(
                    paper_id=paper_id,
                    source_url=paper_data.get('url'),
                    metadata=self._convert_to_metadata(paper_data),
                    processing_status="failed",
                    processing_errors=[error_msg],
                    processed_at=datetime.now()
                )
    
    def _convert_to_metadata(self, paper_data: Dict[str, Any]) -> PaperMetadata:
        """Convert paper data to PaperMetadata model."""
        from .models import Author
        
        authors = []
        for author_data in paper_data.get('authors', []):
            authors.append(Author(name=author_data.get('name', '')))
        
        return PaperMetadata(
            title=paper_data.get('title', ''),
            authors=authors,
            abstract=paper_data.get('abstract', ''),
            arxiv_id=paper_data.get('id') if paper_data.get('source') == 'arxiv' else None,
            publication_date=paper_data.get('publication_date'),
            subjects=paper_data.get('subjects', [])
        )
    
    async def save_results(self, papers: List[ProcessedPaper]) -> None:
        """Save processed papers to JSON files."""
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_subdir = Path(self.config.processing.output_dir) / f"results_{timestamp}"
        output_subdir.mkdir(exist_ok=True)
        
        # Save individual paper files
        for paper in papers:
            filename = f"{paper.paper_id}.json"
            filepath = output_subdir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(paper.model_dump(), f, indent=2, ensure_ascii=False, default=str)
        
        # Save summary file
        summary_data = {
            "processing_timestamp": timestamp,
            "total_papers": len(papers),
            "successful_papers": len([p for p in papers if p.processing_status == "completed"]),
            "partial_papers": len([p for p in papers if p.processing_status == "partial"]),
            "failed_papers": len([p for p in papers if p.processing_status == "failed"]),
            "papers": [
                {
                    "paper_id": p.paper_id,
                    "title": p.metadata.title,
                    "status": p.processing_status,
                    "errors": len(p.processing_errors)
                }
                for p in papers
            ]
        }
        
        summary_filepath = output_subdir / "processing_summary.json"
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_subdir}")
        print(f"Summary: {summary_data['successful_papers']} completed, {summary_data['partial_papers']} partial, {summary_data['failed_papers']} failed")
    
    def get_paper_by_id(self, paper_id: str) -> Optional[ProcessedPaper]:
        """Retrieve a processed paper by ID."""
        for paper in self.processed_papers:
            if paper.paper_id == paper_id:
                return paper
        return None
    
    def get_papers_by_status(self, status: str) -> List[ProcessedPaper]:
        """Retrieve papers by processing status."""
        return [paper for paper in self.processed_papers if paper.processing_status == status]
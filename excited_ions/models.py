"""Data models for the excited_ions pipeline."""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl


class Author(BaseModel):
    """Model for paper author information."""
    name: str
    affiliation: Optional[str] = None
    email: Optional[str] = None


class ExperimentalCondition(BaseModel):
    """Model for experimental conditions."""
    parameter: str
    value: Union[str, float, int]
    unit: Optional[str] = None
    uncertainty: Optional[float] = None


class ExperimentalResult(BaseModel):
    """Model for experimental results."""
    measurement: str
    value: Union[str, float, int]
    unit: Optional[str] = None
    uncertainty: Optional[float] = None
    method: Optional[str] = None
    conditions: List[ExperimentalCondition] = Field(default_factory=list)


class PaperMetadata(BaseModel):
    """Model for paper metadata."""
    title: str
    authors: List[Author] = Field(default_factory=list)
    abstract: str
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pubmed_id: Optional[str] = None
    publication_date: Optional[datetime] = None
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    subjects: List[str] = Field(default_factory=list)


class PaperContent(BaseModel):
    """Model for paper full content."""
    raw_text: str
    sections: Dict[str, str] = Field(default_factory=dict)
    figures: List[Dict[str, Any]] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)


class ExtractedData(BaseModel):
    """Model for data extracted by LLMs."""
    experimental_results: List[ExperimentalResult] = Field(default_factory=list)
    methodologies: List[str] = Field(default_factory=list)
    materials: List[str] = Field(default_factory=list)
    key_findings: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class PaperSummary(BaseModel):
    """Model for LLM-generated paper summary."""
    executive_summary: str
    key_contributions: List[str] = Field(default_factory=list)
    methodology_summary: str
    results_summary: str
    significance: str
    generated_at: datetime = Field(default_factory=datetime.now)


class ExperimentIdea(BaseModel):
    """Model for LLM-generated experiment idea."""
    title: str
    hypothesis: str
    methodology: str
    expected_outcomes: str
    required_resources: List[str] = Field(default_factory=list)
    timeline: Optional[str] = None
    rationale: str
    related_papers: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)


class ProcessedPaper(BaseModel):
    """Model for fully processed paper with all extracted information."""
    paper_id: str
    source_url: Optional[HttpUrl] = None
    metadata: PaperMetadata
    content: Optional[PaperContent] = None
    extracted_data: Optional[ExtractedData] = None
    summary: Optional[PaperSummary] = None
    experiment_idea: Optional[ExperimentIdea] = None
    processing_status: str = Field(default="pending")
    processing_errors: List[str] = Field(default_factory=list)
    processed_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
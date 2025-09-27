# excited_IONs

A comprehensive scientific paper processing pipeline that fetches papers from literature databases, extracts metadata and experimental results using LLMs, generates summaries, and creates novel experiment ideas informed by related research.

## Features

- **Paper Fetching**: Retrieve papers from arXiv (with extensible architecture for PubMed and other sources)
- **Metadata Extraction**: Use LLMs to extract structured experimental data, methodologies, and key findings
- **Intelligent Summarization**: Generate comprehensive summaries with key contributions and significance
- **Experiment Ideation**: Create novel experiment ideas based on literature analysis and identified gaps
- **Structured Output**: Export all results in JSON format for easy integration and analysis
- **Concurrent Processing**: Efficiently process multiple papers with configurable concurrency limits
- **CLI Interface**: Easy-to-use command-line interface for all operations

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Install from Source

```bash
git clone https://github.com/LevanBokeria/excited_IONs.git
cd excited_IONs
pip install -e .
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Configuration

1. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
export OPENAI_API_KEY=your_api_key_here
```

2. **Create configuration file (optional):**
```bash
excited-ions init-config --output-path config.json
# Edit config.json to customize settings
```

## Usage

### Command Line Interface

#### Search and Process Papers

```bash
# Basic usage - process 10 papers about quantum computing
excited-ions process "quantum computing AND ion traps"

# Process more papers with custom output directory
excited-ions process "machine learning AND drug discovery" --max-papers 20 --output-dir ./results

# Skip experiment ideation (faster processing)
excited-ions process "CRISPR gene editing" --no-ideation

# Use different OpenAI model
excited-ions process "photosynthesis AND efficiency" --model gpt-3.5-turbo
```

#### Search Without Processing

```bash
# Preview available papers before processing
excited-ions search "quantum sensors" --max-results 15
```

#### View Results

```bash
# Show processing summary
excited-ions show-results ./output/results_20241227_143022

# Show detailed results for specific paper
excited-ions show-results ./output/results_20241227_143022 --paper-id 2312.12345
```

### Python API

```python
import asyncio
from excited_ions import Pipeline, Config

# Create configuration
config = Config()
config.api.openai_api_key = "your-api-key"
config.processing.max_concurrent_papers = 3

# Initialize pipeline
pipeline = Pipeline(config)

# Process papers
async def main():
    papers = await pipeline.search_and_process_papers(
        query="quantum machine learning",
        max_papers=5,
        enable_ideation=True
    )
    
    for paper in papers:
        print(f"Processed: {paper.metadata.title}")
        if paper.summary:
            print(f"Summary: {paper.summary.executive_summary}")
        if paper.experiment_idea:
            print(f"New idea: {paper.experiment_idea.title}")

# Run the pipeline
asyncio.run(main())
```

## Output Structure

The pipeline generates structured JSON files for each processed paper:

```json
{
  "paper_id": "2312.12345",
  "source_url": "http://arxiv.org/abs/2312.12345",
  "metadata": {
    "title": "Paper Title",
    "authors": [{"name": "Author Name", "affiliation": "University"}],
    "abstract": "Paper abstract...",
    "arxiv_id": "2312.12345",
    "publication_date": "2023-12-20T10:00:00",
    "subjects": ["quant-ph", "cs.LG"]
  },
  "extracted_data": {
    "experimental_results": [...],
    "methodologies": [...],
    "materials": [...],
    "key_findings": [...],
    "limitations": [...]
  },
  "summary": {
    "executive_summary": "Brief summary...",
    "key_contributions": [...],
    "methodology_summary": "Methods overview...",
    "results_summary": "Results overview...",
    "significance": "Why this matters..."
  },
  "experiment_idea": {
    "title": "Novel Experiment Title",
    "hypothesis": "What we expect to find...",
    "methodology": "Experimental approach...",
    "expected_outcomes": "Anticipated results...",
    "required_resources": [...],
    "timeline": "6 months",
    "rationale": "Why this experiment is needed...",
    "related_papers": [...]
  },
  "processing_status": "completed",
  "processed_at": "2023-12-27T14:30:22"
}
```

## Architecture

### Core Components

1. **Fetchers**: Retrieve papers from scientific databases
   - `ArXivFetcher`: Fetches papers from arXiv
   - Extensible for PubMed, CrossRef, etc.

2. **Extractors**: LLM-based information extraction
   - `MetadataExtractor`: Extracts experimental data and methodologies
   - `SummaryGenerator`: Creates comprehensive paper summaries
   - `IdeationEngine`: Generates novel experiment ideas

3. **Pipeline**: Orchestrates the entire processing workflow
   - Concurrent paper processing
   - Error handling and recovery
   - Result caching and storage

4. **CLI**: Command-line interface for easy operation

### Configuration Options

- **API Settings**: OpenAI model, temperature, token limits
- **Fetcher Settings**: Rate limits, result limits, source URLs
- **Processing Settings**: Concurrency, caching, output directories

## Development

### Setting Up Development Environment

```bash
git clone https://github.com/LevanBokeria/excited_IONs.git
cd excited_IONs
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black excited_ions/
flake8 excited_ions/
```

## Examples

### Example 1: Quantum Computing Research

```bash
excited-ions process "quantum computing AND error correction" --max-papers 15
```

### Example 2: Drug Discovery Pipeline

```bash
excited-ions process "machine learning AND drug discovery AND molecular" --output-dir ./drug_discovery_results
```

### Example 3: Climate Science Analysis

```bash
excited-ions process "climate modeling AND machine learning" --no-ideation --max-papers 25
```

## Limitations

- Currently supports arXiv papers (PubMed support planned)
- Requires OpenAI API access
- PDF text extraction is basic (full-text analysis could be improved)
- LLM responses may occasionally require manual validation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built for the LBF7 hackathon
- Uses OpenAI's GPT models for intelligent text processing
- arXiv API for paper retrieval

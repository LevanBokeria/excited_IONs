"""Command-line interface for the excited_ions pipeline."""

import click
import asyncio
import json
from pathlib import Path
from typing import Optional

from .pipeline import Pipeline
from .config import Config


@click.group()
@click.version_option()
def main():
    """
    excited_ions: Scientific Paper Processing Pipeline
    
    Fetch scientific papers, extract metadata and experimental results,
    generate summaries, and create new experiment ideas using LLMs.
    """
    pass


@main.command()
@click.argument('query')
@click.option('--max-papers', '-n', default=10, help='Maximum number of papers to process')
@click.option('--output-dir', '-o', default='./output', help='Output directory for results')
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--no-ideation', is_flag=True, help='Skip experiment ideation step')
@click.option('--model', default='gpt-4', help='OpenAI model to use')
def process(query: str, max_papers: int, output_dir: str, config: Optional[str], no_ideation: bool, model: str):
    """Process papers for a given search query."""
    
    # Load configuration
    if config and Path(config).exists():
        pipeline_config = Config.from_file(config)
    else:
        pipeline_config = Config()
    
    # Override with CLI options
    pipeline_config.processing.output_dir = output_dir
    pipeline_config.api.openai_model = model
    
    # Validate API key
    if not pipeline_config.api.openai_api_key:
        click.echo("Error: OPENAI_API_KEY environment variable is required", err=True)
        click.echo("Please set your OpenAI API key: export OPENAI_API_KEY=your_key_here", err=True)
        return
    
    click.echo(f"Processing papers with query: '{query}'")
    click.echo(f"Max papers: {max_papers}")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Model: {model}")
    click.echo(f"Ideation: {'Disabled' if no_ideation else 'Enabled'}")
    
    # Run the pipeline
    pipeline = Pipeline(pipeline_config)
    
    try:
        processed_papers = asyncio.run(
            pipeline.search_and_process_papers(
                query=query, 
                max_papers=max_papers,
                enable_ideation=not no_ideation
            )
        )
        
        # Display summary
        successful = len([p for p in processed_papers if p.processing_status == "completed"])
        partial = len([p for p in processed_papers if p.processing_status == "partial"])
        failed = len([p for p in processed_papers if p.processing_status == "failed"])
        
        click.echo(f"\nProcessing complete!")
        click.echo(f"✓ Successful: {successful}")
        click.echo(f"⚠ Partial: {partial}")
        click.echo(f"✗ Failed: {failed}")
        
        if processed_papers:
            click.echo(f"\nResults saved to: {pipeline_config.processing.output_dir}")
            
    except KeyboardInterrupt:
        click.echo("\nProcessing interrupted by user")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@main.command()
@click.option('--output-path', '-o', default='./config.json', help='Output path for configuration file')
def init_config(output_path: str):
    """Initialize a configuration file with default settings."""
    
    config = Config()
    config.to_file(output_path)
    
    click.echo(f"Configuration file created: {output_path}")
    click.echo("Please edit the file to customize settings and add your OpenAI API key.")


@main.command()
@click.argument('results_dir')
@click.option('--paper-id', help='Show details for specific paper ID')
def show_results(results_dir: str, paper_id: Optional[str]):
    """Show results from a previous processing run."""
    
    results_path = Path(results_dir)
    if not results_path.exists():
        click.echo(f"Error: Results directory not found: {results_dir}", err=True)
        return
    
    # Find summary file
    summary_files = list(results_path.glob("processing_summary.json"))
    if not summary_files:
        click.echo(f"Error: No processing summary found in {results_dir}", err=True)
        return
    
    with open(summary_files[0], 'r') as f:
        summary = json.load(f)
    
    if paper_id:
        # Show specific paper details
        paper_file = results_path / f"{paper_id}.json"
        if paper_file.exists():
            with open(paper_file, 'r') as f:
                paper_data = json.load(f)
            
            click.echo(f"Paper ID: {paper_data['paper_id']}")
            click.echo(f"Title: {paper_data['metadata']['title']}")
            click.echo(f"Status: {paper_data['processing_status']}")
            
            if paper_data.get('summary'):
                click.echo(f"\nSummary:")
                click.echo(paper_data['summary']['executive_summary'])
            
            if paper_data.get('experiment_idea'):
                click.echo(f"\nExperiment Idea:")
                click.echo(f"Title: {paper_data['experiment_idea']['title']}")
                click.echo(f"Hypothesis: {paper_data['experiment_idea']['hypothesis']}")
        else:
            click.echo(f"Error: Paper {paper_id} not found", err=True)
    else:
        # Show summary
        click.echo(f"Processing Summary ({summary['processing_timestamp']})")
        click.echo(f"Total papers: {summary['total_papers']}")
        click.echo(f"✓ Successful: {summary['successful_papers']}")
        click.echo(f"⚠ Partial: {summary['partial_papers']}")
        click.echo(f"✗ Failed: {summary['failed_papers']}")
        
        click.echo("\nPapers:")
        for paper in summary['papers']:
            status_icon = {"completed": "✓", "partial": "⚠", "failed": "✗"}.get(paper['status'], "?")
            click.echo(f"  {status_icon} {paper['paper_id']}: {paper['title'][:80]}...")


@main.command()
@click.argument('query')
@click.option('--max-results', '-n', default=20, help='Maximum number of results to show')
def search(query: str, max_results: int):
    """Search for papers without processing them."""
    
    config = Config()
    
    from .fetchers.arxiv_fetcher import ArXivFetcher
    
    async def do_search():
        async with ArXivFetcher(config) as fetcher:
            results = await fetcher.search_papers(query, max_results)
            return results
    
    try:
        results = asyncio.run(do_search())
        
        click.echo(f"Found {len(results)} papers for query: '{query}'\n")
        
        for i, paper in enumerate(results, 1):
            click.echo(f"{i}. {paper.get('title', 'No title')}")
            click.echo(f"   ID: {paper.get('id', 'No ID')}")
            click.echo(f"   Date: {paper.get('publication_date', 'Unknown')}")
            if paper.get('authors'):
                authors = [a['name'] for a in paper['authors'][:3]]
                author_str = ', '.join(authors)
                if len(paper['authors']) > 3:
                    author_str += f" (+{len(paper['authors'])-3} more)"
                click.echo(f"   Authors: {author_str}")
            click.echo()
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


if __name__ == '__main__':
    main()
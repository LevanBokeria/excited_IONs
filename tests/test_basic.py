"""Basic tests for the excited_ions package."""

import pytest
from excited_ions.config import Config
from excited_ions.models import PaperMetadata, Author


def test_config_creation():
    """Test that configuration can be created with default values."""
    config = Config()
    assert config.api.openai_model == "gpt-4"
    assert config.fetcher.max_results_per_query == 50
    assert config.processing.max_concurrent_papers == 5


def test_paper_metadata_creation():
    """Test that PaperMetadata model works correctly."""
    authors = [Author(name="Test Author", affiliation="Test University")]
    metadata = PaperMetadata(
        title="Test Paper",
        authors=authors,
        abstract="This is a test abstract.",
        arxiv_id="2312.12345"
    )
    
    assert metadata.title == "Test Paper"
    assert len(metadata.authors) == 1
    assert metadata.authors[0].name == "Test Author"
    assert metadata.arxiv_id == "2312.12345"


def test_package_import():
    """Test that the package can be imported successfully."""
    import excited_ions
    assert excited_ions.__version__ == "0.1.0"


if __name__ == "__main__":
    pytest.main([__file__])
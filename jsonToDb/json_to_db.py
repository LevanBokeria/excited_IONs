#!/usr/bin/env python3
"""
Scientific Papers Database with PostgreSQL + pgvector
Loads JSON files following the scientific paper schema into PostgreSQL
with vector search capabilities using embeddings.
"""

import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer
import numpy as np
from dataclasses import dataclass
import typer

from utils import print_postgres_setup_help

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "scientific_papers"
    user: str = "postgres"
    password: str = "password"

class ScientificPapersDB:
    """Database manager for scientific papers with vector search capabilities."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.conn = None
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2

    def connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password
            )
            self.conn.autocommit = True
            logger.info("Connected to database successfully")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def setup_database(self):
        """Create database schema and enable pgvector extension."""
        if not self.conn:
            raise Exception("Database connection not established")

        cur = self.conn.cursor()

        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Drop existing tables if they exist (for clean setup)
        drop_queries = [
            "DROP TABLE IF EXISTS paper_embeddings CASCADE;",
            'DROP TABLE IF EXISTS "references" CASCADE;',
            "DROP TABLE IF EXISTS equations CASCADE;",
            "DROP TABLE IF EXISTS tables CASCADE;",
            "DROP TABLE IF EXISTS figures CASCADE;",
            "DROP TABLE IF EXISTS sections CASCADE;",
            "DROP TABLE IF EXISTS authors CASCADE;",
            "DROP TABLE IF EXISTS papers CASCADE;"
        ]

        for query in drop_queries:
            cur.execute(query)

        # Create main papers table
        cur.execute("""
            CREATE TABLE papers (
                id SERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                tags TEXT[],
                experimental_setup_description TEXT,
                experimental_results_summary TEXT,
                data_availability_statement TEXT,
                code_repository_url TEXT,
                full_content JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Create authors table
        cur.execute("""
            CREATE TABLE authors (
                id SERIAL PRIMARY KEY,
                paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                author_order INTEGER NOT NULL
            );
        """)

        # Create sections table
        cur.execute("""
            CREATE TABLE sections (
                id SERIAL PRIMARY KEY,
                paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                section_order INTEGER NOT NULL
            );
        """)

        # Create figures table
        cur.execute("""
            CREATE TABLE figures (
                id SERIAL PRIMARY KEY,
                paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
                figure_id TEXT NOT NULL,
                caption TEXT NOT NULL
            );
        """)

        # Create tables table
        cur.execute("""
            CREATE TABLE tables (
                id SERIAL PRIMARY KEY,
                paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
                table_id TEXT NOT NULL,
                caption TEXT NOT NULL,
                headers TEXT[],
                rows TEXT[][]
            );
        """)

        # Create equations table
        cur.execute("""
            CREATE TABLE equations (
                id SERIAL PRIMARY KEY,
                paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
                equation_id TEXT NOT NULL,
                latex TEXT NOT NULL
            );
        """)

        # Create references table
        cur.execute("""
            CREATE TABLE "references" (
                id SERIAL PRIMARY KEY,
                paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
                reference_id TEXT NOT NULL,
                title TEXT NOT NULL,
                authors TEXT[]
            );
        """)

        # Create paper embeddings table for vector search
        cur.execute(f"""
            CREATE TABLE paper_embeddings (
                id SERIAL PRIMARY KEY,
                paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
                embedding_type TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                embedding vector({self.embedding_dim}) NOT NULL,
                text_content TEXT NOT NULL,
                source_section TEXT
            );
        """)

        logger.info("Database schema created successfully")
        cur.close()

    def create_indices(self):
        """Create indices for faster filtering and searching."""
        if not self.conn:
            raise Exception("Database connection not established")

        cur = self.conn.cursor()

        # Text search indices
        cur.execute("CREATE INDEX idx_papers_title ON papers USING GIN (to_tsvector('english', title));")
        cur.execute("CREATE INDEX idx_papers_abstract ON papers USING GIN (to_tsvector('english', abstract));")
        cur.execute("CREATE INDEX idx_papers_tags ON papers USING GIN (tags);")
        cur.execute("CREATE INDEX idx_sections_content ON sections USING GIN (to_tsvector('english', content));")

        # Standard indices
        cur.execute("CREATE INDEX idx_authors_name ON authors (name);")
        cur.execute("CREATE INDEX idx_authors_paper_id ON authors (paper_id);")
        cur.execute("CREATE INDEX idx_sections_paper_id ON sections (paper_id);")
        cur.execute("CREATE INDEX idx_figures_paper_id ON figures (paper_id);")
        cur.execute("CREATE INDEX idx_tables_paper_id ON tables (paper_id);")
        cur.execute("CREATE INDEX idx_equations_paper_id ON equations (paper_id);")
        cur.execute('CREATE INDEX idx_references_paper_id ON "references" (paper_id);')

        # Vector similarity indices (HNSW for fast approximate nearest neighbor search)
        cur.execute("CREATE INDEX idx_paper_embeddings_vector ON paper_embeddings USING hnsw (embedding vector_cosine_ops);")
        cur.execute("CREATE INDEX idx_paper_embeddings_paper_id ON paper_embeddings (paper_id);")
        cur.execute("CREATE INDEX idx_paper_embeddings_type ON paper_embeddings (embedding_type);")
        cur.execute("CREATE INDEX idx_paper_embeddings_source ON paper_embeddings (source_section);")
        cur.execute("CREATE INDEX idx_paper_embeddings_composite ON paper_embeddings (paper_id, embedding_type);")

        logger.info("Database indices created successfully")
        cur.close()

    def load_json_files(self, directory_path: str):
        """Load all JSON files from a directory into the database."""
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory {directory_path} does not exist")

        json_files = list(directory.glob("*.json"))
        if not json_files:
            logger.warning(f"No JSON files found in {directory_path}")
            return

        logger.info(f"Found {len(json_files)} JSON files to process")

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    paper_data = json.load(f)
                self._insert_paper(paper_data)
                logger.info(f"Successfully inserted paper from {json_file.name}")
            except Exception as e:
                logger.error(f"Failed to process {json_file.name}: {e}")
                continue

    def _insert_paper(self, paper_data: Dict[str, Any]):
        """Insert a single paper and all related data into the database."""
        if not self.conn:
            raise Exception("Database connection not established")

        cur = self.conn.cursor()

        try:
            # Insert main paper record
            metadata = paper_data['metadata']
            content = paper_data['content']

            cur.execute("""
                INSERT INTO papers (
                    title, abstract, tags, experimental_setup_description,
                    experimental_results_summary, data_availability_statement,
                    code_repository_url, full_content
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
            """, (
                metadata['title'],
                metadata['abstract'],
                metadata['tags'],
                content['experimental_setup']['description'],
                content['experimental_results']['summary'],
                content['data_availability']['statement'],
                content['code_availability']['repository_url'],
                json.dumps(paper_data)
            ))

            paper_id = cur.fetchone()[0]

            # Insert authors
            for idx, author in enumerate(metadata['authors']):
                cur.execute("""
                    INSERT INTO authors (paper_id, name, author_order)
                    VALUES (%s, %s, %s);
                """, (paper_id, author['name'], idx))

            # Insert sections
            for idx, section in enumerate(content['sections']):
                cur.execute("""
                    INSERT INTO sections (paper_id, title, content, section_order)
                    VALUES (%s, %s, %s, %s);
                """, (paper_id, section['title'], section['content'], idx))

            # Insert figures
            for figure in content['figures']:
                cur.execute("""
                    INSERT INTO figures (paper_id, figure_id, caption)
                    VALUES (%s, %s, %s);
                """, (paper_id, figure['id'], figure['caption']))

            # Insert tables
            for table in content['tables']:
                cur.execute("""
                    INSERT INTO tables (paper_id, table_id, caption, headers, rows)
                    VALUES (%s, %s, %s, %s, %s);
                """, (paper_id, table['id'], table['caption'],
                     table['data']['headers'], table['data']['rows']))

            # Insert equations
            for equation in content['equations']:
                cur.execute("""
                    INSERT INTO equations (paper_id, equation_id, latex)
                    VALUES (%s, %s, %s);
                """, (paper_id, equation['id'], equation['latex']))

            # Insert references
            for reference in paper_data['references']:
                cur.execute("""
                    INSERT INTO "references" (paper_id, reference_id, title, authors)
                    VALUES (%s, %s, %s, %s);
                """, (paper_id, reference['id'], reference['title'], reference['authors']))

            # Generate and insert embeddings
            self._generate_embeddings(paper_id, paper_data, cur)

        except Exception as e:
            logger.error(f"Error inserting paper: {e}")
            raise
        finally:
            cur.close()

    def _split_text_into_chunks(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """Split text into chunks by paragraphs, keeping chunks under max_chunk_size."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            # If adding this paragraph would exceed max_chunk_size, save current chunk
            if current_chunk and len(current_chunk) + len(paragraph) + 2 > max_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add the last chunk if it exists
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Handle case where individual paragraphs are too long
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_chunk_size:
                final_chunks.append(chunk)
            else:
                # Split long paragraphs by sentences
                sentences = chunk.split('. ')
                temp_chunk = ""
                for sentence in sentences:
                    if temp_chunk and len(temp_chunk) + len(sentence) + 2 > max_chunk_size:
                        final_chunks.append(temp_chunk.strip())
                        temp_chunk = sentence
                    else:
                        if temp_chunk:
                            temp_chunk += ". " + sentence
                        else:
                            temp_chunk = sentence
                if temp_chunk.strip():
                    final_chunks.append(temp_chunk.strip())

        return final_chunks

    def _generate_embeddings(self, paper_id: int, paper_data: Dict[str, Any], cur):
        """Generate and store embeddings for the paper."""
        metadata = paper_data['metadata']
        content = paper_data['content']

        # Abstract embedding (single chunk)
        abstract_embedding = self.embedding_model.encode(metadata['abstract'])
        cur.execute("""
            INSERT INTO paper_embeddings (paper_id, embedding_type, chunk_index, embedding, text_content, source_section)
            VALUES (%s, %s, %s, %s, %s, %s);
        """, (paper_id, 'abstract', 0, abstract_embedding.tolist(), metadata['abstract'], 'abstract'))

        # Title embedding (single chunk)
        title_embedding = self.embedding_model.encode(metadata['title'])
        cur.execute("""
            INSERT INTO paper_embeddings (paper_id, embedding_type, chunk_index, embedding, text_content, source_section)
            VALUES (%s, %s, %s, %s, %s, %s);
        """, (paper_id, 'title', 0, title_embedding.tolist(), metadata['title'], 'title'))

        # Section content embeddings (chunked by paragraphs)
        for section in content['sections']:
            section_chunks = self._split_text_into_chunks(section['content'])
            for chunk_idx, chunk_text in enumerate(section_chunks):
                chunk_embedding = self.embedding_model.encode(chunk_text)
                cur.execute("""
                    INSERT INTO paper_embeddings (paper_id, embedding_type, chunk_index, embedding, text_content, source_section)
                    VALUES (%s, %s, %s, %s, %s, %s);
                """, (paper_id, 'section_content', chunk_idx, chunk_embedding.tolist(),
                     chunk_text, section['title']))

        # Experimental setup chunks
        if content['experimental_setup']['description']:
            exp_chunks = self._split_text_into_chunks(content['experimental_setup']['description'])
            for chunk_idx, chunk_text in enumerate(exp_chunks):
                chunk_embedding = self.embedding_model.encode(chunk_text)
                cur.execute("""
                    INSERT INTO paper_embeddings (paper_id, embedding_type, chunk_index, embedding, text_content, source_section)
                    VALUES (%s, %s, %s, %s, %s, %s);
                """, (paper_id, 'experimental_setup', chunk_idx, chunk_embedding.tolist(),
                     chunk_text, 'experimental_setup'))

        # Experimental results chunks
        if content['experimental_results']['summary']:
            results_chunks = self._split_text_into_chunks(content['experimental_results']['summary'])
            for chunk_idx, chunk_text in enumerate(results_chunks):
                chunk_embedding = self.embedding_model.encode(chunk_text)
                cur.execute("""
                    INSERT INTO paper_embeddings (paper_id, embedding_type, chunk_index, embedding, text_content, source_section)
                    VALUES (%s, %s, %s, %s, %s, %s);
                """, (paper_id, 'experimental_results', chunk_idx, chunk_embedding.tolist(),
                     chunk_text, 'experimental_results'))

    def vector_search(self, query: str, embedding_types: List[str] = ['abstract', 'section_content'],
                     limit: int = 5, similarity_threshold: float = 0.7,
                     max_chunks_per_paper: int = 3) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search on papers.

        Args:
            query: Search query text
            embedding_types: Types of embeddings to search (e.g., ['abstract', 'section_content'])
            limit: Maximum number of papers to return
            similarity_threshold: Minimum cosine similarity threshold
            max_chunks_per_paper: Maximum number of matching chunks to return per paper

        Returns:
            List of dictionaries containing paper information, similarity scores, and matching chunks
        """
        if not self.conn:
            raise Exception("Database connection not established")

        # Generate embedding for the query
        query_embedding = self.embedding_model.encode(query)

        cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Find the best matching chunks for each paper
        embedding_types_str = "', '".join(embedding_types)
        cur.execute(f"""
            WITH ranked_chunks AS (
                SELECT
                    p.id as paper_id,
                    p.title,
                    p.abstract,
                    p.tags,
                    pe.text_content,
                    pe.embedding_type,
                    pe.source_section,
                    pe.chunk_index,
                    1 - (pe.embedding <=> %s::vector) as similarity,
                    ROW_NUMBER() OVER (PARTITION BY p.id ORDER BY pe.embedding <=> %s::vector) as chunk_rank
                FROM papers p
                JOIN paper_embeddings pe ON p.id = pe.paper_id
                WHERE pe.embedding_type IN ('{embedding_types_str}')
                    AND 1 - (pe.embedding <=> %s::vector) >= %s
            ),
            paper_scores AS (
                SELECT
                    paper_id,
                    MAX(similarity) as max_similarity,
                    AVG(similarity) as avg_similarity,
                    COUNT(*) as matching_chunks
                FROM ranked_chunks
                WHERE chunk_rank <= %s
                GROUP BY paper_id
            )
            SELECT
                rc.paper_id,
                rc.title,
                rc.abstract,
                rc.tags,
                ps.max_similarity,
                ps.avg_similarity,
                ps.matching_chunks,
                JSON_AGG(
                    JSON_BUILD_OBJECT(
                        'text_content', rc.text_content,
                        'embedding_type', rc.embedding_type,
                        'source_section', rc.source_section,
                        'chunk_index', rc.chunk_index,
                        'similarity', rc.similarity
                    ) ORDER BY rc.similarity DESC
                ) as matching_chunks_details
            FROM ranked_chunks rc
            JOIN paper_scores ps ON rc.paper_id = ps.paper_id
            WHERE rc.chunk_rank <= %s
            GROUP BY rc.paper_id, rc.title, rc.abstract, rc.tags, ps.max_similarity, ps.avg_similarity, ps.matching_chunks
            ORDER BY ps.max_similarity DESC
            LIMIT %s;
        """, (query_embedding.tolist(), query_embedding.tolist(), query_embedding.tolist(),
              similarity_threshold, max_chunks_per_paper, max_chunks_per_paper, limit))

        results = cur.fetchall()
        cur.close()

        return [dict(row) for row in results]

    def text_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform traditional text search on papers using PostgreSQL full-text search.

        Args:
            query: Search query text
            limit: Maximum number of results to return

        Returns:
            List of dictionaries containing paper information and rank scores
        """
        if not self.conn:
            raise Exception("Database connection not established")

        cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("""
            SELECT
                p.id,
                p.title,
                p.abstract,
                p.tags,
                ts_rank(to_tsvector('english', p.title || ' ' || p.abstract),
                       plainto_tsquery('english', %s)) as rank
            FROM papers p
            WHERE to_tsvector('english', p.title || ' ' || p.abstract) @@
                  plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s;
        """, (query, query, limit))

        results = cur.fetchall()
        cur.close()

        return [dict(row) for row in results]

    def get_paper_details(self, paper_id: int) -> Optional[Dict[str, Any]]:
        """Get complete paper details by ID."""
        if not self.conn:
            raise Exception("Database connection not established")

        cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("SELECT * FROM papers WHERE id = %s;", (paper_id,))
        result = cur.fetchone()
        cur.close()

        return dict(result) if result else None

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

def setup_database(recreate_database: bool):
    """Initializes and sets up the database. If `recreate_database` is `True` it will
    drop all tables before recreating them."""
    # Database configuration
    config = DatabaseConfig(
        host="localhost",
        port=5432,
        database="scientific_papers",
        user="postgres",
        password="your_password" # XXX: might wanna change this...
    )

    # Initialize database manager
    db = ScientificPapersDB(config)

    # Connect to database
    db.connect()

    if recreate_database:
        # Setup database schema
        print("Setting up database schema...")
        db.setup_database(recreate_database)

        # Create indices
        print("Creating indices...")
        db.create_indices()

    return db

def add_json_files(db: ScientificPapersDB, json_directory: str):
    """Adds all JSON files from the given directory to the DB."""
    # Load JSON files from directory
    if os.path.exists(json_directory):
        print(f"Loading JSON files from {json_directory}...")
        db.load_json_files(json_directory)
    else:
        print(f"Directory {json_directory} not found. Skipping file loading.")

def example_usage(json_directory: str, recreate_database: bool):
    """Example usage of the ScientificPapersDB class."""
    try:
        db = setup_database(recreate_database)

        add_json_files(db, json_directory)

        # Example vector search queries
        print("\n" + "="*50)
        print("EXAMPLE VECTOR SEARCH QUERIES")
        print("="*50)

        # Example 1: Search abstracts and content for machine learning papers
        print("\n1. Searching for machine learning papers (abstracts + content):")
        results = db.vector_search(
            query="machine learning neural networks deep learning",
            embedding_types=["abstract", "section_content"],
            limit=3,
            similarity_threshold=0.3,
            max_chunks_per_paper=2
        )

        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (Max Similarity: {result['max_similarity']:.3f}, Avg: {result['avg_similarity']:.3f}):")
            print(f"Title: {result['title']}")
            print(f"Abstract preview: {result['abstract'][:150]}...")
            print(f"Tags: {result['tags']}")
            print(f"Matching chunks ({result['matching_chunks']} total):")
            for chunk in result['matching_chunks_details'][:2]:  # Show top 2 chunks
                print(f"  - {chunk['embedding_type']} ({chunk['source_section']}) [Sim: {chunk['similarity']:.3f}]")
                print(f"    {chunk['text_content'][:100]}...")

        # Example 2: Search experimental sections specifically
        print("\n2. Searching experimental methodology sections:")
        results = db.vector_search(
            query="experimental design methodology statistical analysis",
            embedding_types=["experimental_setup", "experimental_results"],
            limit=2,
            similarity_threshold=0.3
        )

        for i, result in enumerate(results, 1):
            print(f"\nExperimental Result {i} (Max Similarity: {result['max_similarity']:.3f}):")
            print(f"Title: {result['title']}")
            for chunk in result['matching_chunks_details']:
                print(f"  - {chunk['embedding_type']} [Sim: {chunk['similarity']:.3f}]")
                print(f"    {chunk['text_content'][:150]}...")

        # Example 3: Search only abstracts for comparison
        print("\n3. Abstract-only search:")
        results = db.vector_search(
            query="machine learning",
            embedding_types=["abstract"],
            limit=2,
            similarity_threshold=0.3
        )

        for i, result in enumerate(results, 1):
            print(f"\nAbstract Result {i} (Similarity: {result['max_similarity']:.3f}):")
            print(f"Title: {result['title']}")
            print(f"Abstract: {result['abstract'][:200]}...")

        # Example 4: Compare with traditional text search
        print("\n4. Traditional text search comparison:")
        text_results = db.text_search("machine learning", limit=2)

        for i, result in enumerate(text_results, 1):
            print(f"\nText Search Result {i} (Rank: {result['rank']:.3f}):")
            print(f"Title: {result['title']}")
            print(f"Abstract preview: {result['abstract'][:200]}...")

        print("\nSearch examples completed!")
        print("\nKey improvements with chunked embeddings:")
        print("- No information loss from truncation")
        print("- Better semantic matching across document sections")
        print("- Ability to see which specific parts of papers match")
        print("- More precise similarity scoring per chunk")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise
    finally:
        db.close()

def main(json_directory: str = "../data",
         print_db_setup_help: bool = False,
         run_example: bool = False,
         setup_db: bool = False,
         add_json_files: bool = False,
         query: str = "",
         embedding_types: List[str] = ['abstract', 'section_content'],
         limit: int = 5,
         similarity_threshold: float = 0.7,
         max_chunks_per_paper: int = 3
         ):
    print("Scientific Papers Database with PostgreSQL + pgvector")
    print("=" * 55)
    print("\nThis script implements chunked embeddings for better semantic search!")

    if print_db_setup_help:
        print("\nDocker Setup Instructions:")
        print_postgres_setup_help()

    print("\n" + "="*55)
    print("Python Requirements:")
    print("pip install psycopg2-binary sentence-transformers numpy")
    print()
    print("Before running main(), ensure you have:")
    print("1. PostgreSQL running (see Docker instructions above)")
    print("2. Updated the database configuration in the script")
    print("3. A directory with JSON files following the scientific paper schema")

    response = input("Do you want to run the main function now? (y/n): ")
    if response.lower() != 'y':
        return

    if run_example:
        example_usage(json_dictionary, setup_db)

    if setup_db:
        # Recreate the entire database. Will drop *everything*!
        db = setup_database(True)
        db.close()

    if add_json_files:
        db = setup_database(False) # do not recreate the DB!
        add_json_files(db, json_directory)

    if len(query):
        db = setup_database(False) # do not recreate the DB!
        result = db.vector_search(query, embedding_types, limit, similarity_threshold, max_chunks_per_paper)

        for r in result:
            print(json.dumps(r, indent=2))

if __name__ == "__main__":
    typer.run(main)

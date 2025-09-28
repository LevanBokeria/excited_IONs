from google import genai
from pydantic import BaseModel
import os
import json
import copy
import typer
import subprocess

import argparse

# TODO: Implement this and test
PROJECT_CONTEXT = """You are an expert in extracting structured data from scientific articles. You extract relevant information, do not skip anything"""


def pdf_to_mardown(pdf_path: str) -> str:
    """
    This is a function that takes in a paper as PDF and converts it to markdown with docling
    """
    import logging

    logger = logging.getLogger(__name__)

    # Test to see that docling is installed
    result = subprocess.run(["docling", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(
            "Docling is not installed or not found in PATH. Please install it https://github.com/docling-project/docling"
        )
        raise EnvironmentError(
            "Docling is not installed or not found in PATH. Please install it https://github.com/docling-project/docling"
        )

    markdown_path = pdf_path.replace("pdf", "md")  # TODO: Fix edge case of .pdf.pdf lol

    if os.path.exists(markdown_path):
        logger.info(
            f"Markdown file {markdown_path} already exists. Using existing file."
        )
        print(f"Markdown file {markdown_path} already exists. Using existing file.")
        return markdown_path
    logger.info(f"Running docling to convert {pdf_path} to markdown.")

    num_threads = os.cpu_count() - 1 if os.cpu_count() else 2

    result = subprocess.run(
        [
            "docling",
            pdf_path,
            "--from",
            "pdf",
            "--to",
            "md",
            "--image-export-mode",
            "referenced",
            "--num-threads",
            str(num_threads),
            "--output",
            str(os.path.dirname(markdown_path)),
            # TODO: In the future try out the vlm
            # "--pipeline",
            # "vlm",
            # "--vlm-model",
            # "granite_docling",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"Error running docling: {result.stderr}")
        print("Error running docling:", result.stderr)
        raise RuntimeError("Docling conversion failed")


def resolve_refs(schema, root=None, seen=None):
    if root is None:
        root = schema
    if seen is None:
        seen = set()

    if isinstance(schema, dict):
        if "$ref" in schema:
            ref_path = schema["$ref"]
            if ref_path in seen:
                # Already resolving this ref → stop to prevent infinite recursion
                return schema
            seen.add(ref_path)

            if not ref_path.startswith("#/"):
                raise ValueError("Only local refs are supported")

            parts = ref_path[2:].split("/")
            target = root
            for part in parts:
                target = target[part]

            # Recursively resolve target, passing the updated seen set
            resolved = resolve_refs(copy.deepcopy(target), root, seen)
            seen.remove(ref_path)
            return resolved
        else:
            return {k: resolve_refs(v, root, seen) for k, v in schema.items()}

    elif isinstance(schema, list):
        return [resolve_refs(item, root, seen) for item in schema]

    else:
        return schema


import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def split_markdown_sections(text):
    """
    Splits markdown text into sections based on headings (## or #).
    Returns a list of (section_title, section_text, start_idx, end_idx).
    """
    import re

    sections = []
    matches = list(re.finditer(r"^(#+ .+)$", text, flags=re.MULTILINE))
    if not matches:
        return [("Full Document", text, 0, len(text))]
    for i, match in enumerate(matches):
        start = match.start()
        title = match.group(1).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end]
        sections.append((title, section_text, start, end))
    return sections


def section_has_schema_info(section_text, schema_keys, schema, client):
    """
    Ask a small LLM if this section contains info about any schema keys.
    Returns a list of relevant keys.
    """
    # Build a list of key: description for the prompt
    key_descs = [
        f"{key}: {schema['properties'][key].get('description', '')}"
        for key in schema_keys
    ]
    prompt = (
        "Given the following section of a scientific article, which of these schema fields does it contain relevant information for?\n"
        "Schema fields and descriptions:\n" + "\n".join(key_descs) + "\nSection:\n"
        f"{section_text}\n"
        "Respond as a JSON list of relevant field names."
    )
    logging.info(
        "Querying small LLM for schema info in section (first 80 chars): %r",
        section_text[:80],
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    try:
        result = json.loads(response.text)
        logging.info("Small LLM response: %r", result)
        return result
    except Exception as e:
        logging.warning("Failed to parse small LLM response: %s", e)
        return []


def extract_field_from_section(field, section_text, field_schema, client):
    """
    Use the main LLM to extract the field from the section text.
    """
    logging.info(
        "Extracting field '%s' from section (first 80 chars): %r",
        field,
        section_text[:80],
    )
    ## XXX: I think it's probably wise to prefix the text of the section with some kind
    ## of prompt here as well!
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=section_text,
        config={
            "response_mime_type": "application/json",
            "response_schema": field_schema,
        },
    )
    try:
        value = response.parsed[field]
        logging.info("Main LLM extracted value for '%s': %r", field, value)
        return value
    except Exception as e:
        logging.warning("Failed to extract field '%s': %s", field, e)
        return None


def main(schema_path: str, paper_path: str):
    api_key = (
        os.getenv("GEMINI_API_KEY")
        if os.getenv("GEMINI_API_KEY")
        else os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)

    with open(schema_path, "r") as f:
        schema = json.load(f)
    schema = resolve_refs(schema)
    schema_keys = list(schema["properties"].keys())
    logging.info("Loaded schema keys: %r", schema_keys)

    if paper_path.endswith(".pdf"):
        paper_md = pdf_to_mardown(paper_path)
    elif paper_path.endswith(".md"):
        paper_md = paper_path
    else:
        # NOTE: This should never happen due to earlier checks
        raise ValueError("Paper file must be .pdf or .md")

    with open(paper_md, "r") as f:
        paper_text = f.read()

    logging.info("Loaded paper text (%d chars)", len(paper_text))

    # Step 1: Split into sections
    sections = split_markdown_sections(paper_text)
    logging.info("Split paper into %d sections", len(sections))

    # Step 2: For each section, ask which schema keys it covers
    section_key_map = {}
    for idx, (title, section_text, start, end) in enumerate(sections):
        logging.info("Processing section %d: %s", idx, title)
        relevant_keys = section_has_schema_info(
            section_text, schema_keys, schema, client
        )
        for key in relevant_keys:
            section_key_map.setdefault(key, []).append(
                (title, section_text, start, end)
            )
            logging.info("Section %d (%s) relevant for key: %s", idx, title, key)

    # Step 3: For each key, merge all relevant section texts and extract info in one LLM call
    output = {}
    total_tokens = 0  # Track aggregate token usage

    def count_tokens(text):
        # Simple whitespace token count; replace with tokenizer if available
        return len(text.split())

    for key in schema_keys:
        field_schema = {
            "title": schema.get("title"),
            "type": "object",
            "properties": {key: schema["properties"][key]},
            "required": [key] if key in schema.get("required", []) else [],
        }
        key_sections = section_key_map.get(key, [])
        logging.info(
            "Extracting key '%s' from %d relevant sections", key, len(key_sections)
        )
        if not key_sections:
            output[key] = None
            logging.info("No value found for key '%s'", key)
            continue

        # Merge all relevant section texts for this key
        merged_section_text = "\n\n".join(
            section_text for _, section_text, _, _ in key_sections
        )
        total_tokens += count_tokens(merged_section_text)

        value = extract_field_from_section(
            key, merged_section_text, field_schema, client
        )
        if value is not None:
            output[key] = value
        else:
            output[key] = None

    outfile = paper_path.replace(".md", ".json")  # TODO: Fix edge case of .md.md lol
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    logging.info("Structured output written to output.json")
    logging.info("Aggregate token usage (approx): %d", total_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a scientific paper PDF to structured JSON using Gemini."
    )
    parser.add_argument("schema_path", type=str, help="Path to the JSON schema file.")
    parser.add_argument(
        "paper_path", type=str, help="Path to the paper file (pdf or markdown)."
    )
    args = parser.parse_args()
    schema_path = args.schema_path
    paper_path = args.paper_path
    assert schema_path.endswith(".json"), "Schema file must be a .json file"

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    if not os.path.exists(paper_path):
        raise FileNotFoundError(f"Paper file not found: {paper_path}")

    typer.run(lambda _: main(schema_path, paper_path))

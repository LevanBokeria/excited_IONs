#!/usr/bin/env python3
import json
import subprocess
import tempfile
import shutil
import sys
import typer
from pathlib import Path
from openai import OpenAI

PROMPT = """
Your task is to convert the given PDF into JSON following the given schema.

Be accurate and do not ignore any information!

It is very important to be extensive in the tags. They should cover topics discussed in the paper, materials, chemicals etc used, type of paper (experimental, theoretical, ...) and so on. These tags will serve as one of the fundamental ways to filter the resulting corpus of data.
"""

def pdf_to_text(pdf_path: str) -> str:
    """Convert PDF to text using pdftotext command."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_pdf = Path(temp_dir) / "test.pdf"
        temp_txt = Path(temp_dir) / "test.txt"

        # Copy PDF to temp directory
        shutil.copy2(pdf_path, temp_pdf)

        # Run pdftotext
        subprocess.run([
            "pdftotext", "-layout", str(temp_pdf)
        ], check=True, cwd=temp_dir)

        # Read the result
        return temp_txt.read_text(encoding='utf-8')

def get_api_key() -> str:
    """Get OpenAI API key from environment or file."""
    import os

    # Try environment variable first
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        return api_key

    # Fallback to reading from a simple text file
    try:
        return Path.home().joinpath('.openai_key').read_text().strip()
    except FileNotFoundError:
        raise ValueError("Please set OPENAI_API_KEY environment variable or create ~/.openai_key file")

def stream_openai_response(client: OpenAI, messages: list, schema: dict) -> str:
    """Stream the OpenAI response and return the complete content."""
    print("🤖 Streaming OpenAI response...")

    stream = client.chat.completions.create(
        model="gpt-5-nano",
        messages=messages,
        response_format={"type": "json_schema", "json_schema": {
            "name" : "pdfToJson",
            "strict" : True,
            "schema" : schema}
                         },
        stream=True
    )

    collected_content = ""

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content_chunk = chunk.choices[0].delta.content
            collected_content += content_chunk

            # Show streaming progress (optional - you can remove this)
            print(".", end="", flush=True)

    print("\n✅ Streaming complete!")
    return collected_content

def main(filename: str, schema: str = '../schemas/basic_schema_openai.json',
         show_stream: bool = True):
    # Get API key
    api_key = get_api_key()
    print(f"API Key: {api_key[:10]}...")

    # Load schema
    with open(schema, 'r') as f:
        schema = json.load(f)

    # Convert PDF to text
    paper_text = pdf_to_text(filename)

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    print(f"Sending OpenAI request: {len(paper_text)} characters")

    # Prepare messages
    messages = [
        {"role": "system", "content": "You are a data extractor."},
        {"role": "user", "content": PROMPT + paper_text}
    ]

    # Stream the response
    if show_stream:
        # Alternative version that shows content as it streams
        json_response = stream_with_live_display(client, messages, schema)
    else:
        # Standard version with progress dots
        json_response = stream_openai_response(client, messages, schema)

    # Parse and pretty print the final JSON
    try:
        parsed_json = json.loads(json_response)
        print("\n" + "="*50)
        print("FINAL RESULT:")
        print("="*50)
        js = json.dumps(parsed_json, indent=2)
        print(js)

        # now write to file
        outfile = filename.replace(".pdf", ".json")
        with open(outfile, "w") as f:
            f.write(js)
    except json.JSONDecodeError as e:
        print(f"\n❌ Error parsing JSON response: {e}")
        print("Raw response:")
        print(json_response)

def stream_with_live_display(client: OpenAI, messages: list, schema: dict) -> str:
    """Stream with live display of content as it arrives."""
    print("🤖 Streaming OpenAI response (live display):")
    print("-" * 50)

    stream = client.chat.completions.create(
        model="gpt-5-nano",
        messages=messages,
        response_format={"type": "json_schema", "json_schema": {
            "name" : "pdfToJson",
            "strict" : True,
            "schema" : schema}
                         },
        stream=True
    )

    collected_content = ""

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content_chunk = chunk.choices[0].delta.content
            collected_content += content_chunk

            # Print content as it arrives
            print(content_chunk, end="", flush=True)

    print("\n" + "-" * 50)
    print("✅ Streaming complete!")
    return collected_content

if __name__ == "__main__":
    typer.run(main)

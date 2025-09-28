from feedfetchtest import fetch_recent_articles

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from a .env file

# Fetch papers from the last 24 hours without downloading PDFs
articles = fetch_recent_articles(
    "scripts/michaels_scripts/mccayfeeds.opml", 
    hours=24*100, 
    download_pdfs=True  # This skips PDF download
)
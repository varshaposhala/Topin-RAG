# Topin Global Question Engine

A Streamlit-based search interface for educational questions powered by Qdrant and Hugging Face embeddings.

## Overview

This app enables natural-language question search across a question bank. It supports subject, topic, tag, and difficulty filtering, and can parse queries such as:

- `give me 10 python coding questions`
- `show me sql mcqs`
- `node js coding questions`

## Files

- `app.py` - main Streamlit application
- `requirements.txt` - Python dependencies
- `topin_cleaned_data.csv` - source dataset used by the question index

## Prerequisites

- Python 3.10+ installed
- `pip` for package installation
- Qdrant instance accessible via `QDRANT_URL` and `QDRANT_API_KEY`
- Optional: `HF_TOKEN` for Hugging Face model access if rate limits are needed
- Optional: `OPENROUTER_API_KEY` if the app uses OpenRouter for intent parsing

## Setup

1. Clone the repository:

```bash
git clone <repo-url>
cd "Topin Rag bot"
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

The app reads secrets from Streamlit secrets or environment variables. Create a file at `.streamlit/secrets.toml` or set environment variables directly.

Example `.streamlit/secrets.toml`:

```toml
QDRANT_URL = "https://your-qdrant-instance"
QDRANT_API_KEY = "your-qdrant-api-key"
OPENROUTER_API_KEY = "your-openrouter-api-key"
HF_TOKEN = "your-huggingface-token"
```

If you prefer environment variables, set:

- `QDRANT_URL`
- `QDRANT_API_KEY`
- `OPENROUTER_API_KEY`
- `HF_TOKEN`

## Run the App

Start the Streamlit app with:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Usage

Type a natural-language query into the app input. Example queries:

- `all python coding questions`
- `give me java mcqs`
- `show me reactjs questions`

The app parses the query and returns matching questions from the indexed dataset.

## Notes

- The app relies on Qdrant collections, so make sure your Qdrant instance is reachable and populated.
- Query parsing supports subject and tag filters, but accuracy depends on matching data in the CSV index.





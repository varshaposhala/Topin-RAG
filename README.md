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

## How Search Works

When a user submits a query, the app runs multiple steps behind the scenes to convert the text into an exact search request.

### 1. Query normalization
The raw text is normalized by lowercasing, removing extra whitespace, and breaking the query into searchable tokens. This ensures the search logic treats `Python`, `python`, and `PYTHON` the same.

### 2. Subject, type, and count detection
The app scans the query for known subjects like `python`, `java`, `sql`, `reactjs`, and `nodejs`. It also detects question types such as:

- `coding`
- `mcq`
- `coding analysis`
- `mixed`

If the query includes a count like `10`, `5`, or `all`, the app records that too.

### 3. Tag extraction and filtering
Structured tags and curriculum tags are extracted from the query only when they match real catalog tags. The app avoids false matches on common words like `questions` or `coding` when they are not tag values.

### 4. Intent building
The parsed subject, question type, difficulty, count, tags, and topic keywords are combined into a single intent object. This object describes exactly what the user wants.

### 5. Collection selection
Using the intent, the app selects the appropriate Qdrant collections to search. For example, a `python coding` request will target `topic_python_coding` and related Python collections, while `sql mcqs` will target SQL MCQ collections.

### 6. Semantic search with embeddings
If the query is not purely tag-based, the app uses Hugging Face embeddings to convert the query into a vector and compares it against stored question vectors in Qdrant. This finds the most relevant rows even when the query wording differs from the exact stored text.

### 7. Result filtering and ranking
The app filters matched rows by the detected intent fields and ranks them by relevance. This ensures returned questions match the subject/type and are the best semantic fit.

### 8. Display
Finally, the matching questions are displayed in the Streamlit UI with a friendly label describing the query results.

This detailed pipeline ensures subject-wise, tag-wise, and field-wise searches work reliably and return the correct rows from the indexed dataset.

## Notes

- The app relies on Qdrant collections, so make sure your Qdrant instance is reachable and populated.
- Query parsing supports subject and tag filters, but accuracy depends on matching data in the CSV index.





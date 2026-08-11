# Topin Global Question Engine

A Streamlit-based search interface for educational questions powered by Pinecone and Hugging Face embeddings.

## Overview

This app enables natural-language question search across a question bank. It supports subject, topic, tag, and difficulty filtering, and can parse queries such as:

- `give me 10 python coding questions`
- `show me sql mcqs`
- `node js coding questions`

## Files

- `app.py` - main Streamlit application
- `pinecone_db.py` - Pinecone database adapter
- `reindex_pinecone.py` - upload CSV questions into Pinecone
- `requirements.txt` - Python dependencies
- `topin_cleaned_data.csv` - source dataset used by the question index

## Prerequisites

- Python 3.10+ installed
- `pip` for package installation
- Pinecone account + API key (`PINECONE_API_KEY`)
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

Create `.streamlit/secrets.toml`:

```toml
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_INDEX_NAME = "topin-questions"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
OPENROUTER_API_KEY = "your-openrouter-api-key"
data_link = "https://your-csv-url/topin_cleaned_data.csv"
```

## Load data into Pinecone

```bash
python -u reindex_pinecone.py
```

Test with a small sample first:

```bash
python -u reindex_pinecone.py --limit 200
```

Then start the app:

```bash
streamlit run app.py
```

`topin_cleaned_data.csv` is the source of truth. If Pinecone data is lost, re-run the reindex script.

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
Using the intent, the app selects the appropriate Pinecone namespaces to search. For example, a `python coding` request will target `topic_python_coding_questions` and related Python namespaces, while `sql mcqs` will target SQL MCQ namespaces.

### 6. Semantic search with embeddings
If the query is not purely tag-based, the app uses Hugging Face embeddings to convert the query into a vector and compares it against stored question vectors in Pinecone. This finds the most relevant rows even when the query wording differs from the exact stored text.

### 7. Result filtering and ranking
The app filters matched rows by the detected intent fields and ranks them by relevance. This ensures returned questions match the subject/type and are the best semantic fit.

### 8. Display
Finally, the matching questions are displayed in the Streamlit UI with a friendly label describing the query results.

This detailed pipeline ensures subject-wise, tag-wise, and field-wise searches work reliably and return the correct rows from the indexed dataset.

## Notes

- The app relies on a populated Pinecone index, so make sure `PINECONE_API_KEY` is set and `reindex_pinecone.py` has been run.
- Query parsing supports subject and tag filters, but accuracy depends on matching data in the CSV index.





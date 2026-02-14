# Vision Data Image Cleaner

LLM-powered pipeline for image de-duplication, validation, and dataset
cleaning.

A configurable FastAPI + CLI tool to clean large-scale survey image
datasets using **pHash + embeddings + multi-LLM validation**.\
Built for rural survey pipelines, geospatial data collection, and
large-scale field datasets.

------------------------------------------------------------------------

## Features

-   Exact duplicate detection using pHash\
-   Near-duplicate detection using image embeddings\
-   Duplicate verification using OpenAI / Gemini / Claude\
-   Canonical image selection using survey timestamp logic\
-   Label validation with multi-LLM consensus\
-   Automatic grouping into:
    -   Clean representative images\
    -   Confirmed duplicates\
    -   Images needing manual review\
-   Config-driven pipeline\
-   FastAPI service + CLI support\
-   CSV reports for auditing

------------------------------------------------------------------------

## Installation

``` bash
git clone https://github.com/your-org/vision-data-cleaning-pipeline.git
cd vision-data-cleaning-pipeline

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

------------------------------------------------------------------------

## Running API

``` bash
python -m uvicorn app.main:app --reload
```

Open: http://127.0.0.1:8000/docs

------------------------------------------------------------------------

## Config Example

configs/default.yaml

``` yaml
input:
  image_folder: ./data/images

llm:
  provider: openai
  model: gpt-4o-mini
  api_key_env: OPENAI_API_KEY

duplicate_detection:
  similarity_threshold: 0.85

output:
  base_folder: ./outputs
```

------------------------------------------------------------------------

## Environment Variables

Create `.env` file:

    OPENAI_API_KEY=xxxx
    GEMINI_API_KEY=xxxx
    CLAUDE_API_KEY=xxxx

Do NOT commit `.env` to GitHub.

------------------------------------------------------------------------

## Outputs

    outputs/run_001/
    ├── clean_images/
    ├── duplicates/
    ├── manual_review/
    └── reports/

------------------------------------------------------------------------

## Developed By

Built as part of data systems work at\
Bharti Institute of Public Policy (ISB)

Maintained by Nitesh Saini.

------------------------------------------------------------------------

## License

MIT License

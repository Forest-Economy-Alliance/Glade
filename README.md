# Glade

Configurable FastAPI pipeline for image de-duplication and dataset preparation.

## Project Phases

### Phase 1 (Current): Duplicate Detection And Unique Image Generation

Phase 1 detects exact and near-duplicates, verifies near-duplicate groups with LLM, merges overlapping duplicate groups, extracts one original per duplicate cluster, and prepares final unique-image outputs for downstream processing.

### Phase 2: Label Validation And Quality Check

Phase 2 runs LLM-based validation on Phase 1 unique outputs:

- Choose provider per run: OpenAI, Gemini, or Claude
- Dataset-specific prompt design (system and user prompts are fully configurable)
- Model JSON output extraction using configurable output keys
- Pass/fail based on configurable expected values for each output key

## Installation

```bash
git clone https://github.com/Forest-Economy-Alliance/vision-data-cleaning-pipeline
cd vision-data-cleaning-pipeline

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

## Run API

```bash
uvicorn app.main:app --reload --port 8000
```

Open: http://127.0.0.1:8000/docs

## UI Dashboard

After starting the API, open:

- `http://127.0.0.1:8000/ui` to manage config values and output CSV file names, then run pipeline
- `http://127.0.0.1:8000/ui/docs` to read methods and technique documentation

The UI provides separate run actions:

- Run Phase 1 to generate `unique_images`
- Run Phase 2 independently on a configurable input folder (defaults to `output/unique_images`)

UI coverage includes:

- Input/output paths
- LLM provider/model/toggles
- Near-duplicate method settings
- Metadata fallback mapping columns
- Root CSV output file naming
- Phase 2 LLM provider, prompts, output schema, and expected-value settings

## Configuration

Main config file: `configs/default.yaml`

Important sections:

- `input.image_folder`: source images folder
- `output.base_folder`: output root folder
- `near_duplicates.active_method`: active near-duplicate method
- `similarity_check_llm.*`: LLM verification settings
- `metadata.*`: fallback metadata CSV for original selection
- `phase2.*`: LLM label validation and quality checks after unique image generation

Important Phase 2 fields:

- `phase2.input_folder`: folder of images to validate in Phase 2
- `phase2.llm.*`: provider/model/api key env/prompts

Example metadata section:

```yaml
metadata:
  csv_path: null
  image_name_column: image_name
  datetime_column: hh_information-datetime
```

## Phase 1 Flow

1. Exact duplicate detection creates `exact_duplicates` groups.
2. Near-duplicate detection creates method-specific groups.
3. LLM verifies near-duplicate groups.
4. LLM-confirmed groups are copied into `exact_duplicates` with `llm_` prefix.
5. Overlapping duplicate groups are merged into `exact_duplicates_merged`.
6. One original image is selected per merged group using earliest timestamp:
   - EXIF datetime first
   - metadata CSV datetime fallback
   - file modified time last fallback
7. Selected originals are copied to `duplicates_original`.
   - Copied filenames are made unique using `group_id` prefix (for example `merged_group_0003_IMG_1234.jpg`) to prevent overwrite.
8. Final non-duplicate images are copied to `unique_images`.

## Phase 1 Outputs

All CSV files are written to output root.

- `exact_duplicates.csv`
- `near_duplicate_groups_hsv_cosine.csv` or `near_duplicate_groups.csv` (depends on active method)
- `near_duplicates.csv`
- `near_duplicates_llm_verified.csv`
- `exact_duplicate_groups.csv`
- `merged_groups_summary.csv`
- `originals_extraction_summary.csv`
- `unique_images.csv`

## Phase 2 Outputs (when enabled)

All CSV files are written to output root.

- `phase2_validation_report.csv`: per-image LLM output fields + pass/fail + raw JSON
- `phase2_failed_images.csv`: images that do not match expected output values or failed validation
- `phase2_summary.csv`: aggregate counters and selected provider/model details

Folders:

- `exact_duplicates/`
- `exact_duplicates_merged/`
- `duplicates_original/`
- `unique_images/`

## Environment Variables

Use `.env` for secrets (example):

```text
OPENAI_API_KEY=xxxx
GEMINI_API_KEY=xxxx
CLAUDE_API_KEY=xxxx
```

Do not commit `.env`.

## License

MIT

# AI Job Analyzer

A Python CLI application that analyzes a job offer, compares it with a resume, and suggests resume improvements **without inventing experience**.

## Goals

1. **Analyze a job offer**: extract the job title, company, contract type, skills, responsibilities, required experience, location, salary, and benefits.
2. **Compare the offer with a resume** and compute a match score out of 100.
3. **Suggest resume improvements** that stay faithful to the candidate's actual experience.

## Architecture principles

- The **LLM** extracts and interprets the job offer text.
- **Python** validates, normalizes, deduplicates, and computes the score.
- **Pydantic** defines the data contract between the two.
- All code (variables, classes, functions) is written in English.

## Progress

- [x] Project structure
- [ ] Pydantic data models (`models.py`) and their tests
- [ ] LLM client (`llm_client.py`)
- [ ] Job offer analysis (`analyzer.py`)
- [ ] Command-line interface (`cli.py`)
- [ ] Resume comparison and score out of 100
- [ ] Resume improvement suggestions

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in the variables (API key, etc.)
```

## Usage

```bash
python3 main.py
```

## Tests

```bash
pytest
```

## Project structure

```
ai-job-analyzer/
├── main.py               # program entry point
├── job_analyzer/
│   ├── __init__.py
│   ├── models.py         # Pydantic models (data contract)
│   ├── llm_client.py     # LLM calls
│   ├── analyzer.py       # extraction, validation, and normalization
│   └── cli.py            # command-line interface
├── tests/
│   └── test_models.py    # model tests
├── job_offers/           # job offers as text files
│   └── example.txt
├── requirements.txt      # Python dependencies
├── .env.example          # configuration template (no secrets)
├── .gitignore
└── README.md
```

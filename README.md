# DrawBook-AI-API

FastAPI-based API service for DrawBook AI.

## Quick Start

- Python 3.10+ recommended
- Install dependencies (use your env manager of choice)

```bash
# create / activate your env (example)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# install packages (add your requirements as needed)
pip install fastapi uvicorn
```

Run the API locally:

```bash
python main.py
# App serves at http://0.0.0.0:8000
```

## Project Structure

- `main.py`: Uvicorn entrypoint
- `api/`: FastAPI routes and logic
- `static/`: Static assets
- `output/`: Generated outputs (git-ignored)

## Notes

- `.gitignore` excludes virtualenvs, caches, and `output/`.
- Adjust dependencies and docs as the project evolves.

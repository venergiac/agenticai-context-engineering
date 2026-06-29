# LangGraph + Ollama + Langfuse Sample

This sample project shows two graph versions:

- Version A: direct Ollama generation using `langgraph` and `langfuse` tracing.
- Version B: prompt compression with `LLMLingua` before Ollama generation, also traced by `langfuse`.

## Files

- `app.py` - example orchestrator and agent graph.
- `requirements.txt` - Python dependencies.
- `docker-compose.yml` - stubbed services for Ollama and Langfuse.
- `.gitignore` - ignores virtualenv and generated Python artifacts.

## Setup

1. Create or activate the Python virtual environment.
2. Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

3. Ensure Ollama CLI is installed and the Docker daemon is running if using `docker compose`.

## Run

Version A (direct Ollama):

```powershell
python app.py --version a --prompt "Write a short product description for a solar-powered backpack."
```

Version B (LLMLingua compression):

```powershell
python app.py --version b --prompt "Write a short product description for a solar-powered backpack."
```

If using a custom local Ollama model, pass `--model`.

## Langfuse [OPTIONAL]

If you want to use LangFuse locally you can use docker compose to install it.

See https://langfuse.com/self-hosting/deployment/docker-compose to install local langfuse.

Set Langfuse environment variables before running the app:

```powershell
set LANGFUSE_BASE_URL=http://localhost:8080
set LANGFUSE_PUBLIC_KEY=local_public_key
set LANGFUSE_SECRET_KEY=local_secret_key
```

If you want to use a local Langfuse service with Docker Compose, update `docker-compose.yml` and set `LANGFUSE_BASE_URL` to `http://localhost:8080`.



## Docker Compose

Bring up the services:

```powershell
docker compose up -d
docker compose exec ollama ollama pull qwen2.5:7b-instruct
```

> Note: This workspace currently cannot verify Docker Compose execution because the Docker daemon is unavailable in this session.

## Notes

- The LLMLingua version uses `PromptCompressor` and may require a compatible Hugging Face model.
- The local Ollama model name defaults to `qwen2.5:7b-instruct`.
- The sample uses `@observe` spans to capture Langfuse trace data for direct and compressed generation steps.

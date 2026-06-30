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

## Docker Compose

Bring up the services:

```powershell
docker compose up -d
docker compose exec ollama ollama pull qwen2.5:7b-instruct
```

## Run

```powershell
python app.py --version llmlingua --prompt "What are the three pillars of Physical AI mentioned in the text, and why is the Sim-to-Real approach necessary?" --context "For decades, artificial intelligence has lived confined within screens, cloud servers, and digital software. 
Language models process words and image generators manipulate pixels at extraordinary speeds, but they lack direct 
interaction with the tangible world. Today, we are witnessing the most crucial technological transition of the century: 
the birth of Physical AI. This discipline represents the definitive convergence of advanced computational intelligence 
and the material world, enabling artificial systems to perceive, understand, move, and act autonomously within 
three-dimensional physical space.

Unlike purely digital AI, Physical AI must contend with the immutable laws of physics: gravity, friction, inertia, 
fluid dynamics, and the sheer unpredictability of unstructured environments. It is not simply a matter of installing 
smart software into an old industrial robot. Physical AI requires a holistic design where hardware (biomimetic sensors, 
high-precision actuators, compliant materials) and software (multimodal neural networks, reinforcement learning, 
physics-aware simulations) evolve and operate together as a single organism.

The foundational pillars of Physical AI include:
1. Advanced Multimodal Perception: Systems fuse data from LiDAR, tactile sensors simulating human skin, pressure sensors, and inertial units.
2. Simulation-to-Real Evolution (Sim-to-Real): Training a robot in the real world is expensive and dangerous. Engineers leverage hyper-realistic digital simulations to let the AI attempt tasks millions of times in seconds, then transfer this intelligence into the physical robot.
3. Physical Commonsense Reasoning: A physical AI must grasp real-world cause and effect (e.g., understanding that a glass vase will shatter if dropped).

The application fields are vast: humanoid robotics, domestic assistants, smart prosthetics, micro-robotic surgery, Level 5 autonomous driving, and construction automation.
However, massive challenges remain: safety engineering, ethical alignment, and energy efficiency bottlenecks tied to local Edge Computing. 
Despite these hurdles, Physical AI marks the end of AI as a mere 'text oracle' and ushers in the era of machines capable of tangibly reshaping reality.
"
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


## Notes

- The LLMLingua version uses `PromptCompressor` and may require a compatible Hugging Face model.
- The local Ollama model name defaults to `qwen2.5:7b-instruct`.
- The sample uses `@observe` spans to capture Langfuse trace data for direct and compressed generation steps.

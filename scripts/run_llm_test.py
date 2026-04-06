"""Run LLM stress test with specified model. Usage: python scripts/run_llm_test.py [qwen|nemotron|openai]"""
import os
import sys
from pathlib import Path

# Load .env file if present
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip("'\""))

model_presets = {
    "qwen": ("Qwen/Qwen2.5-72B-Instruct", "https://router.huggingface.co/v1"),
    "nemotron": ("nvidia/Llama-3.1-Nemotron-70B-Instruct-HF", "https://router.huggingface.co/v1"),
    "openai": ("gpt-4o-mini", "https://api.openai.com/v1"),
}

preset = sys.argv[1] if len(sys.argv) > 1 else "nemotron"
model_name, base_url = model_presets.get(preset, model_presets["qwen"])

hf_token = os.environ.get("HF_TOKEN", "")
if not hf_token:
    print("ERROR: HF_TOKEN not found. Set it in .env or as environment variable.")
    sys.exit(1)
os.environ["HF_TOKEN"] = hf_token
os.environ["MODEL_NAME"] = model_name
os.environ["API_BASE_URL"] = base_url
os.environ.setdefault("ENV_BASE_URL", "http://localhost:7860")

print(f"Running stress test with model={model_name} base_url={base_url}")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.stress_test import main
main()

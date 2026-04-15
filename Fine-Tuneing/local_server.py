"""
Fine-Tuneing/local_server.py

Serves a HuggingFace safetensors model as an OpenAI-compatible HTTP API.
Bypasses llama.cpp/GGUF conversion entirely — runs directly from the
merged finetuned-Qwen3/ directory.

Usage:
    python Fine-Tuneing/local_server.py --model finetuned --port 8080
    python Fine-Tuneing/local_server.py --model base --port 8081
"""

import argparse
import json
import time
import threading
from pathlib import Path

import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
MODELS = {
    "finetuned": PROJECT_ROOT / "Fine-Tuneing" / "finetuned-Qwen3",
    "base":      PROJECT_ROOT / "Fine-Tuneing" / "Qwen3",
}

MAX_NEW_TOKENS = 1024
TEMPERATURE    = 0.7
REPEAT_PENALTY = 1.3

# Global model + lock (one inference at a time on single GPU)
model      = None
tokenizer  = None
model_lock = threading.Lock()

app = Flask(__name__)


def load_model(model_path: str):
    global model, tokenizer
    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    print("Model loaded.")


def generate(messages: list[dict]) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            repetition_penalty=REPEAT_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    body     = request.get_json(force=True)
    messages = body.get("messages", [])

    with model_lock:  # serialize GPU access
        try:
            content = generate(messages)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    response = {
        "id":      f"chatcmpl-local-{int(time.time())}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   "local",
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    return jsonify(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["finetuned", "base"], default="finetuned")
    parser.add_argument("--port",  type=int, default=8080)
    args = parser.parse_args()

    model_path = str(MODELS[args.model])
    load_model(model_path)

    print(f"Server listening on http://localhost:{args.port}")
    print(f"  Model:    {args.model} ({model_path})")
    print(f"  Endpoint: POST /v1/chat/completions")
    print(f"  Health:   GET  /health")

    # threaded=True lets Flask handle concurrent requests without blocking
    app.run(host="0.0.0.0", port=args.port, threaded=True)

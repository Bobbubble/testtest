#!/usr/bin/env python3
"""
Generate LLM feature embeddings for PersonalizedRouter.

Input: an LLM metadata JSON file whose top-level order defines model order.
Output: a pickle file containing a numpy.ndarray with shape
        (num_llms, embedding_dim).

Each row in the output matrix corresponds to the model at the same position in
the input JSON. PersonalizedRouter aligns rows with list(llm_data.keys()).

Usage:
    python generate_llm_embeddings.py --config config.yaml
    python generate_llm_embeddings.py --input llm_data.json --output llm_embedding_data.pkl
"""

import argparse
import json
import os
import pickle
import sys
from typing import Dict

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LLMROUTER_ROOT = os.path.join(PROJECT_ROOT, "LLMRouter")
if os.path.isdir(LLMROUTER_ROOT) and LLMROUTER_ROOT not in sys.path:
    sys.path.insert(0, LLMROUTER_ROOT)

DEFAULT_EMBEDDING_MODEL = os.path.join(PROJECT_ROOT, "embedding_model")


def _load_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required to generate PersonalizedRouter embeddings") from exc
    return np


def _to_1d_float32_array(embedding):
    np = _load_numpy()
    if hasattr(embedding, "detach"):
        embedding = embedding.detach().cpu().numpy()
    embedding_array = np.asarray(embedding, dtype=np.float32)
    if embedding_array.ndim != 1:
        embedding_array = embedding_array.reshape(-1)
    return embedding_array


def _load_embedding_model(model_name: str, device_arg: str):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError("torch and transformers are required to generate embeddings") from exc

    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model, torch.device(device)


def _mean_pool(last_hidden_state, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)


def _embed_text(text: str, tokenizer, model, device, max_length: int):
    import torch

    inputs = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    return embedding[0].cpu()


def generate_llm_embeddings(
    llm_data: Dict,
    output_path: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "auto",
    max_length: int = 256,
):
    """
    Generate and save an LLM embedding matrix.

    Args:
        llm_data: Ordered dictionary-like JSON object with model metadata.
        output_path: Path to save the PersonalizedRouter .pkl embedding matrix.

    Returns:
        numpy.ndarray with shape (num_llms, embedding_dim).
    """
    print("=== LLM EMBEDDINGS GENERATION ===")

    np = _load_numpy()
    if not isinstance(llm_data, dict) or not llm_data:
        raise ValueError("llm_data must be a non-empty JSON object")
    if not output_path.lower().endswith(".pkl"):
        raise ValueError("Output path must end with .pkl for PersonalizedRouter")

    print(f"Processing {len(llm_data)} LLM candidates...")
    print(f"Embedding model: {model_name}")

    tokenizer, model, embedding_device = _load_embedding_model(model_name, device)

    model_names = []
    embeddings = []

    for idx, (model_key, model_info) in enumerate(llm_data.items()):
        if not isinstance(model_info, dict):
            raise ValueError(f"Metadata for {model_key} must be a JSON object")

        feature_text = model_info.get("feature", "")
        if not isinstance(feature_text, str) or not feature_text.strip():
            raise ValueError(f"No non-empty 'feature' field for {model_key}")

        print(f"[{idx}] Generating embedding for {model_key}...")
        try:
            embedding = _embed_text(feature_text, tokenizer, model, embedding_device, max_length)
        except Exception as exc:
            raise RuntimeError(f"Error generating embedding for {model_key}: {exc}") from exc

        model_names.append(model_key)
        embeddings.append(_to_1d_float32_array(embedding))

    embedding_dim = embeddings[0].shape[0]
    for model_name, embedding_array in zip(model_names, embeddings):
        if embedding_array.shape[0] != embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch for {model_name}: "
                f"{embedding_array.shape[0]} != {embedding_dim}"
            )

    embedding_matrix = np.vstack(embeddings).astype(np.float32)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(embedding_matrix, f)

    print("\nModel row order:")
    for idx, model_name in enumerate(model_names):
        print(f"  [{idx}] {model_name}")
    print(f"\nSaved embedding matrix with shape {embedding_matrix.shape} to {output_path}")

    return embedding_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LLM candidate embeddings")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config containing data_path.llm_data and data_path.llm_embedding_data",
    )
    parser.add_argument("--input", type=str, default=None, help="Path to input LLM data JSON")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output LLM embeddings pickle file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Hugging Face embedding model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Embedding device: auto, cpu, cuda, or cuda:0",
    )
    parser.add_argument("--max_length", type=int, default=256, help="Tokenizer max length")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[str, str]:
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        data_path = config.get("data_path", {}) or {}
        input_path = data_path.get("llm_data", "")
        output_path = data_path.get("llm_embedding_data", "")
        if input_path and not os.path.isabs(input_path):
            input_path = os.path.join(PROJECT_ROOT, input_path)
        if output_path and not os.path.isabs(output_path):
            output_path = os.path.join(PROJECT_ROOT, output_path)
    else:
        if args.input is None or args.output is None:
            raise ValueError("Either --config or both --input and --output must be provided")
        input_path = args.input
        output_path = args.output

    return input_path, output_path


def main() -> None:
    args = parse_args()

    try:
        input_path, output_path = resolve_paths(args)

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        print(f"Loading LLM data from: {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            llm_data = json.load(f)

        embedding_matrix = generate_llm_embeddings(
            llm_data,
            output_path,
            model_name=args.model_name,
            device=args.device,
            max_length=args.max_length,
        )

        print("\nLLM embeddings generation completed successfully!")
        print(f"Generated embeddings for {embedding_matrix.shape[0]} LLM candidates")
        print(f"Embedding dimension: {embedding_matrix.shape[1]}")
        print(f"Output file: {output_path}")
    except KeyboardInterrupt:
        print("\nEmbeddings generation interrupted by user")
        sys.exit(130)
    except Exception as exc:
        print(f"\nError during embeddings generation: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

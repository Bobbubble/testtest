"""
Generate query embeddings using Qwen3-Embedding-0.6B.

Reads query_data_{train,validate,test}.jsonl and generates corresponding .pt files.
Query is a multi-turn conversation list, which is flattened into a single string.
Uses GPU with batched inference.

Usage:
    python generate_query_embeddings.py
    python generate_query_embeddings.py --gpu 0 --batch_size 32 --max_length 1024
"""

import os
import json
import argparse
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def flatten_query(query):
    """Flatten a multi-turn conversation list into a single string."""
    if isinstance(query, str):
        return query
    parts = []
    for msg in query:
        if isinstance(msg, str):
            parts.append(f"User: {msg}")
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "\n".join(str(x) for x in content)
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    return "\n".join(parts)


def load_queries(jsonl_path):
    """Load and flatten query texts from a JSONL file, preserving order."""
    queries = []
    with open(jsonl_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                queries.append(flatten_query(data.get("query", "")))
    return queries


def generate_embeddings(queries, tokenizer, model, device, batch_size=32, max_length=1024):
    """Generate embeddings for a list of queries using batched inference."""
    all_embeddings = []
    total = len(queries)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = queries[start:end]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        all_embeddings.append(embeddings.cpu())

        if (start // batch_size + 1) % 10 == 0 or end == total:
            print(f"  Processed {end}/{total} queries")

    return torch.cat(all_embeddings, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Generate Query Embeddings with Qwen3-Embedding-0.6B")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=1280)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, running on CPU (will be slow)")

    # Load model
    print(f"Loading model: {args.model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, padding_side='left')
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    model = model.to(device).eval()
    embedding_dim = model.config.hidden_size
    print(f"Model loaded. Embedding dim: {embedding_dim}")

    # Process each split (note: validate, not valid)
    splits = [("train", "train"), ("validate", "validate"), ("test", "test")]
    for split_in, split_out in splits:
        jsonl_path = os.path.join(script_dir, f"query_data_{split_in}.jsonl")
        output_path = os.path.join(script_dir, f"query_embeddings_{split_out}.pt")

        if not os.path.exists(jsonl_path):
            print(f"Skipping {split_in}: {jsonl_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {split_in}: {jsonl_path}")

        queries = load_queries(jsonl_path)
        print(f"Loaded {len(queries)} queries")

        embeddings = generate_embeddings(
            queries, tokenizer, model, device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        print(f"Embedding tensor shape: {embeddings.shape}")
        torch.save(embeddings, output_path)
        print(f"Saved to {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()

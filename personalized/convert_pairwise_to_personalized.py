#!/usr/bin/env python3
"""
Convert pairwise preference JSONL files into PersonalizedRouter routing CSVs.

For each input pairwise row, this script writes two routing rows:
- selected answer: effect = 1
- tied answer: effect = 0.5
- unselected answer: effect = 0

It also embeds flattened queries and task descriptions with the local embedding
model, then stores those vectors directly in CSV columns as JSON arrays.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSONALIZED_ROOT = PROJECT_ROOT / "personalized"
DEFAULT_EMBEDDING_MODEL = PROJECT_ROOT / "embedding_model"

TASK_DESCRIPTIONS = {
    "mt_bench": (
        "MT-Bench is a multi-turn instruction-following benchmark that evaluates "
        "assistant responses on open-ended conversational tasks requiring reasoning, "
        "writing, role-play, and problem solving."
    ),
    "chatbot_arena": (
        "Chatbot Arena contains real user prompts for open-ended assistant dialogue. "
        "The task emphasizes helpfulness, relevance, safety, and user preference in "
        "natural conversational settings."
    ),
}

CSV_FIELDS = [
    "comparison_id",
    "user_id",
    "persona_id",
    "performance_preference",
    "task_id",
    "task_name",
    "query",
    "query_embedding",
    "effect",
    "cost",
    "ground_truth",
    "metric",
    "llm",
    "task_description",
    "task_description_embedding",
    "response",
    "reward",
    "best_llm",
    "input_price",
    "output_price",
]


def flatten_query(query: Any) -> str:
    if isinstance(query, str):
        return query
    if isinstance(query, list):
        parts: list[str] = []
        for msg in query:
            if isinstance(msg, str):
                parts.append(f"User: {msg}")
                continue
            if not isinstance(msg, dict):
                parts.append(str(msg))
                continue
            role = str(msg.get("role", "") or "").strip()
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(str(item) for item in content)
            content = str(content)
            if role:
                parts.append(f"{role.capitalize()}: {content}")
            else:
                parts.append(content)
        return "\n".join(part for part in parts if part)
    return str(query)


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            yield line_no, obj


def load_records(paths: list[Path]) -> dict[Path, list[tuple[int, dict[str, Any]]]]:
    records_by_path: dict[Path, list[tuple[int, dict[str, Any]]]] = {}
    for path in paths:
        records_by_path[path] = list(iter_jsonl(path))
    return records_by_path


def build_user_mapping(records_by_path: dict[Path, list[tuple[int, dict[str, Any]]]]) -> dict[Any, int]:
    persona_ids = {
        row.get("persona_id")
        for records in records_by_path.values()
        for _, row in records
        if row.get("persona_id") is not None
    }

    def sort_key(persona_id: Any) -> tuple[int, Any]:
        try:
            return (0, int(persona_id))
        except (TypeError, ValueError):
            return (1, str(persona_id))

    return {persona_id: idx for idx, persona_id in enumerate(sorted(persona_ids, key=sort_key))}


def load_llm_data(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        llm_data = json.load(f)
    if not isinstance(llm_data, dict) or not llm_data:
        raise ValueError(f"llm_data must be a non-empty JSON object: {path}")
    return llm_data


def load_embedding_model(model_path: Path | str, device_arg: str):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError("torch and transformers are required for embedding generation") from exc

    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModel.from_pretrained(str(model_path)).to(device)
    model.eval()
    return tokenizer, model, torch.device(device)


def mean_pool(last_hidden_state, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)


def embed_texts(
    texts: list[str],
    tokenizer,
    model,
    device,
    batch_size: int,
    max_length: int,
) -> dict[str, list[float]]:
    import torch

    embeddings: dict[str, list[float]] = {}
    total = len(texts)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        for text, embedding in zip(batch_texts, pooled.cpu().numpy()):
            embeddings[text] = [float(x) for x in embedding]

        print(f"Embedded {end}/{total} texts")

    return embeddings


def effect_for_position(judge: Any, pair_position: int) -> float:
    if judge == 0:
        return 0.5
    if judge == pair_position:
        return 1.0
    return 0.0


def best_llm_for_row(row: dict[str, Any]) -> str:
    judge = row.get("judge")
    if judge == 1:
        return str(row.get("model_1", ""))
    if judge == 2:
        return str(row.get("model_2", ""))
    return ""


def task_description_for(task_name: Any) -> str:
    task_key = str(task_name or "").strip()
    return TASK_DESCRIPTIONS.get(
        task_key,
        (
            f"{task_key or 'This task'} is an open-ended language task for evaluating "
            "assistant response quality, instruction following, and user preference."
        ),
    )


def split_name_from_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("query_data_"):
        return stem[len("query_data_") :]
    return stem


def make_output_row(
    *,
    row: dict[str, Any],
    comparison_id: str,
    pair_position: int,
    user_mapping: dict[Any, int],
    llm_data: dict[str, dict[str, Any]],
    query_embeddings: dict[str, list[float]],
    task_embeddings: dict[str, list[float]],
) -> dict[str, Any]:
    model_key = row.get(f"model_{pair_position}")
    model_key = str(model_key or "")
    model_info = llm_data.get(model_key, {})
    input_price = model_info.get("input_price", 0.0) or 0.0
    output_price = model_info.get("output_price", 0.0) or 0.0

    query_text = flatten_query(row.get("query", ""))
    task_description = task_description_for(row.get("task_name"))
    effect = effect_for_position(row.get("judge"), pair_position)

    return {
        "comparison_id": comparison_id,
        "user_id": user_mapping[row.get("persona_id")],
        "persona_id": row.get("persona_id", ""),
        "performance_preference": 1.0,
        "task_id": row.get("task_id", ""),
        "task_name": row.get("task_name", ""),
        "query": query_text,
        "query_embedding": json.dumps(query_embeddings[query_text], separators=(",", ":")),
        "effect": effect,
        "cost": float(input_price) + float(output_price),
        "ground_truth": row.get("ground_truth", ""),
        "metric": row.get("metric", ""),
        "llm": model_key,
        "task_description": task_description,
        "task_description_embedding": json.dumps(
            task_embeddings[task_description], separators=(",", ":")
        ),
        "response": row.get(f"answer_{pair_position}", ""),
        "reward": effect,
        "best_llm": best_llm_for_row(row),
        "input_price": input_price,
        "output_price": output_price,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert pairwise JSONL preference data to PersonalizedRouter CSV data"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[
            PERSONALIZED_ROOT / "query_data_train.jsonl",
            PERSONALIZED_ROOT / "query_data_validate.jsonl",
            PERSONALIZED_ROOT / "query_data_test.jsonl",
        ],
        help="Input query_data_*.jsonl files",
    )
    parser.add_argument(
        "--llm_data",
        type=Path,
        default=PERSONALIZED_ROOT / "llm_data.json",
        help="LLM metadata JSON with prices",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PERSONALIZED_ROOT / "routing_data",
        help="Directory for routing_data_*.csv outputs",
    )
    parser.add_argument(
        "--embedding_model",
        type=Path,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Local embedding model path or Hugging Face model name",
    )
    parser.add_argument("--device", default="auto", help="Embedding device: auto, cpu, cuda, cuda:0")
    parser.add_argument("--batch_size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--max_length", type=int, default=256, help="Tokenizer max length")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [path.resolve() for path in args.inputs]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    llm_data = load_llm_data(args.llm_data.resolve())
    records_by_path = load_records(input_paths)
    user_mapping = build_user_mapping(records_by_path)

    query_texts = sorted(
        {
            flatten_query(row.get("query", ""))
            for records in records_by_path.values()
            for _, row in records
        }
    )
    task_descriptions = sorted(
        {
            task_description_for(row.get("task_name"))
            for records in records_by_path.values()
            for _, row in records
        }
    )
    texts_to_embed = query_texts + [t for t in task_descriptions if t not in set(query_texts)]

    print(f"Loaded {sum(len(v) for v in records_by_path.values())} pairwise rows")
    print(f"User mapping size: {len(user_mapping)}")
    print(f"Unique queries: {len(query_texts)}")
    print(f"Task descriptions: {len(task_descriptions)}")
    print(f"Embedding model: {args.embedding_model}")

    tokenizer, model, device = load_embedding_model(args.embedding_model, args.device)
    all_embeddings = embed_texts(
        texts_to_embed,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    query_embeddings = {text: all_embeddings[text] for text in query_texts}
    task_embeddings = {text: all_embeddings[text] for text in task_descriptions}

    for input_path, records in records_by_path.items():
        output_rows: list[dict[str, Any]] = []
        split_name = split_name_from_path(input_path)
        for source_row, row in records:
            comparison_id = f"{split_name}:{source_row:06d}"
            output_rows.append(
                make_output_row(
                    row=row,
                    comparison_id=comparison_id,
                    pair_position=1,
                    user_mapping=user_mapping,
                    llm_data=llm_data,
                    query_embeddings=query_embeddings,
                    task_embeddings=task_embeddings,
                )
            )
            output_rows.append(
                make_output_row(
                    row=row,
                    comparison_id=comparison_id,
                    pair_position=2,
                    user_mapping=user_mapping,
                    llm_data=llm_data,
                    query_embeddings=query_embeddings,
                    task_embeddings=task_embeddings,
                )
            )

        output_path = args.output_dir / f"routing_data_{split_name}.csv"
        write_csv(output_path, output_rows)
        print(f"Wrote {len(output_rows)} rows to {output_path}")

    write_json(
        args.output_dir / "user_mapping.json",
        [{"persona_id": persona_id, "user_id": user_id} for persona_id, user_id in user_mapping.items()],
    )
    write_json(args.output_dir / "task_descriptions.json", TASK_DESCRIPTIONS)
    print(f"Wrote mappings to {args.output_dir}")


if __name__ == "__main__":
    main()

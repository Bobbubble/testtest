import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def load_json(input_path: Path) -> Any:
    if input_path.suffix.lower() == ".jsonl":
        with input_path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def convert_mt_bench(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metadata_records: List[Dict[str, Any]] = []

    for item in records:
        question_id = item.get("question_id", "")
        metadata_records.append(
            {
                "task_name": "mt_bench",
                "query": item.get("turns", []),
                "ground_truth": "",
                "metric": "llm_judge",
                "choices": "",
                "task_id": f"mt_bench_{question_id}",
            }
        )

    return metadata_records


def convert_chatbot_arena(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metadata_records: List[Dict[str, Any]] = []

    for item in records:
        question_id = item.get("question_id", "")
        metadata_records.append(
            {
                "task_name": "chatbot_arena",
                "query": item.get("content", []),
                "ground_truth": "",
                "metric": "llm_judge",
                "choices": "",
                "task_id": f"chatbot_arena_{question_id}",
            }
        )

    return metadata_records


def _compute_split_sizes(total: int, ratios: Tuple[int, int, int]) -> Tuple[int, int, int]:
    ratio_sum = sum(ratios)
    base_sizes = [(total * r) // ratio_sum for r in ratios]
    remainder = total - sum(base_sizes)

    for i in range(remainder):
        base_sizes[i] += 1

    return tuple(base_sizes)


def split_records(
    records: List[Dict[str, Any]],
    ratios: Tuple[int, int, int] = (8, 1, 1),
    seed: int = 42,
    shuffle: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    items = list(records)

    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(items)

    train_size, validate_size, test_size = _compute_split_sizes(len(items), ratios)

    train_records = items[:train_size]
    validate_records = items[train_size : train_size + validate_size]
    test_records = items[train_size + validate_size : train_size + validate_size + test_size]

    return train_records, validate_records, test_records


def write_jsonl(output_path: Path, records: Sequence[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_splits(
    output_root: Path,
    train_records: Sequence[Dict[str, Any]],
    validate_records: Sequence[Dict[str, Any]],
    test_records: Sequence[Dict[str, Any]],
) -> None:
    write_jsonl(output_root / "train" / "metadata.jsonl", train_records)
    write_jsonl(output_root / "validate" / "metadata.jsonl", validate_records)
    write_jsonl(output_root / "test" / "metadata.jsonl", test_records)


def write_query_data_files(
    output_root: Path,
    train_records: Sequence[Dict[str, Any]],
    validate_records: Sequence[Dict[str, Any]],
    test_records: Sequence[Dict[str, Any]],
) -> None:
    write_jsonl(output_root / "query_data_train.jsonl", train_records)
    write_jsonl(output_root / "query_data_validate.jsonl", validate_records)
    write_jsonl(output_root / "query_data_test.jsonl", test_records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert mt_bench and chatbot_arena to metadata, split each dataset, "
            "then merge splits into root train/validate/test outputs."
        )
    )
    parser.add_argument(
        "--mt_input",
        type=Path,
        default=Path("mt_bench") / "origin" / "question.jsonl",
        help="Input file for mt_bench.",
    )
    parser.add_argument(
        "--arena_input",
        type=Path,
        default=Path("chatbot_arena") / "question_content.jsonl",
        help="Input file for chatbot_arena.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Root output directory for merged train/validate/test splits.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling before splitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_root = Path(__file__).resolve().parent
    output_root = args.output_root if args.output_root is not None else script_root

    mt_raw = load_json(args.mt_input)
    arena_raw = load_json(args.arena_input)

    mt_metadata = convert_mt_bench(mt_raw)
    arena_metadata = convert_chatbot_arena(arena_raw)

    mt_train, mt_validate, mt_test = split_records(
        mt_metadata, ratios=(8, 1, 1), seed=args.seed, shuffle=True
    )
    arena_train, arena_validate, arena_test = split_records(
        arena_metadata, ratios=(8, 1, 1), seed=args.seed, shuffle=True
    )

    train_records = mt_train + arena_train
    validate_records = mt_validate + arena_validate
    test_records = mt_test + arena_test

    save_splits(output_root, train_records, validate_records, test_records)
    write_query_data_files(output_root, train_records, validate_records, test_records)

    print(f"mt_bench input: {args.mt_input}")
    print(f"chatbot_arena input: {args.arena_input}")
    print(
        "mt_bench splits: "
        f"total={len(mt_metadata)}, train={len(mt_train)}, "
        f"validate={len(mt_validate)}, test={len(mt_test)}"
    )
    print(
        "chatbot_arena splits: "
        f"total={len(arena_metadata)}, train={len(arena_train)}, "
        f"validate={len(arena_validate)}, test={len(arena_test)}"
    )
    print(
        "merged splits: "
        f"train={len(train_records)}, validate={len(validate_records)}, test={len(test_records)}"
    )

    expected_total = len(mt_metadata) + len(arena_metadata)
    merged_total = len(train_records) + len(validate_records) + len(test_records)
    if merged_total != expected_total:
        raise RuntimeError(
            f"Merged count mismatch: merged_total={merged_total}, expected_total={expected_total}"
        )
    print(f"Merged total: {merged_total}")
    print(f"Output root: {output_root.resolve()}")


if __name__ == "__main__":
    main()

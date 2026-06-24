import json
import os
import random
import re
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterator

from openai import OpenAI
from tqdm import tqdm
import yaml


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML root must be a mapping object: {path}")
    return cfg


def as_str_list(value: Any, key: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return value
    raise ValueError(f"Config key '{key}' must be a list of strings")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{i}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Each line must be a JSON object at {path}:{i}")
            yield obj


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def flush_jsonl_buffer(out_f, buffer: list[str]) -> None:
    if not buffer:
        return
    out_f.write("\n".join(buffer) + "\n")
    out_f.flush()
    buffer.clear()


def resolve_config_path(base_dir: Path, raw: str) -> Path:
    p = Path(raw.strip())
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def resolve_query_files(query_data_paths: list[str] | None) -> list[Path]:
    paths: list[Path] = []

    if query_data_paths:
        paths.extend(Path(p) for p in query_data_paths)

    if not paths:
        raise ValueError("Provide 'query_data_paths' in config")

    deduped: list[Path] = []
    seen: set[Path] = set()
    for p in paths:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        deduped.append(rp)

    missing = [p for p in deduped if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing query_data files: " + ", ".join(str(p) for p in missing))

    return deduped


def extract_models_from_config(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("Config key 'llm_pool' must be a non-empty list of model names")
    models: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            models.append(item.strip())
    if not models:
        raise ValueError("Config key 'llm_pool' must contain at least one non-empty model name")
    return models


def extract_personas(persona_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    personas: list[dict[str, Any]] = []
    for r in persona_records:
        persona_id = r.get("index")
        persona = r.get("persona")
        if not isinstance(persona_id, int):
            raise ValueError("Each persona record must contain integer key: index")
        if not isinstance(persona, str) or not persona.strip():
            raise ValueError("Each persona record must contain non-empty string key: persona")
        personas.append({"index": persona_id, "persona": persona.strip()})
    if not personas:
        raise ValueError("No valid persona found in personas.jsonl")
    return personas


def normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(x) for x in content)
    return str(content)


def build_query_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    query = record.get("query")
    if isinstance(query, list):
        messages: list[dict[str, str]] = []
        for msg in query:
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content")
                if isinstance(role, str) and content is not None:
                    messages.append({"role": role, "content": normalize_content(content)})
                continue
            if isinstance(msg, str) and msg.strip():
                messages.append({"role": "user", "content": msg.strip()})
        if messages:
            return messages

    if isinstance(query, str) and query.strip():
        return [{"role": "user", "content": query.strip()}]

    content = record.get("content")
    if isinstance(content, list) and content:
        return [{"role": "user", "content": normalize_content(content[0])}]

    raise ValueError("Unable to build query messages from record")


def call_chat_completion(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    max_retries: int,
    temperature: float,
    max_tokens: int,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content
            return content or ""
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt == max_retries:
                break
            time.sleep(min(2 ** (attempt - 1), 8))

    raise RuntimeError(f"Chat completion failed for model={model}: {last_error}")


def parse_judge(raw: str) -> int:
    text = raw.strip()
    lowered = text.lower()

    if "tie" in lowered:
        return 0

    one_match = re.search(r"\b1\b", lowered)
    two_match = re.search(r"\b2\b", lowered)

    if one_match and not two_match:
        return 1
    if two_match and not one_match:
        return 2

    compact = re.sub(r"[^a-z0-9]", "", lowered)
    if compact == "1":
        return 1
    if compact == "2":
        return 2
    if compact == "tie":
        return 0

    raise ValueError(f"Unexpected judge output: {raw!r}")


def build_judge_prompt(persona: str, question: str, answer_1: str, answer_2: str) -> str:
    return (
        f"You are simulating the following user persona:\n"
        f"{persona}.\n\n"
        
        f"As this persona, evaluate which assistant response you would personally prefer.\n"

        f"[User Question]\n{question}\n\n"
        f"[Assistant 1]\n{answer_1}\n\n"
        f"[Assistant 2]\n{answer_2}\n\n"

        "Output ONLY one of the following tokens exactly:\n"
        "1\n"
        "2\n"
        "Tie\n"
        "Do not output anything else."
    )


def generate_one_row(
    client: OpenAI,
    row: dict[str, Any],
    model_1: str,
    model_2: str,
    max_retries: int,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    messages = build_query_messages(row)
    answer_1 = call_chat_completion(
        client=client,
        model=model_1,
        messages=messages,
        max_retries=max_retries,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    answer_2 = call_chat_completion(
        client=client,
        model=model_2,
        messages=messages,
        max_retries=max_retries,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    row_out = dict(row)
    row_out["model_1"] = model_1
    row_out["answer_1"] = answer_1
    row_out["model_2"] = model_2
    row_out["answer_2"] = answer_2
    return row_out


def judge_one_row(
    client: OpenAI,
    row: dict[str, Any],
    judge_model: str,
    persona_id: int,
    persona: str,
    max_retries: int,
    max_tokens: int,
) -> dict[str, Any]:
    messages = build_query_messages(row)
    question = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            question = msg.get("content", "")
            break
    if not question and messages:
        question = messages[-1].get("content", "")

    answer_1 = normalize_content(row.get("answer_1", ""))
    answer_2 = normalize_content(row.get("answer_2", ""))
    judge_prompt = build_judge_prompt(persona, question, answer_1, answer_2)
    judge_raw = call_chat_completion(
        client=client,
        model=judge_model,
        messages=[{"role": "user", "content": judge_prompt}],
        max_retries=max_retries,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    judge = parse_judge(judge_raw)

    row_out = dict(row)
    row_out["judge"] = judge
    row_out["persona_id"] = persona_id
    return row_out


def generate_answers(
    client: OpenAI,
    query_files: list[Path],
    models: list[str],
    answers_output_path: Path,
    max_retries: int,
    temperature: float,
    max_tokens: int,
    save_every: int,
    workers: int,
    max_in_flight: int,
) -> None:
    total_queries = sum(count_jsonl_rows(p) for p in query_files)
    pbar = tqdm(total=total_queries, desc="Generate", unit="query")

    written = 0
    failed = 0
    buffer: list[str] = []
    sequence = 0
    next_to_write = 0
    completed_lines: dict[int, str | None] = {}
    task_ids: dict[int, str] = {}
    pending: dict[Future[dict[str, Any]], int] = {}

    def flush_ready_lines(out_f) -> None:
        nonlocal written, next_to_write
        while next_to_write in completed_lines:
            line = completed_lines.pop(next_to_write)
            if line is not None:
                buffer.append(line)
                written += 1
                if len(buffer) >= save_every:
                    flush_jsonl_buffer(out_f, buffer)
            task_ids.pop(next_to_write, None)
            next_to_write += 1

    def collect_done(out_f, done_futures: set[Future[dict[str, Any]]]) -> None:
        nonlocal failed
        for fut in done_futures:
            seq = pending.pop(fut)
            task_id = task_ids.get(seq, "unknown")
            try:
                row_out = fut.result()
                completed_lines[seq] = json.dumps(row_out, ensure_ascii=False)
            except Exception as e:  # noqa: BLE001
                failed += 1
                completed_lines[seq] = None
                print(f"[generate] skip task_id={task_id} error={e}")
            pbar.update(1)
        flush_ready_lines(out_f)

    with answers_output_path.open("w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for query_file in query_files:
                for row_index, row in enumerate(iter_jsonl(query_file), 1):
                    task_id = str(row.get("task_id", "")).strip() or f"{query_file.name}:{row_index}"
                    model_1 = random.choice(models)
                    model_2 = random.choice(models)
                    print(f"[generate] task_id={task_id} model_1={model_1} model_2={model_2}")
                    fut = executor.submit(
                        generate_one_row,
                        client,
                        row,
                        model_1,
                        model_2,
                        max_retries,
                        temperature,
                        max_tokens,
                    )
                    pending[fut] = sequence
                    task_ids[sequence] = task_id
                    sequence += 1

                    if len(pending) >= max_in_flight:
                        done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
                        collect_done(out_f, done)

            while pending:
                done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
                collect_done(out_f, done)

        flush_jsonl_buffer(out_f, buffer)

    pbar.close()
    print(f"Generate done. written={written}, failed={failed}, total={total_queries}")


def judge_answers(
    client: OpenAI,
    answers_output_path: Path,
    output_path: Path,
    judge_model: str,
    personas: list[dict[str, Any]],
    max_retries: int,
    max_tokens: int,
    save_every: int,
    workers: int,
    max_in_flight: int,
) -> None:
    total_answers = count_jsonl_rows(answers_output_path)
    pbar = tqdm(total=total_answers, desc="Judge", unit="query")

    written = 0
    failed = 0
    buffer: list[str] = []
    sequence = 0
    next_to_write = 0
    completed_lines: dict[int, str | None] = {}
    task_ids: dict[int, str] = {}
    pending: dict[Future[dict[str, Any]], int] = {}

    def flush_ready_lines(out_f) -> None:
        nonlocal written, next_to_write
        while next_to_write in completed_lines:
            line = completed_lines.pop(next_to_write)
            if line is not None:
                buffer.append(line)
                written += 1
                if len(buffer) >= save_every:
                    flush_jsonl_buffer(out_f, buffer)
            task_ids.pop(next_to_write, None)
            next_to_write += 1

    def collect_done(out_f, done_futures: set[Future[dict[str, Any]]]) -> None:
        nonlocal failed
        for fut in done_futures:
            seq = pending.pop(fut)
            task_id = task_ids.get(seq, "unknown")
            try:
                row_out = fut.result()
                completed_lines[seq] = json.dumps(row_out, ensure_ascii=False)
            except Exception as e:  # noqa: BLE001
                failed += 1
                completed_lines[seq] = None
                print(f"[judge] skip task_id={task_id} error={e}")
            pbar.update(1)
        flush_ready_lines(out_f)

    with output_path.open("w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for row in iter_jsonl(answers_output_path):
                task_id = str(row.get("task_id", "")).strip() or "unknown"
                persona_record = random.choice(personas)
                persona_id = persona_record["index"]
                persona = persona_record["persona"]
                fut = executor.submit(
                    judge_one_row,
                    client,
                    row,
                    judge_model,
                    persona_id,
                    persona,
                    max_retries,
                    max_tokens,
                )
                pending[fut] = sequence
                task_ids[sequence] = task_id
                sequence += 1

                if len(pending) >= max_in_flight:
                    done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
                    collect_done(out_f, done)

            while pending:
                done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
                collect_done(out_f, done)

        flush_jsonl_buffer(out_f, buffer)

    pbar.close()
    print(f"Judge done. written={written}, failed={failed}, total={total_answers}")


def main() -> None:
    config_path = Path(__file__).resolve().parent / "personalized_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = load_yaml_config(config_path)
    config_dir = config_path.parent.resolve()

    query_data_paths = as_str_list(cfg.get("query_data_paths"), "query_data_paths")
    query_data_paths = (
        [str(resolve_config_path(config_dir, p)) for p in query_data_paths]
        if query_data_paths is not None
        else None
    )

    llm_pool_cfg = cfg.get("llm_pool")

    personas_raw = cfg.get("personas", "personas.jsonl")
    if not isinstance(personas_raw, str) or not personas_raw.strip():
        raise ValueError("Config key 'personas' must be a non-empty string")
    personas_path = resolve_config_path(config_dir, personas_raw)

    output_raw = cfg.get("output")
    if not isinstance(output_raw, str) or not output_raw.strip():
        raise ValueError("Config key 'output' is required and must be a non-empty string")
    output_dir = resolve_config_path(config_dir, output_raw)

    judge_model = cfg.get("judge_model")
    if not isinstance(judge_model, str) or not judge_model.strip():
        raise ValueError("Config key 'judge_model' is required and must be a non-empty string")
    judge_model = judge_model.strip()

    base_url = cfg.get("base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("Config key 'base_url' is required and must be a non-empty string")
    base_url = base_url.strip()

    api_key = cfg.get("api_key") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    seed = int(cfg.get("seed", 42))
    max_retries = int(cfg.get("max_retries", 3))
    temperature = float(cfg.get("temperature", 0.0))
    max_tokens = int(cfg.get("max_tokens", 1024))
    save_every = int(cfg.get("save_every", 10))
    generate_workers = int(cfg.get("generate_workers", 8))
    judge_workers = int(cfg.get("judge_workers", 8))
    max_in_flight = int(cfg.get("max_in_flight", max(generate_workers, judge_workers) * 2))

    if save_every < 1:
        raise ValueError("Config key 'save_every' must be >= 1")
    if generate_workers < 1:
        raise ValueError("Config key 'generate_workers' must be >= 1")
    if judge_workers < 1:
        raise ValueError("Config key 'judge_workers' must be >= 1")
    if max_in_flight < 1:
        raise ValueError("Config key 'max_in_flight' must be >= 1")

    if not api_key:
        raise ValueError("Missing API key. Set 'api_key' in config or OPENROUTER_API_KEY/OPENAI_API_KEY")

    random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    query_files = resolve_query_files(query_data_paths)
    models = extract_models_from_config(llm_pool_cfg)
    personas = extract_personas(list(iter_jsonl(personas_path)))

    client = OpenAI(api_key=api_key, base_url=base_url)

    for idx, query_file in enumerate(query_files, 1):
        file_stem = query_file.stem
        answers_output_path = output_dir / f"{file_stem}_answers.jsonl"
        output_path = output_dir / f"{file_stem}.jsonl"

        print(f"[file {idx}/{len(query_files)}] start: {query_file}")
        generate_answers(
            client=client,
            query_files=[query_file],
            models=models,
            answers_output_path=answers_output_path,
            max_retries=max_retries,
            temperature=temperature,
            max_tokens=max_tokens,
            save_every=save_every,
            workers=generate_workers,
            max_in_flight=max(max_in_flight, generate_workers),
        )

        if not answers_output_path.exists():
            raise FileNotFoundError(f"Answers output not found: {answers_output_path}")

        judge_answers(
            client=client,
            answers_output_path=answers_output_path,
            output_path=output_path,
            judge_model=judge_model,
            personas=personas,
            max_retries=max_retries,
            max_tokens=max_tokens,
            save_every=save_every,
            workers=judge_workers,
            max_in_flight=max(max_in_flight, judge_workers),
        )
        print(f"[file {idx}/{len(query_files)}] done. Answers: {answers_output_path} | Final: {output_path}")


if __name__ == "__main__":
    main()


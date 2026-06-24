# Group-Based PersonalizedRouter

This version of `PersonalizedRouter` trains on comparison groups instead of dense
`user x query x llm` matrices.

## Difference From The Dense Version

The old dense version expected every user/query example to contain all candidate
LLMs, then learned an N-way winner from a fixed-size block. That made the data
shape strict: rows had to line up as a product of users, queries, and model
count.

The group-based version treats each comparison as a candidate set:

```text
comparison_id, user_id, query, llm, effect
train:000001,  0,       ...,   model_a, 1.0
train:000001,  0,       ...,   model_b, 0.0
```

Rows with the same `comparison_id` belong to the same comparison. A group can
contain 2 candidates for pairwise data or N candidates for N-way data.

## Supported Data

- Pairwise winner/loser: `effect = 1.0 / 0.0`
- Pairwise tie: `effect = 0.5 / 0.5`
- N-way winner: winner `effect = 1.0`, others `0.0`
- N-way soft preference: any non-negative `effect` values inside the group

Training normalizes `effect` within each `comparison_id` and applies group
softmax loss. Missing LLMs do not need fake rows.

## Required CSV Columns

```text
comparison_id
user_id
query
query_embedding
llm
effect
```

Recommended columns:

```text
persona_id
task_id
task_name
cost
metric
task_description
task_description_embedding
response
reward
best_llm
input_price
output_price
```

## Current Local Files

Converted routing CSVs are under:

```text
personalized/routing_data/
```

The group config is:

```text
personalized/personalizedrouter_config.yaml
```

## Training

```powershell
conda run -n router python -c "import sys; sys.path.insert(0,'LLMRouter'); from llmrouter.models import PersonalizedRouter; from llmrouter.models.personalizedrouter.trainer import PersonalizedRouterTrainer; r=PersonalizedRouter('personalized/personalizedrouter_config.yaml'); t=PersonalizedRouterTrainer(r); print(t.train())"
```

## Inference

By default, inference scores all LLMs in `llm_data.json`.

To route over a custom candidate set, pass `candidates`:

```python
router.route_single({
    "query": "...",
    "user_id": 0,
    "query_embedding": [...],
    "task_description_embedding": [...],
    "candidates": ["model_a", "model_b"]
})
```

`candidates` may contain 2 models for pairwise routing or any N models for
N-way routing.

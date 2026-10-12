# Data formats

## Graph: `graph.json`

```json
{"triples": [["Mira", "studied_at", "University of Oxford"], ["University of Oxford", "located_in", "Oxford"]]}
```

The canonical identity of a graph is the SHA-256 hash of its sorted, deduplicated triple list. Direction of execution (`+` / `-`) is encoded in candidate actions, not in the stored KG triples.

## Questions: JSONL

```json
{"question_id": "mira-1", "question": "In which country is the university that Mira attended?", "topic_entities": ["Mira"], "answers": ["United Kingdom"], "split": "test"}
```

Use canonical entity IDs. Converters in `scripts/prepare_dataset.py` normalize common source keys; the dataset's official evaluation and entity alias rules take precedence where applicable. A question can have more than one linked root; each root constitutes a separate search context.

## Drafter supervision: JSONL

```json
{"question_id": "mira-1", "state_text": "Q: ... | root=Mira | path=...", "actions": ["studied_at University of Oxford", "born_in Paris", "STOP"], "positives": [true, false, false]}
```

Gold answers are used for labeling **training** paths. Test-time graph search and scheduling use only the question, graph, learned scores, and previously acquired candidates.

## Predictions: JSONL

```json
{"question_id":"mira-1", "prediction":["United Kingdom"], "reference":["United Kingdom"], "selected_id":"c1", "input_tokens":1024, "elapsed_ms":320.1, "n_candidates":12, "selector_calls":1}
```

`reference` may be omitted from exported predictions used in shared or blind evaluation settings. Use a separate reference file when scoring such predictions. The provided `evaluate_predictions.py` expects merged canonical IDs and checks all expected question IDs.

## Candidate bank: JSONL

One row per question, with `candidates` containing records of `id`, `context`, `path`, `score` and `answers`. Each path action contains `head`, `relation`, `direction`, and `tail`. The external registry stores complete answers and is not serialized into the frozen model input.

## Final call cost profile

```json
{"upper_token_bounds":[512,1024,2048],"conservative_ms":[80,140,260]}
```

The structure above is illustrative of the file format; generate hardware/model-specific values with `calibrate_reader_cost.py`.

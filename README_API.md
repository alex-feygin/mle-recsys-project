# RecSys FastAPI Microservice

Recommendation service that combines offline ALS-based recommendations with online content-based recommendations.

## Architecture

The service is logically split into two stores, each exposed at its own URL:

| Component | URL | Role |
|-----------|-----|------|
| `EventStore` | `http://127.0.0.1:8020` | In-memory user event history |
| `RecommendationService` | `http://127.0.0.1:8010` | Offline + online + blended recs |
| Main app | `http://127.0.0.1:8000` | Composes both stores, exposes all endpoints |

## Project Structure

```
├── recommendation_service.py   # FastAPI service (RecommendationService + EventStore)
├── test_service.py             # Service tests (3 scenarios)
└── recsys/
    ├── recommendations/
    │   ├── personal_als.parquet  # ALS personalized recs (user_id, item_id, score, rank)
    │   ├── cold_recs.parquet     # Popular items fallback (item_id, score)
    │   └── similar.parquet       # Item-to-item cosine similarity (track_id, similar_track_id, score, rank)
    └── data/
        ├── items.parquet         # Track metadata
        └── events.parquet        # User interaction history
```

## Running the Service

```bash
uvicorn recommendation_service:app --host 0.0.0.0 --port 8000 --reload
```

API docs: http://localhost:8000/docs

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/put_event?user_id=&item_id=` | Store a user interaction event |
| `GET` | `/get_events?user_id=&k=` | Retrieve recent events for a user |
| `GET` | `/recommendations_offline?user_id=&k=` | Offline ALS recs (cold fallback for new users) |
| `GET` | `/recommendations_online?user_id=&k=` | Online content-based recs from recent events |
| `GET` | `/recommendations?user_id=&k=` | Blended offline + online recs |

All recommendation endpoints return `{"recs": [...]}`.

### Online Recommendations Logic

1. Retrieve last `ONLINE_HISTORY_DEPTH=3` events for the user
2. For each event (seed track), fetch similar items from `similar.parquet`
3. Merge all candidates
4. Sort by similarity score (descending)
5. Remove duplicates (first occurrence wins)
6. Return top `k`

### Blending Strategy

1. Fetch offline recs (personal ALS or cold fallback)
2. Fetch online recs (content-based from last 3 events)
3. Alternate items: offline[0], online[0], offline[1], online[1], ...
4. Append remaining tail
5. Remove duplicates (first occurrence wins)
6. Return first `k` items

## Running Tests

```bash
uvicorn recommendation_service:app --host 0.0.0.0 --port 8000
# in another terminal:
python test_service.py
```

### Test Scenarios

- **Case 1** — Cold user (no personal recs, no history): offline -> popular items; online -> empty; blended -> popular items
- **Case 2** — Warm user, no event history: offline -> personal ALS recs; online -> empty; blended -> personal ALS recs
- **Case 3** — Warm user with injected event history: online -> content recs; blended -> interleaved offline + online, no duplicates, length = k

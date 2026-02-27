"""
RecSys FastAPI microservice.

Endpoints:
  POST /put_event               — store a user event
  GET  /recommendations_offline — ALS recs (cold fallback)
  GET  /recommendations_online  — content-based recs from recent events
  GET  /recommendations         — blended offline + online
"""
import logging
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

RECS_DIR = Path(__file__).parent / "recommendations"
MAX_EVENTS_PER_USER = 20
DEFAULT_K = 10

# ---------------------------------------------------------------------------
# In-memory stores (populated at startup)
# ---------------------------------------------------------------------------
personal_als: dict[int, list[int]] = {}   # user_id -> ordered item_ids
cold_recs: list[int] = []                  # ordered item_ids (popularity desc)
content_recs: dict[int, list[tuple[int, float]]] = {}  # item_id -> [(item_id, score)]

# In-memory event store: user_id -> deque of item_ids (most-recent first)
user_events: dict[int, deque] = defaultdict(lambda: deque(maxlen=MAX_EVENTS_PER_USER))

# Counters for logging
_offline_requests = 0
_cold_fallbacks = 0
_online_requests = 0


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield


def _load_artifacts() -> None:
    """Load all parquet artifacts once at startup."""
    logger.info("Loading recommendation artifacts …")

    # --- personal ALS recs ---
    df_als = pd.read_parquet(RECS_DIR / "personal_als.parquet")
    # columns: user_id, item_id, score, rank  (already sorted by rank asc)
    for user_id, grp in df_als.sort_values("rank").groupby("user_id"):
        personal_als[int(user_id)] = grp["item_id"].astype(int).tolist()
    logger.info("Loaded personal_als: %d users", len(personal_als))

    # --- cold recs ---
    df_cold = pd.read_parquet(RECS_DIR / "cold_recs.parquet")
    # columns: item_id, score  (sort descending by score)
    cold_recs.extend(
        df_cold.sort_values("score", ascending=False)["item_id"].astype(int).tolist()
    )
    logger.info("Loaded cold_recs: %d items", len(cold_recs))

    # --- content recs (item-to-item similarity) ---
    # content_recs.parquet: user_id, item_id, score, rank
    # We interpret user_id as the *seed* track used to build recs for that user.
    # For online serving we need item-level lookup:
    #   seed_item -> [(similar_item, score), ...]
    # The notebook stored per-user (user -> seed track -> similar tracks).
    # We rebuild the item-level index from the mapping.
    df_content = pd.read_parquet(RECS_DIR / "content_recs.parquet")
    # columns: user_id (encoded), item_id, score, rank
    # To build item-level index we need the seed track per user.
    # The notebook used the user's latest track as the seed.
    # We load events to recover that mapping.
    events_path = Path(__file__).parent / "data" / "events.parquet"
    df_events = pd.read_parquet(events_path, columns=["user_id", "track_id", "track_seq"])

    seed_per_user = (
        df_events.sort_values(["user_id", "track_seq"], ascending=[True, False])
        .drop_duplicates("user_id")[["user_id", "track_id"]]
    )
    seed_map = dict(zip(seed_per_user["user_id"].astype(int), seed_per_user["track_id"].astype(int)))

    # Merge seed track onto content_recs
    df_content["seed_item"] = df_content["user_id"].astype(int).map(seed_map)
    df_content = df_content.dropna(subset=["seed_item"])
    df_content["seed_item"] = df_content["seed_item"].astype(int)

    for seed_item, grp in df_content.sort_values("rank").groupby("seed_item"):
        content_recs[int(seed_item)] = list(
            zip(grp["item_id"].astype(int).tolist(), grp["score"].tolist())
        )
    logger.info("Loaded content_recs: %d seed items", len(content_recs))


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="RecSys Service", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_offline(user_id: int, k: int) -> list[int]:
    global _offline_requests, _cold_fallbacks
    _offline_requests += 1
    if user_id in personal_als:
        return personal_als[user_id][:k]
    _cold_fallbacks += 1
    logger.debug("Cold fallback for user %d", user_id)
    return cold_recs[:k]


def _get_online(user_id: int, k: int) -> list[int]:
    global _online_requests
    _online_requests += 1
    history = list(user_events[user_id])  # most-recent first
    if not history:
        return []

    seeds = history[:3]
    candidates: list[tuple[int, float]] = []
    for seed in seeds:
        candidates.extend(content_recs.get(seed, []))

    if not candidates:
        return []

    # Sort by similarity score descending
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Remove duplicates (keep first occurrence)
    seen: set[int] = set()
    result: list[int] = []
    for item_id, _ in candidates:
        if item_id not in seen:
            seen.add(item_id)
            result.append(item_id)
        if len(result) >= k:
            break
    return result


def _blend(offline: list[int], online: list[int], k: int) -> list[int]:
    """Alternate items from offline and online, append tail, deduplicate."""
    blended: list[int] = []
    seen: set[int] = set()
    max_len = max(len(offline), len(online))
    for i in range(max_len):
        for lst in (offline, online):
            if i < len(lst):
                item = lst[i]
                if item not in seen:
                    seen.add(item)
                    blended.append(item)
    return blended[:k]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/put_event", summary="Store a user interaction event")
def put_event(user_id: int, item_id: int):
    """Add item_id to the front of the user's recent-event history."""
    dq = user_events[user_id]
    dq.appendleft(item_id)
    return {"user_id": user_id, "item_id": item_id, "stored": True}


@app.get("/recommendations_offline", summary="Offline ALS recommendations (cold fallback)")
def recommendations_offline(
    user_id: int,
    k: int = Query(DEFAULT_K, ge=1, le=100),
):
    recs = _get_offline(user_id, k)
    logger.info(
        "offline | user=%d k=%d returned=%d  [offline_total=%d cold_total=%d]",
        user_id, k, len(recs), _offline_requests, _cold_fallbacks,
    )
    return {"user_id": user_id, "recommendations": recs}


@app.get("/recommendations_online", summary="Online content-based recommendations")
def recommendations_online(
    user_id: int,
    k: int = Query(DEFAULT_K, ge=1, le=100),
):
    recs = _get_online(user_id, k)
    logger.info(
        "online  | user=%d k=%d returned=%d  [online_total=%d]",
        user_id, k, len(recs), _online_requests,
    )
    return {"user_id": user_id, "recommendations": recs}


@app.get("/recommendations", summary="Blended offline + online recommendations")
def recommendations(
    user_id: int,
    k: int = Query(DEFAULT_K, ge=1, le=100),
):
    offline = _get_offline(user_id, k)
    online = _get_online(user_id, k)
    blended = _blend(offline, online, k)
    return {"user_id": user_id, "recommendations": blended}

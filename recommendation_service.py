"""
RecSys FastAPI microservice.

Endpoints:
  POST /put_event               - store a user event
  GET  /recommendations_offline - ALS recs (cold fallback)
  GET  /recommendations_online  - content-based recs from recent events
  GET  /recommendations         - blended offline + online
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

RECS_DIR = Path(__file__).parent / "recsys" / "recommendations"
DATA_DIR = Path(__file__).parent / "recsys" / "data"
DEFAULT_K = 10


# ---------------------------------------------------------------------------
# Recommendation Storage Layer
# ---------------------------------------------------------------------------
class RecommendationStore:

    def __init__(self):
        self.personal = None   # dict: user_id -> list[item_id]
        self.cold = None       # list[item_id]  (popularity desc)
        self.content = None    # dict: seed_item_id -> list[(item_id, score)]

    def load(self):
        """
        Load all parquet artifacts into memory.
        - personal_als.parquet  -> personalized offline recommendations
        - cold_recs.parquet     -> default (most popular) recommendations
        - content_recs.parquet  -> cosine_similarity-based item2item recs
        """
        logger.info("Loading recommendation artifacts ...")

        # --- personal ALS recs: user_id -> ordered item_ids ---
        df_als = pd.read_parquet(RECS_DIR / "personal_als.parquet")
        self.personal = {
            int(uid): grp.sort_values("rank")["item_id"].astype(int).tolist()
            for uid, grp in df_als.groupby("user_id")
        }
        logger.info("Loaded personal_als: %d users", len(self.personal))

        # --- cold recs: ordered item_ids by popularity ---
        df_cold = pd.read_parquet(RECS_DIR / "cold_recs.parquet")
        self.cold = (
            df_cold.sort_values("score", ascending=False)["item_id"]
            .astype(int)
            .tolist()
        )
        logger.info("Loaded cold_recs: %d items", len(self.cold))

        # --- content recs: seed_item -> [(similar_item, score), ...] ---
        # content_recs.parquet stores per-user recs keyed by user_id (encoded).
        # The notebook used each user's most-recent track as the seed.
        # We recover the seed-track-per-user mapping from events.parquet and
        # build an item-level lookup index.
        df_content = pd.read_parquet(RECS_DIR / "content_recs.parquet")
        df_events = pd.read_parquet(
            DATA_DIR / "events.parquet",
            columns=["user_id", "track_id", "track_seq"],
        )
        seed_map = (
            df_events
            .sort_values(["user_id", "track_seq"], ascending=[True, False])
            .drop_duplicates("user_id")
            .set_index("user_id")["track_id"]
            .astype(int)
            .to_dict()
        )
        df_content["seed_item"] = df_content["user_id"].astype(int).map(seed_map)
        df_content = df_content.dropna(subset=["seed_item"])
        df_content["seed_item"] = df_content["seed_item"].astype(int)

        self.content = {
            int(seed): list(
                zip(grp.sort_values("rank")["item_id"].astype(int).tolist(),
                    grp.sort_values("rank")["score"].tolist())
            )
            for seed, grp in df_content.groupby("seed_item")
        }
        logger.info("Loaded content_recs: %d seed items", len(self.content))

    def get_offline(self, user_id: int, k: int) -> list:
        """
        Return personal recommendations if available.
        Otherwise return cold recommendations.
        """
        if user_id in self.personal:
            return self.personal[user_id][:k]
        logger.debug("Cold fallback for user %d", user_id)
        return self.cold[:k]

    def get_online(self, user_id: int, user_events: list, k: int) -> list:
        """
        Generate online recommendations using content_recs.
        - Use last 3 events
        - Collect similar items
        - Sort by similarity score (descending)
        - Deduplicate (keep first occurrence)
        - Return top k
        """
        seeds = user_events[:3]
        if not seeds:
            return []

        candidates: list[tuple[int, float]] = []
        for seed in seeds:
            candidates.extend(self.content.get(seed, []))

        if not candidates:
            return []

        candidates.sort(key=lambda x: x[1], reverse=True)

        seen: set[int] = set()
        result: list[int] = []
        for item_id, _ in candidates:
            if item_id not in seen:
                seen.add(item_id)
                result.append(item_id)
            if len(result) >= k:
                break
        return result

    def blend(self, recs_offline: list, recs_online: list, k: int) -> list:
        """
        Blend offline and online lists:
        - Alternate items (offline[0], online[0], offline[1], online[1], ...)
        - Append remaining tail
        - Deduplicate (keep first occurrence)
        - Return first k
        """
        blended: list[int] = []
        seen: set[int] = set()
        max_len = max(len(recs_offline), len(recs_online))
        for i in range(max_len):
            for lst in (recs_offline, recs_online):
                if i < len(lst):
                    item = lst[i]
                    if item not in seen:
                        seen.add(item)
                        blended.append(item)
        return blended[:k]


# ---------------------------------------------------------------------------
# Event Store
# ---------------------------------------------------------------------------
class EventStore:

    def __init__(self, max_events_per_user: int = 10):
        self.events: dict[int, list[int]] = {}
        self.max_events_per_user = max_events_per_user

    def put(self, user_id: int, item_id: int) -> None:
        """Store latest event (most recent first). Keep only max_events_per_user events."""
        if user_id not in self.events:
            self.events[user_id] = []
        self.events[user_id].insert(0, item_id)
        self.events[user_id] = self.events[user_id][: self.max_events_per_user]

    def get(self, user_id: int, k: int) -> list[int]:
        """Return last k events for user."""
        return self.events.get(user_id, [])[:k]


# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------
store = RecommendationStore()
event_store = EventStore(max_events_per_user=10)

# Counters for logging
_offline_requests = 0
_cold_fallbacks = 0
_online_requests = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.load()
    yield


app = FastAPI(title="RecSys Service", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/put_event", summary="Store a user interaction event")
def put_event(user_id: int, item_id: int):
    event_store.put(user_id, item_id)
    return {"user_id": user_id, "item_id": item_id, "stored": True}


@app.get("/recommendations_offline", summary="Offline ALS recommendations (cold fallback)")
def recommendations_offline(
    user_id: int,
    k: int = Query(DEFAULT_K, ge=1, le=100),
):
    global _offline_requests, _cold_fallbacks
    _offline_requests += 1
    is_cold = user_id not in store.personal
    if is_cold:
        _cold_fallbacks += 1
    recs = store.get_offline(user_id, k)
    logger.info(
        "offline | user=%d k=%d returned=%d cold=%s [total=%d cold_total=%d]",
        user_id, k, len(recs), is_cold, _offline_requests, _cold_fallbacks,
    )
    return {"recs": recs}


@app.get("/recommendations_online", summary="Online content-based recommendations")
def recommendations_online(
    user_id: int,
    k: int = Query(DEFAULT_K, ge=1, le=100),
):
    global _online_requests
    _online_requests += 1
    events = event_store.get(user_id, 3)
    recs = store.get_online(user_id, events, k)
    logger.info(
        "online  | user=%d k=%d returned=%d [total=%d]",
        user_id, k, len(recs), _online_requests,
    )
    return {"recs": recs}


@app.get("/recommendations", summary="Blended offline + online recommendations")
def recommendations(
    user_id: int,
    k: int = Query(DEFAULT_K, ge=1, le=100),
):
    events = event_store.get(user_id, 3)
    recs_offline = store.get_offline(user_id, k)
    recs_online = store.get_online(user_id, events, k)
    recs = store.blend(recs_offline, recs_online, k)
    return {"recs": recs}

"""
RecSys FastAPI microservice.

Endpoints:
  POST /put_event               - store a user event
  GET  /get_events              - retrieve recent events for a user
  GET  /recommendations_offline - ALS recs (cold fallback)
  GET  /recommendations_online  - content-based recs from recent events
  GET  /recommendations         - blended offline + online

Architecture:
  EventStore        runs at events_store_url        (default: http://127.0.0.1:8020)
  RecommendationService runs at recommendation_store_url (default: http://127.0.0.1:8010)
  This app composes both at the top-level URL (default: http://127.0.0.1:8000)
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Query

logger = logging.getLogger("uvicorn.error")

RECS_DIR = Path(__file__).parent / "recsys" / "recommendations"
DATA_DIR = Path(__file__).parent / "recsys" / "data"

DEFAULT_K = 10
MAX_EVENTS_PER_USER = 10
ONLINE_HISTORY_DEPTH = 3   # how many recent events to use for online recs

events_store_url = "http://127.0.0.1:8020"
recommendation_store_url = "http://127.0.0.1:8010"


# ---------------------------------------------------------------------------
# RecommendationService
# ---------------------------------------------------------------------------
class RecommendationService:

    def __init__(self, url: str = recommendation_store_url):
        self.url = url
        self.personal = None   # dict: user_id -> list[item_id]
        self.cold = None       # list[item_id]  (popularity desc)
        self.content = None    # dict: track_id -> list[(similar_track_id, score)]
        self._stats = {
            "offline_requests": 0,
            "cold_fallbacks":   0,
            "online_requests":  0,
        }

    def stats(self) -> dict:
        logger.info("Stats for recommendations")
        for name, value in self._stats.items():
            logger.info(f"{name:<30} {value} ")
        return self._stats

    def load(self):
        """
        Load all parquet artifacts into memory.
        - personal_als.parquet -> personalized offline recommendations
        - cold_recs.parquet    -> default (most popular) recommendations
        - similar.parquet      -> item-to-item cosine similarity index
        """
        logger.info("Loading recommendation artifacts ...")
        self._load_personal()
        self._load_cold()
        self._load_content()

    def _load_personal(self):
        """Load personal_als.parquet -> user_id: [item_id, ...]"""
        personal_als = pd.read_parquet(RECS_DIR / "personal_als.parquet")
        self.personal = {
            int(uid): grp.sort_values("rank")["item_id"].astype(int).tolist()
            for uid, grp in personal_als.groupby("user_id")
        }
        logger.info("Loaded personal_als: %d users", len(self.personal))

    def _load_cold(self):
        """Load cold_recs.parquet -> [item_id, ...] ordered by popularity."""
        cold_recs = pd.read_parquet(RECS_DIR / "cold_recs.parquet")
        self.cold = (
            cold_recs.sort_values("score", ascending=False)["item_id"]
            .astype(int)
            .tolist()
        )
        logger.info("Loaded cold_recs: %d items", len(self.cold))

    def _load_content(self):
        """
        Load similar.parquet -> track_id: [(similar_track_id, score), ...]
        Schema: track_id, similar_track_id, score, rank
        """
        similar = pd.read_parquet(RECS_DIR / "similar.parquet")
        similar = similar.dropna(subset=["similar_track_id"])
        self.content = {
            int(tid): list(
                zip(
                    grp.sort_values("rank")["similar_track_id"].astype(int).tolist(),
                    grp.sort_values("rank")["score"].tolist(),
                )
            )
            for tid, grp in similar.groupby("track_id")
        }
        logger.info("Loaded similar: %d seed tracks", len(self.content))

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
        Generate online recommendations using similar.parquet.
        - Receive pre-sliced events list (caller controls depth)
        - Collect similar items for each seed
        - Sort by similarity score (descending)
        - Deduplicate (keep first occurrence)
        - Return top k
        """
        if not user_events:
            return []

        candidates: list[tuple[int, float]] = []
        for seed in user_events:
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
            for rec in (recs_offline, recs_online):
                if i < len(rec):
                    item = rec[i]
                    if item not in seen:
                        seen.add(item)
                        blended.append(item)
        return blended[:k]


# ---------------------------------------------------------------------------
# EventStore
# ---------------------------------------------------------------------------
class EventStore:

    def __init__(self, url: str = events_store_url, max_events_per_user: int = MAX_EVENTS_PER_USER):
        self.url = url
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
store = RecommendationService()
event_store = EventStore()


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


@app.get("/get_events", summary="Retrieve recent events for a user")
def get_events(
    user_id: int,
    k: int = Query(ONLINE_HISTORY_DEPTH, ge=1, le=MAX_EVENTS_PER_USER),
):
    events = event_store.get(user_id, k)
    return {"user_id": user_id, "events": events}


@app.get("/recommendations_offline", summary="Offline ALS recommendations (cold fallback)")
def recommendations_offline(
    user_id: int,
    k: int = Query(DEFAULT_K, ge=1, le=100),
):
    store._stats["offline_requests"] += 1
    is_cold = user_id not in store.personal
    if is_cold:
        store._stats["cold_fallbacks"] += 1
    recs = store.get_offline(user_id, k)
    logger.info(
        "offline | user=%d k=%d returned=%d cold=%s [total=%d cold_total=%d]",
        user_id, k, len(recs), is_cold,
        store._stats["offline_requests"], store._stats["cold_fallbacks"],
    )
    return {"recs": recs}


@app.get("/recommendations_online", summary="Online content-based recommendations")
def recommendations_online(
    user_id: int,
    k: int = Query(DEFAULT_K, ge=1, le=100),
):
    store._stats["online_requests"] += 1
    events = event_store.get(user_id, ONLINE_HISTORY_DEPTH)
    recs = store.get_online(user_id, events, k)
    logger.info(
        "online  | user=%d k=%d returned=%d [total=%d]",
        user_id, k, len(recs), store._stats["online_requests"],
    )
    return {"recs": recs}


@app.get("/recommendations", summary="Blended offline + online recommendations")
def recommendations(
    user_id: int,
    k: int = Query(DEFAULT_K, ge=1, le=100),
):
    events = event_store.get(user_id, ONLINE_HISTORY_DEPTH)
    recs_offline = store.get_offline(user_id, k)
    recs_online = store.get_online(user_id, events, k)
    recs = store.blend(recs_offline, recs_online, k)
    return {"recs": recs}

"""
EventStore microservice.

Runs at: http://127.0.0.1:8020

Endpoints:
  POST /put_event  - store a user event (most-recent first)
  GET  /get_events - retrieve recent events for a user
"""
import logging
from fastapi import FastAPI, Query

logger = logging.getLogger("uvicorn.error")

MAX_EVENTS_PER_USER = 10
ONLINE_HISTORY_DEPTH = 3

events_store_url = "http://127.0.0.1:8020"


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
event_store = EventStore()

app = FastAPI(title="EventStore Service")


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

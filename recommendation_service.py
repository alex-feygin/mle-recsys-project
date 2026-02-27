"""
RecSys gateway microservice.

Runs at: http://127.0.0.1:8000

Composes two sub-services via HTTP:
  EventStore            http://127.0.0.1:8020  (events_app.py)
  RecommendationService http://127.0.0.1:8010  (recommendations_app.py)

Endpoints:
  POST /put_event               - store a user event
  GET  /get_events              - retrieve recent events for a user
  GET  /recommendations_offline - ALS recs (cold fallback)
  GET  /recommendations_online  - content-based recs from recent events
  GET  /recommendations         - blended offline + online
"""
import logging

import httpx
from fastapi import FastAPI, HTTPException, Query

logger = logging.getLogger("uvicorn.error")

DEFAULT_K = 10
MAX_EVENTS_PER_USER = 10
ONLINE_HISTORY_DEPTH = 3   # how many recent events to use for online recs

events_store_url = "http://127.0.0.1:8020"
recommendation_store_url = "http://127.0.0.1:8010"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _call(method: str, url: str, **kwargs) -> dict:
    try:
        resp = httpx.request(method, url, timeout=10.0, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as exc:
        logger.error("Upstream call failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}")


def _get(url: str, **params) -> dict:
    return _call("GET", url, params=params)


def _post(url: str, **params) -> dict:
    return _call("POST", url, params=params)


# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------
app = FastAPI(title="RecSys Gateway")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/put_event", summary="Store a user interaction event")
def put_event(user_id: int, item_id: int):
    return _post(f"{events_store_url}/put_event", user_id=user_id, item_id=item_id)


@app.get("/get_events", summary="Retrieve recent events for a user")
def get_events(
    user_id: int,
    k: int = Query(ONLINE_HISTORY_DEPTH, ge=1, le=MAX_EVENTS_PER_USER),
):
    return _get(f"{events_store_url}/get_events", user_id=user_id, k=k)


@app.get("/recommendations_offline", summary="Offline ALS recommendations (cold fallback)")
def recommendations_offline(
    user_id: int,
    k: int = Query(DEFAULT_K, ge=1, le=100),
):
    return _get(
        f"{recommendation_store_url}/recommendations_offline",
        user_id=user_id, k=k,
    )


@app.get("/recommendations_online", summary="Online content-based recommendations")
def recommendations_online(
    user_id: int,
    k: int = Query(DEFAULT_K, ge=1, le=100),
):
    events: list[int] = _get(
        f"{events_store_url}/get_events",
        user_id=user_id, k=ONLINE_HISTORY_DEPTH,
    )["events"]
    events_param = ",".join(str(e) for e in events)
    return _get(
        f"{recommendation_store_url}/recommendations_online",
        user_id=user_id, events=events_param, k=k,
    )


@app.get("/recommendations", summary="Blended offline + online recommendations")
def recommendations(
    user_id: int,
    k: int = Query(DEFAULT_K, ge=1, le=100),
):
    events: list[int] = _get(
        f"{events_store_url}/get_events",
        user_id=user_id, k=ONLINE_HISTORY_DEPTH,
    )["events"]
    events_param = ",".join(str(e) for e in events)

    recs_offline: list[int] = _get(
        f"{recommendation_store_url}/recommendations_offline",
        user_id=user_id, k=k,
    )["recs"]
    recs_online: list[int] = _get(
        f"{recommendation_store_url}/recommendations_online",
        user_id=user_id, events=events_param, k=k,
    )["recs"]

    offline_param = ",".join(str(r) for r in recs_offline)
    online_param  = ",".join(str(r) for r in recs_online)
    return _get(
        f"{recommendation_store_url}/blend",
        recs_offline=offline_param, recs_online=online_param, k=k,
    )

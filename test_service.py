# -*- coding: utf-8 -*-
"""
Structured tests for the RecSys FastAPI service.

Start all three services before running:
    uvicorn events_app:app            --host 0.0.0.0 --port 8020
    uvicorn recommendations_app:app   --host 0.0.0.0 --port 8010
    uvicorn recommendation_service:app --host 0.0.0.0 --port 8000

Run tests:
    pytest test_service.py -v
"""
import pytest
import requests

BASE_URL = "http://localhost:8000"
K = 10

# Known IDs from artifacts produced by the notebook.
# user_id == 1  -> has personal ALS recs AND track 628687 is in similar.parquet
# user_id == 12 -> has personal ALS recs, never receives injected events (Case 2 isolation)
WARM_USER_CASE2 = 12
WARM_USER_CASE3 = 1
WARM_USER_SEED_TRACK = 628687   # track for user 1 that exists in similar.parquet

# User absent from personal_als.parquet (guaranteed cold start)
COLD_USER = 2_374_581

COLD_RECS_EXPECTED = [53404, 178529, 37384, 48951, 148345, 328683, 10216, 52100, 137670, 178495]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def client():
    with requests.Session() as session:
        yield session


@pytest.fixture(scope="session", autouse=True)
def inject_events(client):
    """Inject seed events for WARM_USER_CASE3 once before the full test session."""
    client.post(
        f"{BASE_URL}/put_event",
        params={"user_id": WARM_USER_CASE3, "item_id": WARM_USER_SEED_TRACK},
    )
    client.post(
        f"{BASE_URL}/put_event",
        params={"user_id": WARM_USER_CASE3, "item_id": WARM_USER_SEED_TRACK},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def recs(client: requests.Session, endpoint: str, user_id: int) -> list:
    resp = client.get(f"{BASE_URL}{endpoint}", params={"user_id": user_id, "k": K})
    resp.raise_for_status()
    return resp.json()["recs"]


# ---------------------------------------------------------------------------
# Case 1 - Cold user (no personal recs, no event history)
# ---------------------------------------------------------------------------
def test_cold_offline_returns_popular_items(client):
    assert recs(client, "/recommendations_offline", COLD_USER) == COLD_RECS_EXPECTED[:K]


def test_cold_online_is_empty(client):
    assert recs(client, "/recommendations_online", COLD_USER) == []


def test_cold_blended_equals_cold_recs(client):
    assert recs(client, "/recommendations", COLD_USER) == COLD_RECS_EXPECTED[:K]


# ---------------------------------------------------------------------------
# Case 2 - Warm user with personal recs but no event history
# ---------------------------------------------------------------------------
def test_warm_offline_returns_personal_recs(client):
    result = recs(client, "/recommendations_offline", WARM_USER_CASE2)
    assert result != COLD_RECS_EXPECTED[:K]
    assert len(result) == K
    assert len(set(result)) == K


def test_warm_online_is_empty_without_history(client):
    assert recs(client, "/recommendations_online", WARM_USER_CASE2) == []


def test_warm_blended_equals_offline_without_history(client):
    offline = recs(client, "/recommendations_offline", WARM_USER_CASE2)
    blended = recs(client, "/recommendations",         WARM_USER_CASE2)
    assert blended == offline


# ---------------------------------------------------------------------------
# Case 3 - Warm user with personal recs AND injected event history
# ---------------------------------------------------------------------------
def test_get_events_returns_stored_events(client):
    resp = client.get(
        f"{BASE_URL}/get_events",
        params={"user_id": WARM_USER_CASE3, "k": 3},
    )
    resp.raise_for_status()
    assert len(resp.json()["events"]) > 0


def test_online_non_empty_after_events(client):
    result = recs(client, "/recommendations_online", WARM_USER_CASE3)
    assert len(result) > 0
    assert len(set(result)) == len(result)


def test_blended_length_and_no_duplicates(client):
    result = recs(client, "/recommendations", WARM_USER_CASE3)
    assert len(result) == K
    assert len(set(result)) == K


def test_blended_contains_both_offline_and_online(client):
    offline = set(recs(client, "/recommendations_offline", WARM_USER_CASE3))
    online  = set(recs(client, "/recommendations_online",  WARM_USER_CASE3))
    blended =     recs(client, "/recommendations",         WARM_USER_CASE3)
    assert any(x in offline for x in blended), "Blended must contain offline items"
    assert any(x in online  for x in blended), "Blended must contain online items"

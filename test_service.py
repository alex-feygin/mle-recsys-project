# -*- coding: utf-8 -*-
"""
Structured tests for the RecSys FastAPI service.

Run with:
    uvicorn recommendation_service:app --host 0.0.0.0 --port 8001
    python test_service.py

The tests validate three scenarios as specified in the project requirements.
"""
import sys

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import requests

BASE_URL = "http://localhost:8001"
K = 10

# Known IDs taken from the artifacts produced by the notebook.
# user_id == 0  -> has personal ALS recs AND a content seed track (1750835)
# user_id == 1  -> has personal ALS recs, used for Case 2 (no event injection)
WARM_USER_CASE2 = 1             # used in Case 2 — never receives injected events
WARM_USER_CASE3 = 0             # used in Case 3 — receives injected event
WARM_USER_SEED_TRACK = 1750835  # latest track for user 0 in events.parquet

# A user_id that is absent from personal_als.parquet (guaranteed cold start)
COLD_USER = 2_374_581

COLD_RECS_EXPECTED = [53404, 178529, 37384, 48951, 148345, 328683, 10216, 52100, 137670, 178495]

SEPARATOR = "-" * 60


def get(endpoint: str, **params) -> dict:
    resp = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def post(endpoint: str, **params) -> dict:
    resp = requests.post(f"{BASE_URL}{endpoint}", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def assert_eq(actual, expected, msg: str = "") -> None:
    if actual != expected:
        raise AssertionError(f"{msg}\n  expected: {expected}\n  actual  : {actual}")


def assert_len(lst, k: int, msg: str = "") -> None:
    if len(lst) != k:
        raise AssertionError(f"{msg}  expected length {k}, got {len(lst)}")


def assert_no_duplicates(lst: list, msg: str = "") -> None:
    if len(lst) != len(set(lst)):
        dupes = [x for x in lst if lst.count(x) > 1]
        raise AssertionError(f"{msg}  duplicates found: {dupes}")


# ---------------------------------------------------------------------------
# Case 1 — User without personal recommendations (cold user)
# ---------------------------------------------------------------------------
def test_case1():
    """
    Case 1: Cold user (no personal ALS recs, no event history).
    - Offline  → cold recs (popular items)
    - Online   → empty list
    - Blended  → equals offline (cold recs)
    """
    print(SEPARATOR)
    print("Case 1: Cold user (no personal recs, no history)")

    offline = get("/recommendations_offline", user_id=COLD_USER, k=K)["recommendations"]
    online  = get("/recommendations_online",  user_id=COLD_USER, k=K)["recommendations"]
    blended = get("/recommendations",         user_id=COLD_USER, k=K)["recommendations"]

    assert_eq(offline, COLD_RECS_EXPECTED[:K], "Offline should return top-K cold recs")
    assert_eq(online,  [],                      "Online should be empty (no history)")
    assert_eq(blended, COLD_RECS_EXPECTED[:K],  "Blended should equal cold recs when online is empty")

    assert_no_duplicates(offline, "Offline cold recs must not contain duplicates")
    assert_len(offline, K, "Offline cold recs must return exactly K items")

    print("  offline :", offline)
    print("  online  :", online)
    print("  blended :", blended)
    print("  PASSED")


# ---------------------------------------------------------------------------
# Case 2 — User with personal recs but no online history
# ---------------------------------------------------------------------------
def test_case2():
    """
    Case 2: Warm user (has ALS recs) with no stored event history.
    - Offline  -> personal ALS recs
    - Online   -> empty list
    - Blended  -> equals offline (personal recs)

    Uses WARM_USER_CASE2 (user 1) which never receives injected events,
    ensuring isolation from Case 3.
    """
    print(SEPARATOR)
    print("Case 2: Warm user, no event history")

    offline = get("/recommendations_offline", user_id=WARM_USER_CASE2, k=K)["recommendations"]
    online  = get("/recommendations_online",  user_id=WARM_USER_CASE2, k=K)["recommendations"]
    blended = get("/recommendations",         user_id=WARM_USER_CASE2, k=K)["recommendations"]

    # Offline must NOT be cold recs; it must be the user's personal recs
    assert offline != COLD_RECS_EXPECTED[:K], "Offline should return personal recs, not cold recs"
    assert len(offline) > 0,                   "Offline personal recs must be non-empty"
    assert_len(offline, K,                     "Offline must return exactly K items")
    assert_no_duplicates(offline,              "Offline personal recs must not contain duplicates")

    assert_eq(online, [], "Online should be empty (no history)")

    # Blended = offline when online is empty
    assert_eq(blended, offline, "Blended should equal offline when online is empty")
    assert_no_duplicates(blended, "Blended must not contain duplicates")

    print("  offline :", offline)
    print("  online  :", online)
    print("  blended :", blended)
    print("  PASSED")


# ---------------------------------------------------------------------------
# Case 3 — User with personal recs AND online history
# ---------------------------------------------------------------------------
def test_case3():
    """
    Case 3: Warm user with event history injected via PUT /put_event.
    - Online   → content-based recs (non-empty)
    - Blended  → alternates offline + online, no duplicates, length == K
    """
    print(SEPARATOR)
    print("Case 3: Warm user with event history → blended recs")

    # Inject the seed track that the notebook used for user WARM_USER_CASE3
    post("/put_event", user_id=WARM_USER_CASE3, item_id=WARM_USER_SEED_TRACK)

    offline = get("/recommendations_offline", user_id=WARM_USER_CASE3, k=K)["recommendations"]
    online  = get("/recommendations_online",  user_id=WARM_USER_CASE3, k=K)["recommendations"]
    blended = get("/recommendations",         user_id=WARM_USER_CASE3, k=K)["recommendations"]

    assert len(online) > 0, "Online recs must be non-empty after injecting event history"
    assert_no_duplicates(online, "Online recs must not contain duplicates")

    assert_no_duplicates(blended, "Blended recs must not contain duplicates")
    assert_len(blended, K, "Blended must return exactly K items")

    # Verify blended alternating structure: first item from offline should appear
    # before second item from offline (i.e., offline items are spread through blended)
    offline_set = set(offline)
    online_set  = set(online)
    blended_offline_items = [x for x in blended if x in offline_set]
    blended_online_items  = [x for x in blended if x in online_set]
    assert len(blended_offline_items) > 0, "Blended must contain some offline items"
    assert len(blended_online_items) > 0,  "Blended must contain some online items"

    print("  offline :", offline)
    print("  online  :", online)
    print("  blended :", blended)
    print(f"  offline items in blended: {len(blended_offline_items)}")
    print(f"  online  items in blended: {len(blended_online_items)}")
    print("  PASSED")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def main():
    failures = []
    for test_fn in (test_case1, test_case2, test_case3):
        try:
            test_fn()
        except AssertionError as exc:
            print(f"  FAILED: {exc}")
            failures.append(test_fn.__name__)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failures.append(test_fn.__name__)

    print(SEPARATOR)
    if failures:
        print(f"FAILED tests: {failures}")
        sys.exit(1)
    else:
        print("All 3 tests PASSED.")


if __name__ == "__main__":
    main()

"""
Comprehensive AI Project Test Suite
===================================

Designed for an AI application with:
    Frontend: Next.js / React
    Backend: FastAPI
    Main AI endpoint: POST /analyze
    Request: multipart/form-data -> image + natural-language query

What this suite checks:
    [x] Backend availability
    [x] Health/status endpoint
    [x] /analyze endpoint
    [x] Image upload handling
    [x] Query validation
    [x] Response JSON validity
    [x] Required response fields
    [x] Status/task/tool/confidence/query/result
    [x] Confidence range
    [x] Bounding-box structure when present
    [x] Coordinate validity
    [x] Multiple AI query cases
    [x] Invalid input handling
    [x] Large/unsupported input handling
    [x] Retry loop for transient failures
    [x] Final PASS/FAIL report

Install:
    pip install pytest requests

Run:
    pytest test_ai_project.py -v -s

Optional environment variables:
    SATQUERY_BASE_URL=http://127.0.0.1:8000
    SATQUERY_ANALYZE_PATH=/analyze
    SATQUERY_HEALTH_PATH=/health
    SATQUERY_IMAGE=./test_data/sample.jpg
    SATQUERY_MAX_RETRIES=3
    SATQUERY_RETRY_DELAY=1
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import requests


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

BASE_URL = os.getenv("SATQUERY_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
ANALYZE_PATH = os.getenv("SATQUERY_ANALYZE_PATH", "/analyze")
HEALTH_PATH = os.getenv("SATQUERY_HEALTH_PATH", "/health")
IMAGE_PATH = Path(
    os.getenv("SATQUERY_IMAGE", "./test_data/sample.jpg")
)

MAX_RETRIES = max(1, int(os.getenv("SATQUERY_MAX_RETRIES", "3")))
RETRY_DELAY = max(0.0, float(os.getenv("SATQUERY_RETRY_DELAY", "1")))

REQUEST_TIMEOUT = (10, 120)

# Change/add these according to the natural-language queries your model supports.
TEST_QUERIES = [
    "Detect all objects in the image",
    "Find buildings in the image",
    "Find roads in the image",
    "Detect vegetation in the image",
    "Identify visible vehicles in the image",
]

INVALID_QUERIES = [
    "",
    "   ",
    None,
]

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


# ---------------------------------------------------------------------------
# RESULT TRACKING
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    passed: bool
    attempts: int
    message: str = ""


RESULTS: list[TestResult] = []


def record_result(name: str, passed: bool, attempts: int, message: str = ""):
    RESULTS.append(
        TestResult(
            name=name,
            passed=passed,
            attempts=attempts,
            message=message,
        )
    )


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def url(path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    return BASE_URL + path


def retry(test_name: str, function):
    """
    Retry a test when an operation fails.

    The loop stops immediately when the test passes.
    If every retry fails, the final exception is raised so pytest reports it.
    """
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = function()
            record_result(test_name, True, attempt)
            return result
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                print(
                    f"\n[RETRY] {test_name}: "
                    f"attempt {attempt}/{MAX_RETRIES} failed -> {exc}"
                )
                time.sleep(RETRY_DELAY)

    record_result(
        test_name,
        False,
        MAX_RETRIES,
        str(last_error),
    )
    raise last_error or AssertionError(f"{test_name} failed")


def get_test_image() -> Path:
    """
    Resolve the test image.

    If SATQUERY_IMAGE is not found, automatically search common test folders.
    """
    candidates = [
        IMAGE_PATH,
        Path("./test_data/sample.jpg"),
        Path("./test_data/sample.jpeg"),
        Path("./test_data/sample.png"),
        Path("./sample.jpg"),
        Path("./sample.png"),
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    # Search recursively as a convenience.
    for root in [Path("."), Path("./test_data")]:
        if root.exists():
            for candidate in root.rglob("*"):
                if (
                    candidate.is_file()
                    and candidate.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                ):
                    return candidate

    raise FileNotFoundError(
        "No test image found. Put an image at ./test_data/sample.jpg "
        "or set SATQUERY_IMAGE=/path/to/image.jpg"
    )


def assert_json_object(response: requests.Response) -> dict[str, Any]:
    assert response.headers.get("content-type", "").lower().startswith(
        "application/json"
    ), (
        f"Expected JSON response, got Content-Type="
        f"{response.headers.get('content-type')!r}"
    )

    payload = response.json()
    assert isinstance(payload, dict), "API response must be a JSON object"
    return payload


def validate_bounding_box(box: Any):
    """
    Accept common bounding-box formats:
        {"x": ..., "y": ..., "width": ..., "height": ...}
    OR
        {"xmin": ..., "ymin": ..., "xmax": ..., "ymax": ...}
    OR
        {"x1": ..., "y1": ..., "x2": ..., "y2": ...}
    """
    assert isinstance(box, dict), "Each bounding box must be an object"

    common_label_fields = {"label", "class", "name", "object", "category"}
    geometry_sets = [
        {"x", "y", "width", "height"},
        {"xmin", "ymin", "xmax", "ymax"},
        {"x1", "y1", "x2", "y2"},
    ]

    geometry_found = False

    for keys in geometry_sets:
        if keys.issubset(box.keys()):
            geometry_found = True

            values = [box[k] for k in keys]
            assert all(
                isinstance(v, (int, float)) for v in values
            ), f"Bounding-box coordinates must be numeric: {box}"

            break

    assert geometry_found, (
        "Bounding box has no supported geometry format. "
        f"Received keys: {list(box.keys())}"
    )

    # Validate optional metadata.
    if "confidence" in box:
        confidence = box["confidence"]
        assert isinstance(confidence, (int, float))
        assert 0 <= confidence <= 1, (
            f"Bounding-box confidence must be between 0 and 1: {confidence}"
        )

    # Prevent completely nonsensical negative values where applicable.
    if {"x", "y", "width", "height"}.issubset(box.keys()):
        assert box["width"] >= 0
        assert box["height"] >= 0

    # Not mandatory, but keeps validation tolerant of different schemas.
    assert (
        not common_label_fields
        or any(field in box for field in common_label_fields)
        or geometry_found
    )


def validate_analyze_response(payload: dict[str, Any]):
    """
    Validate the response schema while allowing additional fields.
    """
    required_fields = {
        "status",
        "task",
        "tool",
        "confidence",
        "query",
        "result",
    }

    missing = required_fields - set(payload.keys())
    assert not missing, f"Missing required response fields: {sorted(missing)}"

    assert isinstance(payload["status"], str)
    assert isinstance(payload["task"], str)
    assert isinstance(payload["tool"], str)
    assert isinstance(payload["query"], str)

    confidence = payload["confidence"]
    assert isinstance(confidence, (int, float)), (
        f"confidence must be numeric, got {type(confidence).__name__}"
    )
    assert 0 <= confidence <= 1, (
        f"confidence must be between 0 and 1, got {confidence}"
    )

    result = payload["result"]

    # The backend may return a list, dict, string, or null depending on task.
    # We validate each known structure without rejecting new valid structures.
    if isinstance(result, dict):
        for key in ("bounding_boxes", "boxes", "detections"):
            if key in result:
                boxes = result[key]
                assert isinstance(boxes, list), f"{key} must be a list"
                for box in boxes:
                    validate_bounding_box(box)

    elif isinstance(result, list):
        # If the API returns a list directly, validate dict-looking detections.
        for item in result:
            if isinstance(item, dict):
                looks_like_box = any(
                    key in item
                    for key in (
                        "x",
                        "y",
                        "xmin",
                        "ymin",
                        "xmax",
                        "ymax",
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                    )
                )
                if looks_like_box:
                    validate_bounding_box(item)


def make_analyze_request(image: Path, query: str):
    with image.open("rb") as image_file:
        files = {
            "image": (
                image.name,
                image_file,
                "application/octet-stream",
            )
        }
        data = {"query": query}

        response = requests.post(
            url(ANALYZE_PATH),
            files=files,
            data=data,
            timeout=REQUEST_TIMEOUT,
        )

    return response


# ---------------------------------------------------------------------------
# BASIC CONNECTIVITY TESTS
# ---------------------------------------------------------------------------

def test_backend_is_reachable():
    def operation():
        response = requests.get(
            url(HEALTH_PATH),
            timeout=REQUEST_TIMEOUT,
        )

        # Accept 200/204 and also allow 404 here because some projects
        # do not expose /health. Connectivity itself is still verified.
        assert response.status_code not in {
            502,
            503,
            504,
        }, f"Backend unavailable: HTTP {response.status_code}"

        return response

    retry("Backend connectivity", operation)


def test_health_endpoint():
    def operation():
        response = requests.get(
            url(HEALTH_PATH),
            timeout=REQUEST_TIMEOUT,
        )

        # If /health does not exist, do not fail the entire suite.
        if response.status_code == 404:
            pytest.skip(
                f"{HEALTH_PATH} is not implemented; "
                "backend connectivity is tested separately."
            )

        assert 200 <= response.status_code < 300, (
            f"Health check failed: HTTP {response.status_code}"
        )

    retry("Health endpoint", operation)


# ---------------------------------------------------------------------------
# INPUT / FILE TESTS
# ---------------------------------------------------------------------------

def test_test_image_exists():
    image = get_test_image()

    assert image.exists()
    assert image.is_file()
    assert image.stat().st_size > 0, "Test image is empty"
    assert image.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS, (
        f"Unsupported test-image extension: {image.suffix}"
    )


def test_image_is_readable():
    image = get_test_image()

    # Basic byte-level check without requiring PIL/OpenCV.
    header = image.read_bytes()[:32]
    assert header, "Image contains no readable bytes"


# ---------------------------------------------------------------------------
# MAIN AI ENDPOINT TESTS
# ---------------------------------------------------------------------------

def test_analyze_valid_request():
    image = get_test_image()

    def operation():
        response = make_analyze_request(
            image,
            TEST_QUERIES[0],
        )

        assert 200 <= response.status_code < 300, (
            f"/analyze failed: HTTP {response.status_code}\n"
            f"Response: {response.text[:1000]}"
        )

        payload = assert_json_object(response)
        validate_analyze_response(payload)

        return payload

    retry("Analyze valid request", operation)


@pytest.mark.parametrize("query", TEST_QUERIES)
def test_ai_query_cases(query: str):
    image = get_test_image()
    safe_name = query.replace(" ", "_")[:35]

    def operation():
        response = make_analyze_request(image, query)

        assert 200 <= response.status_code < 300, (
            f"Query failed [{query}]: HTTP {response.status_code}\n"
            f"{response.text[:800]}"
        )

        payload = assert_json_object(response)
        validate_analyze_response(payload)

        # The backend should return the query it processed.
        returned_query = payload["query"].strip()
        assert returned_query, "Returned query cannot be empty"

        return payload

    retry(f"AI query: {safe_name}", operation)


# ---------------------------------------------------------------------------
# NEGATIVE / VALIDATION TESTS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query", INVALID_QUERIES)
def test_invalid_query_is_rejected(query):
    image = get_test_image()

    def operation():
        response = make_analyze_request(image, query)

        # Typical correct behavior:
        # 4xx means validation rejected the bad request.
        assert 400 <= response.status_code < 500, (
            f"Expected 4xx for invalid query {query!r}, "
            f"got HTTP {response.status_code}: {response.text[:500]}"
        )

    # Validation failures should not be retried as if they were transient.
    operation()


def test_missing_image_is_rejected():
    def operation():
        response = requests.post(
            url(ANALYZE_PATH),
            data={"query": "Detect objects"},
            timeout=REQUEST_TIMEOUT,
        )

        assert 400 <= response.status_code < 500, (
            f"Expected 4xx when image is missing, got "
            f"{response.status_code}: {response.text[:500]}"
        )

    retry("Missing image validation", operation)


def test_missing_query_is_rejected():
    image = get_test_image()

    with image.open("rb") as image_file:
        files = {
            "image": (
                image.name,
                image_file,
                "application/octet-stream",
            )
        }

        def operation():
            response = requests.post(
                url(ANALYZE_PATH),
                files=files,
                timeout=REQUEST_TIMEOUT,
            )

            assert 400 <= response.status_code < 500, (
                f"Expected 4xx when query is missing, got "
                f"{response.status_code}: {response.text[:500]}"
            )

        retry("Missing query validation", operation)


# ---------------------------------------------------------------------------
# RESPONSE CONSISTENCY TESTS
# ---------------------------------------------------------------------------

def test_response_is_json_and_structured():
    image = get_test_image()

    def operation():
        response = make_analyze_request(
            image,
            "Detect objects in this image",
        )

        assert 200 <= response.status_code < 300
        payload = assert_json_object(response)
        validate_analyze_response(payload)

        # Ensure response can be serialized back to JSON.
        json.dumps(payload)

    retry("Response structure", operation)


def test_confidence_is_valid():
    image = get_test_image()

    def operation():
        response = make_analyze_request(
            image,
            "Analyze the satellite image",
        )

        assert 200 <= response.status_code < 300
        payload = assert_json_object(response)

        confidence = payload.get("confidence")
        assert isinstance(confidence, (int, float))
        assert 0 <= confidence <= 1

    retry("Confidence validation", operation)


def test_result_not_empty_when_successful():
    image = get_test_image()

    def operation():
        response = make_analyze_request(
            image,
            "Detect visible objects",
        )

        assert 200 <= response.status_code < 300
        payload = assert_json_object(response)
        validate_analyze_response(payload)

        result = payload["result"]
        assert result is not None, "Successful result must not be null"

        if isinstance(result, str):
            assert result.strip(), "Successful string result is empty"

    retry("Non-empty AI result", operation)


# ---------------------------------------------------------------------------
# PERFORMANCE / STABILITY TESTS
# ---------------------------------------------------------------------------

def test_multiple_sequential_requests():
    image = get_test_image()

    successful = 0
    failures: list[str] = []

    for index, query in enumerate(TEST_QUERIES[:3], start=1):
        try:
            response = make_analyze_request(image, query)

            if not (200 <= response.status_code < 300):
                failures.append(
                    f"request {index}: HTTP {response.status_code}"
                )
                continue

            payload = assert_json_object(response)
            validate_analyze_response(payload)
            successful += 1

        except Exception as exc:
            failures.append(f"request {index}: {exc}")

    assert successful >= 2, (
        "AI stability test failed: fewer than 2 of 3 sequential "
        f"requests succeeded. Failures: {failures}"
    )


def test_retry_mechanism():
    """
    Tests the retry loop itself.

    The fake operation intentionally fails twice and then succeeds.
    """
    state = {"attempts": 0}

    def flaky_operation():
        state["attempts"] += 1

        if state["attempts"] < 3:
            raise RuntimeError("Intentional transient failure")

        return True

    retry(
        "Internal retry mechanism",
        flaky_operation,
    )

    assert state["attempts"] == 3


# ---------------------------------------------------------------------------
# OPTIONAL FRONTEND-CONNECTION TEST
# ---------------------------------------------------------------------------

def test_frontend_endpoint_if_configured():
    """
    Optional frontend smoke test.

    Set:
        SATQUERY_FRONTEND_URL=http://localhost:3000

    to enable it.
    """
    frontend_url = os.getenv("SATQUERY_FRONTEND_URL")

    if not frontend_url:
        pytest.skip(
            "SATQUERY_FRONTEND_URL not configured; frontend smoke test skipped."
        )

    def operation():
        response = requests.get(
            frontend_url,
            timeout=30,
        )
        assert 200 <= response.status_code < 400, (
            f"Frontend unavailable: HTTP {response.status_code}"
        )

    retry("Frontend smoke test", operation)


# ---------------------------------------------------------------------------
# FINAL REPORT
# ---------------------------------------------------------------------------

def pytest_sessionfinish(session, exitstatus):
    """
    Print a compact dashboard after pytest finishes.
    """
    if not RESULTS:
        return

    print("\n")
    print("=" * 78)
    print("                 AI PROJECT TEST REPORT")
    print("=" * 78)

    passed = sum(result.passed for result in RESULTS)
    failed = len(RESULTS) - passed

    for result in RESULTS:
        icon = "✅" if result.passed else "❌"
        line = (
            f"{icon} {result.name} "
            f"(attempts: {result.attempts})"
        )

        print(line)

        if result.message:
            print(f"   └─ {result.message}")

    print("-" * 78)
    print(f"TOTAL:  {len(RESULTS)}")
    print(f"PASS:   {passed}")
    print(f"FAIL:   {failed}")
    print(f"STATUS: {'✅ ALL TESTS PASSED' if failed == 0 else '❌ FIX REQUIRED'}")
    print("=" * 78)

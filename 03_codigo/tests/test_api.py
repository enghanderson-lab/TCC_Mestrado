import json

import pytest
from fastapi.testclient import TestClient

from video_search import api
from video_search.pipeline import IndexSummary, SearchResult


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "INDEX_ROOT", tmp_path / "index")
    return TestClient(api.app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_index_job_rejects_missing_video(client, tmp_path):
    response = client.post(
        "/index",
        json={"video_path": str(tmp_path / "nao_existe.mp4"), "name": "meu_video"},
    )
    assert response.status_code == 404


def test_create_index_job_runs_and_reports_status(client, tmp_path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake video bytes")

    def fake_run_index(video_path, output_dir, interval=1.0, batch_size=16, limit=0, on_progress=None):
        if on_progress is not None:
            on_progress(5)
        return IndexSummary(output_dir=output_dir, frame_count=5)

    monkeypatch.setattr(api, "run_index", fake_run_index)

    response = client.post("/index", json={"video_path": str(video_path), "name": "meu_video"})
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    status_response = client.get(f"/index/jobs/{job_id}")
    assert status_response.status_code == 200
    body = status_response.json()
    assert body["status"] == "done"
    assert body["frame_count"] == 5


def test_get_index_job_not_found(client):
    response = client.get("/index/jobs/inexistente")
    assert response.status_code == 404


def test_create_index_job_rejects_invalid_name(client, tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake video bytes")

    response = client.post("/index", json={"video_path": str(video_path), "name": "../escape"})
    assert response.status_code == 400


def test_search_rejects_missing_index(client):
    response = client.post("/search", json={"query": "homem de camisa branca", "index": "nao_existe"})
    assert response.status_code == 404


def test_search_returns_results(client, monkeypatch):
    index_dir = api.INDEX_ROOT / "meu_video"
    index_dir.mkdir(parents=True)
    (index_dir / "records.json").write_text("[]", encoding="utf-8")

    def fake_run_search(query, index_dir, top_k=12, with_caption=True):
        return [
            SearchResult(
                frame_index=3,
                timestamp_sec=6.0,
                video="video.mp4",
                retrieval_score=0.42,
                confidence=0.81,
                caption="um homem com camisa branca",
            )
        ]

    monkeypatch.setattr(api, "run_search", fake_run_search)

    response = client.post("/search", json={"query": "homem de camisa branca", "index": "meu_video"})
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 1
    assert results[0]["frame_index"] == 3
    assert results[0]["confidence"] == 0.81


def test_list_indexes(client):
    index_dir = api.INDEX_ROOT / "meu_video"
    index_dir.mkdir(parents=True)
    (index_dir / "records.json").write_text(json.dumps([{}, {}]), encoding="utf-8")
    (index_dir / "metadata.json").write_text(json.dumps({"model_name": "siglip"}), encoding="utf-8")

    response = client.get("/indexes")
    assert response.status_code == 200
    body = response.json()
    assert body == [{"name": "meu_video", "frame_count": 2, "model_name": "siglip"}]

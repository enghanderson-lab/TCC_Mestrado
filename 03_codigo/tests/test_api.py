"""Teste de ponta a ponta da API: FastAPI + Container reais (SQLite, FAISS,
OpenCV, GpuExecutor, jobs em background); só os modelos são fakes."""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import FakeCaptioner, FakeImageEmbedder, FakeScorer, FakeTextEmbedder, write_video
from video_search.application.indexing_service import IndexingService
from video_search.application.search_service import SearchService
from video_search.bootstrap import Container
from video_search.config import Settings
from video_search.entrypoints.api.main import create_app

KEY = {"X-Api-Key": "test-key"}
GRAY = [(v, v, v) for v in (0, 60, 120, 200, 255)] * 4


def _container(tmp_path) -> Container:
    settings = Settings(
        api_key="test-key",
        db_path=tmp_path / "api.db",
        index_root=tmp_path / "index",
        upload_dir=tmp_path / "uploads",
    )
    c = Container(settings)
    # Substitui só os modelos: o resto da montagem (jobs, câmeras, SQLite) é o real.
    c._instances["indexing"] = IndexingService(FakeImageEmbedder(), c.frames, c.indexes, c.gpu)
    c._instances["search"] = SearchService(
        FakeTextEmbedder(), FakeCaptioner(), FakeScorer(), c.indexes, c.frames, c.gpu
    )
    return c


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(_container(tmp_path), load_models=False)) as test_client:
        yield test_client


def _wait_job(client, job_id, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/v1/index/{job_id}/status", headers=KEY).json()
        if body["status"] != "running":
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} nao terminou")


def _upload(client, tmp_path, camera_id="cam1"):
    video = write_video(tmp_path / f"{camera_id}.mp4", GRAY)
    with open(video, "rb") as f:
        resp = client.post(
            "/api/v1/index/upload", headers=KEY,
            data={"camera_id": camera_id, "interval": "0.2"},
            files={"file": (f"{camera_id}.mp4", f, "video/mp4")},
        )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_auth_and_health(client):
    health = client.get("/api/v1/health").json()
    assert health["status"] == "ok"
    assert health["models_loaded"] is False
    assert health["active_jobs"] == 0

    assert client.get("/api/v1/cameras").status_code == 401
    assert client.get("/api/v1/cameras", headers={"X-Api-Key": "errada"}).status_code == 401
    assert client.post("/api/v1/search", headers=KEY, json={"query": "x"}).status_code == 404


def test_full_flow_upload_index_search_frames(client, tmp_path):
    job = _upload(client, tmp_path)
    assert job["status"] == "running"
    assert job["frames_estimated"] == 20

    done = _wait_job(client, job["job_id"])
    assert done["status"] == "done", done
    assert done["frames_indexed"] == 20

    cameras = client.get("/api/v1/cameras", headers=KEY).json()
    assert [c["camera_id"] for c in cameras] == ["cam1"]
    assert cameras[0]["frame_count"] == 20

    fast = client.post("/api/v1/search", headers=KEY, json={"query": "cena clara", "top_k": 3}).json()
    assert fast["mode"] == "fast"
    assert len(fast["results"]) == 3
    assert all(r["description"] is None for r in fast["results"])

    detailed = client.post(
        "/api/v1/search", headers=KEY,
        json={"query": "cena clara", "mode": "detailed", "top_k": 3, "camera_ids": ["cam1"]},
    ).json()
    top = detailed["results"][0]
    assert top["description"] == "cena clara"
    assert top["confidence"] == pytest.approx(1.0)

    frame = client.get(top["frame_url"], headers=KEY)
    assert frame.status_code == 200
    assert frame.headers["content-type"] == "image/jpeg"
    assert frame.content[:2] == b"\xff\xd8"

    clip = client.get("/api/v1/clips/cam1/0?duration=2", headers=KEY)
    assert clip.status_code == 200, clip.text
    assert clip.headers["content-type"] == "video/mp4"
    assert b"ftyp" in clip.content[:32]

    assert client.post(
        "/api/v1/search", headers=KEY, json={"query": "x", "camera_ids": ["nao-existe"]}
    ).status_code == 404


def test_websocket_progress_and_cancel_finished(client, tmp_path):
    job = _upload(client, tmp_path)

    with client.websocket_connect(f"/api/v1/index/ws/{job['job_id']}?api_key=test-key") as ws:
        messages = []
        while True:
            msg = ws.receive_json()
            messages.append(msg)
            if msg["status"] != "running":
                break
    assert messages[-1]["status"] == "done"

    with client.websocket_connect(f"/api/v1/index/ws/{job['job_id']}?api_key=errada") as ws:
        assert ws.receive_json() == {"error": "unauthorized"}

    assert client.delete(f"/api/v1/index/{job['job_id']}", headers=KEY).status_code == 400
    assert client.delete("/api/v1/index/nao-existe", headers=KEY).status_code == 404


def test_upload_rejects_bad_extension_and_missing_path(client):
    resp = client.post(
        "/api/v1/index/upload", headers=KEY,
        data={"camera_id": "x"}, files={"file": ("nota.txt", b"abc", "text/plain")},
    )
    assert resp.status_code == 400
    resp = client.post("/api/v1/index/start", headers=KEY, json={"camera_id": "x", "video_path": "nao/existe.mp4"})
    assert resp.status_code == 400


def test_delete_camera_removes_index(client, tmp_path):
    _wait_job(client, _upload(client, tmp_path)["job_id"])
    index_dir = Path(client.get("/api/v1/cameras", headers=KEY).json()[0]["output_dir"])
    assert index_dir.exists()

    assert client.delete("/api/v1/cameras/cam1", headers=KEY).json() == {"deleted": "cam1"}
    assert not index_dir.exists()
    assert client.get("/api/v1/cameras", headers=KEY).json() == []
    assert client.delete("/api/v1/cameras/cam1", headers=KEY).status_code == 404


def test_job_status_survives_restart(tmp_path):
    with TestClient(create_app(_container(tmp_path), load_models=False)) as first:
        job_id = _upload(first, tmp_path)["job_id"]
        assert _wait_job(first, job_id)["status"] == "done"

    # nova instância da API sobre o mesmo banco
    with TestClient(create_app(_container(tmp_path), load_models=False)) as second:
        status = second.get(f"/api/v1/index/{job_id}/status", headers=KEY).json()
        assert status["status"] == "done"
        assert status["frames_indexed"] == 20
        assert len(second.get("/api/v1/cameras", headers=KEY).json()) == 1


class _SlowEmbedder(FakeImageEmbedder):
    def encode_images(self, images, profiler=None):
        time.sleep(0.1)
        return super().encode_images(images, profiler)


def test_cancel_running_job_stops_indexing(tmp_path):
    c = _container(tmp_path)
    c._instances["indexing"] = IndexingService(_SlowEmbedder(), c.frames, c.indexes, c.gpu)
    video = write_video(tmp_path / "long.mp4", GRAY * 10)  # 200 frames

    with TestClient(create_app(c, load_models=False)) as client:
        job = client.post(
            "/api/v1/index/start", headers=KEY,
            json={"camera_id": "long", "video_path": video, "interval": 0.2, "batch_size": 1},
        ).json()
        time.sleep(0.3)
        assert client.get("/api/v1/health").json()["active_jobs"] == 1
        assert client.delete(f"/api/v1/index/{job['job_id']}", headers=KEY).json() == {"cancelled": job["job_id"]}

        final = _wait_job(client, job["job_id"])
        assert final["status"] == "cancelled"
        assert final["frames_indexed"] < 200
        # job cancelado nao registra a camera como indexada
        assert client.get("/api/v1/cameras", headers=KEY).json() == []
        assert client.get("/api/v1/health").json()["active_jobs"] == 0

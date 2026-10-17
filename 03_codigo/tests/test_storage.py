import os
import time

import numpy as np
import pytest

from video_search.adapters.storage.faiss_index import FAISS_AVAILABLE, FaissIndex
from video_search.adapters.storage.file_index import FileIndexRepository
from video_search.adapters.storage.sqlite import SQLiteDatabase
from video_search.adapters.storage.sqlite_cameras import SQLiteCameraRepository
from video_search.adapters.storage.sqlite_jobs import SQLiteJobRepository
from video_search.domain.entities import Camera, FrameRecord, IndexJob, JobStatus
from video_search.domain.mmr import MmrReranker

needs_faiss = pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss nao instalado")


def _vectors(n=6, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _build(path, vectors):
    index = FaissIndex(use_gpu=False)
    for i, vec in enumerate(vectors):
        index.add(vec, FrameRecord(index=i, timestamp_sec=i * 2.0, video="v.mp4"))
    index.save(path)


@needs_faiss
def test_faiss_search_returns_stored_vectors(tmp_path):
    vectors = _vectors()
    _build(tmp_path / "idx", vectors)
    loaded = FaissIndex.load(tmp_path / "idx", use_gpu=False)

    results = loaded.search_with_embeddings(vectors[3], top_k=3)

    assert len(results) == 3
    score, rec, emb = results[0]
    assert rec.index == 3
    assert score > 0.99
    np.testing.assert_allclose(emb, vectors[3], atol=1e-6)
    # top_k maior que o indice nao gera ids invalidos (-1)
    assert len(loaded.search_with_embeddings(vectors[0], top_k=50)) == len(vectors)


@needs_faiss
def test_repository_caches_until_reindexed_and_deletes(tmp_path):
    repo = FileIndexRepository()
    path = str(tmp_path / "idx")
    _build(path, _vectors(n=4))

    first = repo.load(path)
    assert repo.load(path) is first

    _build(path, _vectors(n=5, seed=1))
    records = tmp_path / "idx" / "records.json"
    stat = records.stat()
    os.utime(records, (stat.st_atime, stat.st_mtime + 10))  # garante mtime novo

    reloaded = repo.load(path)
    assert reloaded is not first
    assert len(reloaded) == 5

    repo.delete(path)
    assert not (tmp_path / "idx").exists()
    with pytest.raises(FileNotFoundError):
        repo.load(path)


def test_mmr_prefers_diverse_candidate():
    query = np.array([1.0, 0.0], dtype=np.float32)
    a = np.array([0.99, 0.141], dtype=np.float32)
    b = np.array([0.9, -0.436], dtype=np.float32)
    recs = [FrameRecord(index=i, timestamp_sec=float(i), video="v.mp4") for i in range(3)]
    candidates = [(0.99, recs[0], a), (0.98, recs[1], a.copy()), (0.90, recs[2], b)]

    picked = MmrReranker(0.5).rerank(query, candidates, k=2)

    assert [rec.index for _, rec in picked] == [0, 2]


def test_sqlite_cameras_crud(tmp_path):
    repo = SQLiteCameraRepository(SQLiteDatabase(tmp_path / "db.sqlite"))
    repo.upsert(Camera("cam1", "a.mp4", "index/cam1", frame_count=3, indexed_at="2026-01-01"))
    repo.upsert(Camera("cam2", "b.mp4", "index/cam2", frame_count=4, indexed_at="2026-02-01"))

    assert repo.get("cam1").frame_count == 3
    assert [c.camera_id for c in repo.list_all()] == ["cam2", "cam1"]  # mais recente primeiro
    assert repo.delete("cam1") is True
    assert repo.delete("cam1") is False
    assert repo.get("cam1") is None


def test_sqlite_jobs_persist_and_mark_interrupted(tmp_path):
    db_path = tmp_path / "db.sqlite"
    repo = SQLiteJobRepository(SQLiteDatabase(db_path))
    running = IndexJob(camera_id="cam1", frames_estimated=10)
    done = IndexJob(camera_id="cam2")
    done.frames_indexed = 7
    done.finish(JobStatus.DONE)
    repo.save(running)
    repo.save(done)

    # "reinício": novo repositório sobre o mesmo arquivo
    restarted = SQLiteJobRepository(SQLiteDatabase(db_path))
    assert restarted.mark_interrupted() == 1

    interrupted = restarted.get(running.job_id)
    assert interrupted.status == JobStatus.ERROR
    assert "reiniciada" in interrupted.error_msg
    assert interrupted.finished_at is not None
    kept = restarted.get(done.job_id)
    assert (kept.status, kept.frames_indexed) == (JobStatus.DONE, 7)
    assert kept.elapsed_sec <= time.time() - kept.started_at + 1

from typing import List, Optional
from pydantic import BaseModel, Field


class IndexStartRequest(BaseModel):
    camera_id: str
    video_path: str
    interval: float = Field(default=2.0, ge=0.1, le=60.0)
    batch_size: int = Field(default=16, ge=1, le=64)


class IndexStatusResponse(BaseModel):
    job_id: str
    camera_id: str
    status: str
    frames_indexed: int
    frames_estimated: int
    elapsed_sec: float
    fps: float
    error: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: str = Field(default="fast", pattern="^(fast|detailed)$")
    camera_ids: List[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=50)
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)


class FrameResult(BaseModel):
    camera_id: str
    timestamp_sec: float
    score: float
    frame_url: str
    description: Optional[str] = None
    confidence: Optional[float] = None


class SearchResponse(BaseModel):
    query_time_ms: int
    mode: str
    results: List[FrameResult]


class CameraInfo(BaseModel):
    camera_id: str
    video_path: str
    output_dir: str
    frame_count: int
    indexed_at: Optional[str] = None
    status: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    vram_used_mb: Optional[float] = None
    vram_total_mb: Optional[float] = None
    active_jobs: int

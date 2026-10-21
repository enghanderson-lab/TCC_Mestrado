"""O container só carrega cada modelo quando ele é de fato usado."""

from conftest import FakeCaptioner, FakeImageEmbedder, FakeScorer, FakeTextEmbedder
from video_search.bootstrap import Container
from video_search.config import Settings


class _CountingModels:
    def __init__(self):
        self.loaded = []
        self.siglip = type("Siglip", (FakeImageEmbedder, FakeTextEmbedder), {})()

    def get_siglip(self):
        self.loaded.append("siglip")
        return self.siglip

    def get_qwen(self):
        self.loaded.append("qwen")
        return FakeCaptioner()

    def get_scorer(self):
        self.loaded.append("nli")
        return FakeScorer()

    def unload_models(self):
        pass


def test_search_service_defers_qwen_and_nli(tmp_path):
    c = Container(Settings(db_path=tmp_path / "db.sqlite"))
    c.models = _CountingModels()
    try:
        search = c.search
        assert set(c.models.loaded) == {"siglip"}   # FAST e indexação não pagam Qwen/NLI

        search._scorer.score("q", ["legenda"])      # primeiro uso real do NLI
        assert "nli" in c.models.loaded and "qwen" not in c.models.loaded
    finally:
        c.close()

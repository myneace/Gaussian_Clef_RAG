"""Integration tests for the gaussian_rag library facade (Phase 09-01: LIB-01 to LIB-04).

Tests verify:
  LIB-01 - Top-level package exports and version.
  LIB-02 - UniversalIngestor handling Markdown, JSON records, raw strings, CSV, and files.
  LIB-03 - GaussianRAG facade: ingest, search, save/load round-trip.
  LIB-04 - Standalone independent operation without cloud tokens.
"""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# LIB-01: Top-level imports and version
# ---------------------------------------------------------------------------


class TestTopLevelExports:
    """LIB-01 — Package exports and version string."""

    def test_import_gaussian_rag_module(self):
        import gaussian_rag  # noqa: F401
        assert hasattr(gaussian_rag, "__version__")
        assert gaussian_rag.__version__ == "0.1.0"

    def test_import_gaussian_rag_facade(self):
        from gaussian_rag import GaussianRAG
        assert GaussianRAG is not None

    def test_import_universal_ingestor(self):
        from gaussian_rag import UniversalIngestor
        assert UniversalIngestor is not None

    def test_import_core_components(self):
        from gaussian_rag import (
            GaussianEmbedder,
            KnowledgeStore,
            GaussianRetriever,
            TaxonomyRouter,
        )
        assert all(
            cls is not None
            for cls in [GaussianEmbedder, KnowledgeStore, GaussianRetriever, TaxonomyRouter]
        )

    def test_import_core_types(self):
        from gaussian_rag import (
            GaussianKnowledge,
            QueryRepresentation,
            RetrievedChunk,
            SenseNode,
            DisambiguationPacket,
            MemoryOps,
        )
        for cls in [
            GaussianKnowledge,
            QueryRepresentation,
            RetrievedChunk,
            SenseNode,
            DisambiguationPacket,
            MemoryOps,
        ]:
            assert cls is not None

    def test_all_list_contents(self):
        import gaussian_rag
        expected_names = [
            "GaussianRAG",
            "UniversalIngestor",
            "GaussianEmbedder",
            "KnowledgeStore",
            "GaussianRetriever",
            "TaxonomyRouter",
            "GaussianKnowledge",
            "QueryRepresentation",
            "RetrievedChunk",
            "__version__",
        ]
        for name in expected_names:
            assert name in gaussian_rag.__all__, f"'{name}' missing from __all__"


# ---------------------------------------------------------------------------
# Fixture: a minimal GaussianRAG instance (no cloud, no LLM)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rag():
    """Standalone GaussianRAG instance — no cloud tokens required."""
    from gaussian_rag import GaussianRAG
    return GaussianRAG.create(
        dimension=384,
        embedding_model="all-MiniLM-L6-v2",
        use_clef=False,
        use_generator=False,
    )


# ---------------------------------------------------------------------------
# LIB-02: UniversalIngestor
# ---------------------------------------------------------------------------


class TestUniversalIngestor:
    """LIB-02 — Format-agnostic ingestion."""

    def test_ingest_raw_strings(self, rag):
        """Raw string list is chunked and stored."""
        from gaussian_rag import GaussianKnowledge
        texts = [
            "Weigh your flour instead of scooping it.",
            "Rest the dough before you knead it.",
            "Shape the loaf with a light hand.",
        ]
        nodes = rag.ingestor.ingest_texts(texts)
        assert len(nodes) >= len(texts), "Expect at least one node per text"
        assert all(isinstance(n, GaussianKnowledge) for n in nodes)

    def test_ingest_markdown_heading_metadata(self, rag):
        """Markdown headings are captured in node metadata breadcrumbs."""
        md = """# The 12 Laws of Good Bread

## Law 1: Weigh, Don't Scoop

A cup of flour is a guess; 125 grams is a fact.
Scooping from the bag packs in extra flour unnoticed.

## Law 2: Hydrate Before You Knead

Mix flour and water alone, then rest for thirty minutes.
"""
        nodes = rag.ingestor.ingest_markdown(md, document_id="bakers-handbook-md-test")
        assert len(nodes) >= 2, "Expected at least one node per heading section"

        texts_with_breadcrumbs = [
            n for n in nodes if "breadcrumb" in n.metadata
        ]
        assert texts_with_breadcrumbs, "No nodes have 'breadcrumb' in metadata"

        breadcrumbs = {n.metadata.get("breadcrumb") for n in texts_with_breadcrumbs}
        assert any("Law 1" in b for b in breadcrumbs if b), (
            "Expected breadcrumb containing 'Law 1'"
        )

    def test_ingest_markdown_preserves_heading_level(self, rag):
        """Heading level is stored in metadata."""
        md = """# Top Level

## Sub Section

Some content here.
"""
        nodes = rag.ingestor.ingest_markdown(md, document_id="heading-level-test")
        levels = [n.metadata.get("heading_level") for n in nodes]
        assert any(lvl == 2 for lvl in levels), "Expected heading_level=2 node"

    def test_ingest_records_json_dicts(self, rag):
        """JSON records are ingested with field extraction."""
        records: list[dict[str, Any]] = [
            {"id": "law-1", "text": "Weigh your flour, do not scoop it", "law": 1},
            {"id": "law-2", "text": "Let the dough rest before kneading", "law": 2},
            {"id": "law-3", "text": "Salt is not optional", "law": 3},
        ]
        nodes = rag.ingestor.ingest_records(records, text_field="text", id_field="id")
        assert len(nodes) >= 3
        # Metadata should include the non-text fields
        law_values = [n.metadata.get("law") for n in nodes if n.metadata.get("law")]
        assert len(law_values) >= 1, "Expected 'law' field in node metadata"

    def test_ingest_records_custom_text_field(self, rag):
        """Custom text_field parameter is respected."""
        records = [
            {"title": "Recipe Index", "description": "Every baker eventually masters the same three basics."},
        ]
        nodes = rag.ingestor.ingest_records(records, text_field="description")
        assert len(nodes) >= 1
        assert "masters the same three basics" in nodes[0].text.lower()

    def test_ingest_file_plain_text(self, rag):
        """Plain .txt file is ingested as raw text."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(
                "Law 1: Weigh, don't scoop.\n\n"
                "Law 2: Hydrate before you knead.\n"
            )
            txt_path = Path(fh.name)
        try:
            nodes = rag.ingestor.ingest_file(txt_path)
            assert len(nodes) >= 1
        finally:
            txt_path.unlink(missing_ok=True)

    def test_ingest_file_markdown(self, rag):
        """Markdown .md file is parsed with heading metadata."""
        content = "# Baking\n\n## Law 1\n\nWeigh, don't scoop.\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(content)
            md_path = Path(fh.name)
        try:
            nodes = rag.ingestor.ingest_file(md_path)
            assert len(nodes) >= 1
            breadcrumb_nodes = [n for n in nodes if n.metadata.get("breadcrumb")]
            assert breadcrumb_nodes, "Markdown file should produce breadcrumb metadata"
        finally:
            md_path.unlink(missing_ok=True)

    def test_ingest_file_json(self, rag):
        """JSON array file is ingested as records."""
        data = [
            {"id": "r1", "text": "Flour absorbs water differently in each kitchen."},
            {"id": "r2", "text": "Cold slows the yeast but not the flavour."},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as fh:
            json.dump(data, fh)
            json_path = Path(fh.name)
        try:
            nodes = rag.ingestor.ingest_file(json_path)
            assert len(nodes) >= 2
        finally:
            json_path.unlink(missing_ok=True)

    def test_ingest_file_jsonl(self, rag):
        """JSONL file is parsed line-by-line."""
        lines = [
            json.dumps({"id": "a", "text": "Line one passage content."}),
            json.dumps({"id": "b", "text": "Line two passage content."}),
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as fh:
            fh.write("\n".join(lines))
            jsonl_path = Path(fh.name)
        try:
            nodes = rag.ingestor.ingest_file(jsonl_path)
            assert len(nodes) >= 2
        finally:
            jsonl_path.unlink(missing_ok=True)

    def test_ingest_file_csv(self, rag):
        """CSV file is ingested using header row as field names."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
        ) as fh:
            writer = csv.DictWriter(fh, fieldnames=["id", "text", "category"])
            writer.writeheader()
            writer.writerow({"id": "c1", "text": "Hydration changes how the dough feels.", "category": "baking"})
            writer.writerow({"id": "c2", "text": "A sharp blade scores better than a dull one.", "category": "technique"})
            csv_path = Path(fh.name)
        try:
            nodes = rag.ingestor.ingest_file(csv_path)
            assert len(nodes) >= 2
            categories = [n.metadata.get("category") for n in nodes]
            assert "baking" in categories or "technique" in categories
        finally:
            csv_path.unlink(missing_ok=True)

    def test_ingest_directory(self, rag):
        """Directory walk ingests all supported file types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "file1.txt").write_text("First document text.", encoding="utf-8")
            (base / "file2.md").write_text(
                "# Doc\n\n## Section\n\nSecond document text.", encoding="utf-8"
            )
            nodes = rag.ingestor.ingest_directory(base)
            assert len(nodes) >= 2


# ---------------------------------------------------------------------------
# LIB-03: GaussianRAG facade
# ---------------------------------------------------------------------------


class TestGaussianRAGFacade:
    """LIB-03 — High-level GaussianRAG facade API."""

    @pytest.fixture(autouse=True)
    def fresh_rag(self):
        """Each test gets a clean, independent RAG instance."""
        from gaussian_rag import GaussianRAG
        self.rag = GaussianRAG.create(use_clef=False, use_generator=False)

    # -- Ingest helpers -------------------------------------------------------

    def test_add_texts_returns_nodes(self):
        from gaussian_rag import GaussianKnowledge
        nodes = self.rag.add_texts(
            ["Weigh your flour, do not scoop it.", "Rest the dough before shaping."]
        )
        assert len(nodes) >= 2
        assert all(isinstance(n, GaussianKnowledge) for n in nodes)

    def test_add_texts_with_metadata(self):
        nodes = self.rag.add_texts(
            ["Law 1 text", "Law 2 text"],
            metadata=[{"law": 1}, {"law": 2}],
        )
        assert len(nodes) >= 2
        law_vals = [n.metadata.get("law") for n in nodes]
        assert 1 in law_vals or 2 in law_vals

    def test_add_markdown(self):
        md = "# Laws\n\n## Law 1\n\nWeigh, don't scoop.\n"
        nodes = self.rag.add_markdown(md, document_id="test-laws")
        assert len(nodes) >= 1

    def test_add_records(self):
        records = [
            {"id": "l1", "text": "Weigh, don't scoop."},
            {"id": "l2", "text": "Salt is not optional."},
        ]
        nodes = self.rag.add_records(records)
        assert len(nodes) >= 2

    def test_add_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as fh:
            fh.write("Good bread is built by those who respect the clock.")
            txt_path = Path(fh.name)
        try:
            nodes = self.rag.add_file(txt_path)
            assert len(nodes) >= 1
        finally:
            txt_path.unlink(missing_ok=True)

    # -- Search ---------------------------------------------------------------

    def test_search_returns_relevant_chunks(self):
        from gaussian_rag import RetrievedChunk
        self.rag.add_texts([
            "Weigh your flour. A scooped cup quietly adds fifty extra grams.",
            "Trust the long cold rise. Overnight in the fridge builds flavour.",
            "Handle the dough gently so the bubbles you grew survive shaping.",
        ])
        results = self.rag.search("Why is my loaf dense?", top_k=3)
        assert isinstance(results, list)
        assert len(results) >= 1
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_search_returns_scores(self):
        self.rag.add_texts([
            "Fermentation keeps working whether you are watching or not.",
        ])
        results = self.rag.search("Time and fermentation", top_k=5)
        for chunk in results:
            assert chunk.confidence is not None
            assert chunk.rank is not None

    def test_search_empty_store_returns_empty(self):
        results = self.rag.search("anything")
        assert results == []

    def test_item_count(self):
        assert self.rag.item_count == 0
        self.rag.add_texts(["Single passage."])
        assert self.rag.item_count >= 1

    # -- Save / Load round-trip -----------------------------------------------

    def test_save_and_load_round_trip(self):
        """Saved store can be loaded and searched with the same results."""
        texts = [
            "Weigh your flour on a scale instead of scooping a cup.",
            "Let the dough rest before you knead it.",
            "Give the oven a long, generous preheat.",
        ]
        self.rag.add_texts(texts)
        original_count = self.rag.item_count
        original_results = self.rag.search("weigh flour", top_k=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            saved_path = self.rag.save(tmpdir)
            assert (saved_path / "knowledge.json").exists()
            assert (saved_path / "lexical.db").exists()

            from gaussian_rag import GaussianRAG
            loaded_rag = GaussianRAG.load(tmpdir)
            assert loaded_rag.item_count == original_count

            loaded_results = loaded_rag.search("weigh flour", top_k=3)
            assert len(loaded_results) >= 1, "Load + search must return results"

            # Top result texts should overlap
            original_ids = {r.knowledge.id for r in original_results}
            loaded_ids = {r.knowledge.id for r in loaded_results}
            assert original_ids & loaded_ids, (
                "Loaded index should return at least one overlapping top result"
            )

            # Close store connections so Windows releases file handles before tmpdir deletion
            loaded_rag.close()
            self.rag.close()

    def test_save_requires_directory(self):
        """save() raises ValueError when no directory is provided."""
        with pytest.raises(ValueError, match="No save directory"):
            self.rag.save()


# ---------------------------------------------------------------------------
# LIB-04: Standalone independent operation
# ---------------------------------------------------------------------------


class TestStandaloneOperation:
    """LIB-04 — Operates without any cloud tokens or LLM credentials."""

    def test_create_without_cloud_tokens(self):
        """GaussianRAG.create() succeeds with no cloud credentials."""
        from gaussian_rag import GaussianRAG
        rag = GaussianRAG.create(use_clef=False, use_generator=False)
        assert rag is not None
        assert rag._clef_client is None
        assert rag._generator is None

    def test_search_without_cloud_tokens(self):
        """search() works end-to-end with no cloud or LLM required."""
        from gaussian_rag import GaussianRAG
        rag = GaussianRAG.create(use_clef=False)
        rag.add_texts([
            "The 12 Laws of Good Bread cover basics for baking at home.",
            "Law 1 says to weigh your flour instead of scooping it.",
            "Law 3 says salt is not optional.",
        ])
        results = rag.search("How do I bake better bread?", top_k=3)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_query_without_cloud_tokens(self):
        """async query() works locally — returns empty answer without generator."""
        from gaussian_rag import GaussianRAG
        rag = GaussianRAG.create(use_clef=False, use_generator=False)
        rag.add_texts(["Bread is flour, water, salt and time."])

        result = await rag.query("What is bread?", top_k=3)
        assert "question" in result
        assert "retrieved_chunks" in result
        assert "context" in result
        assert "answer" in result
        assert result["question"] == "What is bread?"
        assert result["answer"] == "", "No generator — answer should be empty string"

    def test_repr(self):
        from gaussian_rag import GaussianRAG
        rag = GaussianRAG.create()
        r = repr(rag)
        assert "GaussianRAG" in r
        assert "clef=disabled" in r
        assert "generator=disabled" in r

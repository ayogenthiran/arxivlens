import unittest
from pathlib import Path

from src.data_processing.document_processor import DocumentProcessor


class ExtractYearTests(unittest.TestCase):
    def setUp(self):
        self.processor = DocumentProcessor()

    def test_revised_paper_uses_earliest_version_and_keeps_updated_year(self):
        data = {
            "id": "1601.00001",
            "title": "Old paper",
            "abstract": "Abstract",
            "update_date": "2024-03-01",
            "versions": [
                {"version": "v2", "created": "Fri, 1 Mar 2024 00:00:00 GMT"},
                {"version": "v1", "created": "Mon, 4 Jan 2016 00:00:00 GMT"},
            ],
        }
        doc = self.processor._build_document(data, Path("snapshot.json"))
        self.assertEqual(doc["year"], 2016)
        self.assertEqual(doc["submission_date"], "2016-01-04")
        self.assertEqual(doc["updated_year"], 2024)
        chunks = self.processor.chunk_documents([doc])
        self.assertEqual(chunks[0]["metadata"]["year"], 2016)
        self.assertEqual(chunks[0]["metadata"]["updated_year"], 2024)

    def test_unparseable_versions_fall_back_to_modern_id(self):
        data = {
            "id": "1601.00001",
            "update_date": "2024-03-01",
            "versions": [
                {"version": "v1", "created": "not-a-date"},
                {"version": "v2"},
            ],
        }
        self.assertEqual(self.processor._extract_year(data), 2016)

    def test_missing_versions_uses_modern_arxiv_id(self):
        data = {"id": "2401.02329", "update_date": "2025-01-01"}
        doc = self.processor._build_document(data, Path("snapshot.json"))
        self.assertEqual(doc["year"], 2024)
        self.assertIsNone(doc["submission_date"])
        self.assertEqual(doc["updated_year"], 2025)

    def test_legacy_id_uses_earliest_version_created(self):
        data = {
            "id": "hep-th/9901001",
            "update_date": "2024-01-01",
            "versions": [
                {"version": "v1", "created": "Fri, 1 Jan 1999 00:00:00 GMT"},
            ],
        }
        self.assertEqual(self.processor._extract_year(data), 1999)

    def test_update_date_alone_supplies_year(self):
        data = {"id": "hep-th/9901001", "update_date": "2024-01-01"}
        doc = self.processor._build_document(data, Path("snapshot.json"))
        self.assertEqual(doc["year"], 2024)
        self.assertIsNone(doc["submission_date"])
        self.assertEqual(doc["updated_year"], 2024)

    def test_unparseable_dates_yield_none(self):
        data = {
            "id": "hep-th/9901001",
            "update_date": "not-a-date",
            "versions": [
                {"version": "v1", "created": "not-a-date"},
                {"version": "v2", "created": ""},
            ],
        }
        doc = self.processor._build_document(data, Path("snapshot.json"))
        self.assertIsNone(doc["year"])
        self.assertIsNone(doc["updated_year"])

    def test_long_abstract_stays_one_chunk(self):
        abstract = "Sentence about retrieval. " * 80
        data = {
            "id": "2401.00001",
            "title": "A long abstract",
            "abstract": abstract,
            "update_date": "2024-02-01",
            "versions": [{"version": "v1", "created": "Thu, 1 Feb 2024 00:00:00 GMT"}],
        }
        doc = self.processor._build_document(data, Path("snapshot.json"))
        chunks = self.processor.chunk_documents([doc])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["metadata"]["chunk_id"], "2401.00001-0")
        self.assertEqual(chunks[0]["text"], f"A long abstract\n\n{abstract.strip()}")


if __name__ == "__main__":
    unittest.main()

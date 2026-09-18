import csv
import hashlib
import json
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class RepositoryMetadataTests(unittest.TestCase):
    def test_citation_metadata(self):
        citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
        version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()

        self.assertEqual(citation["cff-version"], "1.2.0")
        self.assertEqual(citation["title"], "NucleoScan")
        self.assertEqual(citation["version"], version)
        self.assertEqual(citation["license"], "MIT")
        self.assertEqual(
            citation["repository-code"],
            "https://github.com/ahiriadil-lab/NucleoScan",
        )
        self.assertEqual(
            citation["authors"][0]["orcid"],
            "https://orcid.org/0000-0001-5170-6538",
        )

    def test_notebook_is_valid_json(self):
        notebook = json.loads((ROOT / "RUN_ON_COLAB.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertIn("cells", notebook)

    def test_bundled_data_checksums(self):
        with (ROOT / "DATA_MANIFEST.tsv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))

        self.assertGreater(len(rows), 0)
        for row in rows:
            path = ROOT / row["path"]
            self.assertTrue(path.is_file(), row["path"])
            self.assertEqual(path.stat().st_size, int(row["bytes"]))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, row["sha256"], row["path"])


if __name__ == "__main__":
    unittest.main()

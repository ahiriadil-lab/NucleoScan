import unittest
from tempfile import NamedTemporaryFile

import numpy as np

from core.score import (
    compute_fragment_scores,
    normalize_vector,
    score_residues_marginal,
)
from nucleoscan import _normalize_sequence, _read_sequence_fasta, _safe_name


class CoreScoringTests(unittest.TestCase):
    def test_normalize_vector_bounds(self):
        result = normalize_vector(np.array([2.0, 4.0, 6.0]))
        np.testing.assert_allclose(result, np.array([0.0, 0.5, 1.0]))

    def test_constant_vector_uses_neutral_value(self):
        result = normalize_vector(np.array([3.0, 3.0]))
        np.testing.assert_allclose(result, np.array([0.5, 0.5]))

    def test_fragment_to_residue_scoring(self):
        metrics = [
            {
                "length": length,
                "rmsd_mean": 1.0 - 0.1 * index,
                "q_mean": 0.05 * index,
                "q_local_mean": 0.10 + 0.05 * index,
                "hydro_rg_mean": 0.8 - 0.05 * index,
                "plddt_mean": 65.0 + 5.0 * index,
                "ss_dominant": "H",
            }
            for index, length in enumerate(range(3, 7))
        ]

        fragment_scores = compute_fragment_scores(metrics)
        residue_scores = score_residues_marginal(fragment_scores, max_length=6)

        self.assertEqual(fragment_scores["length"].tolist(), [3, 4, 5, 6])
        self.assertEqual(residue_scores["residue"].tolist(), [1, 2, 3, 4, 5, 6])
        self.assertTrue(residue_scores["nucleus_score"].between(0.0, 3.0).all())
        self.assertEqual(residue_scores["is_nucleus"].dtype, bool)


class SequenceInputTests(unittest.TestCase):
    def test_normalize_sequence(self):
        self.assertEqual(_normalize_sequence(" acd\nEFG "), "ACDEFG")
        with self.assertRaises(ValueError):
            _normalize_sequence("ACD*")

    def test_single_sequence_fasta(self):
        with NamedTemporaryFile("w", suffix=".fasta") as handle:
            handle.write(">query\nACDE\nFGH\n")
            handle.flush()
            self.assertEqual(_read_sequence_fasta(handle.name), "ACDEFGH")

    def test_safe_name(self):
        self.assertEqual(_safe_name("my protein/1"), "my_protein_1")


if __name__ == "__main__":
    unittest.main()

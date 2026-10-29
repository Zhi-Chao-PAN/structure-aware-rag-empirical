import unittest
import sys
import os
import json
import subprocess
import tempfile
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from evaluation.auto_score import auto_score, normalize_text, extract_numbers


class TestAutoScore(unittest.TestCase):

    def test_normalize_text(self):
        self.assertEqual(normalize_text("  Hello World  "), "hello world")
        self.assertEqual(normalize_text("<think>ignored</think>Answer"), "answer")

    def test_extract_numbers(self):
        text = "The revenue is 10,500.50 and 60.9B"
        # We expect standard number extraction
        expected = ["10500.50", "60.9"]
        self.assertEqual(extract_numbers(text), expected)

    def test_long_unformatted_and_signed_numbers(self):
        self.assertEqual(
            extract_numbers("1000 -123456 +1234.5 1,234"),
            ["1000", "-123456", "+1234.5", "1234"],
        )

    def test_cli_preserves_manual_grade_and_labels_heuristic(self):
        script = Path(__file__).resolve().parents[1] / "src/evaluation/auto_score.py"
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "input.csv", Path(tmp) / "output.csv"
            pd.DataFrame(
                [{"model_answer": "not 100 but 200", "ground_truth": "100", "score": 0}]
            ).to_csv(source, index=False)
            run = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input",
                    str(source),
                    "--output",
                    str(target),
                ],
                capture_output=True,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            result = pd.read_csv(target)
            self.assertEqual(result.loc[0, "score"], 0)
            self.assertEqual(result.loc[0, "review_status"], "required")
            self.assertIn("heuristic_score", result.columns)
            source.write_text("wrong,columns\n1,2\n")
            run = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input",
                    str(source),
                    "--output",
                    str(target),
                ],
                capture_output=True,
            )
            self.assertEqual(run.returncode, 2)

    def test_numeric_scoring_exact(self):
        row = pd.Series(
            {
                "model_answer": "The net income was 1,234 million.",
                "ground_truth": "1,234",
            }
        )
        self.assertEqual(auto_score(row), 1.0)

    def test_numeric_scoring_partial(self):
        row = pd.Series({"model_answer": "It was 100.", "ground_truth": "100 and 200"})
        self.assertEqual(auto_score(row), 0.5)

    def test_text_scoring_exact(self):
        row = pd.Series(
            {"model_answer": "The answer is Yes absolutely.", "ground_truth": "Yes"}
        )
        self.assertEqual(auto_score(row), 1.0)

    def test_text_scoring_negative(self):
        # 'No' should not match 'Not' loosely without boundaries,
        # but our current logic uses token set intersection which handles word boundaries.
        row = pd.Series({"model_answer": "It is NOT supported.", "ground_truth": "No"})
        # "not" vs "no" -> 0.0 match intersection
        self.assertEqual(auto_score(row), 0.0)

    def test_text_scoring_wrong(self):
        row = pd.Series({"model_answer": "The answer is No.", "ground_truth": "Yes"})
        self.assertEqual(auto_score(row), 0.0)


if __name__ == "__main__":
    unittest.main()

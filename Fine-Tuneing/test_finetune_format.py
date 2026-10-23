"""Tests for the fine-tuning data format. Run from this directory:

    python -m unittest test_finetune_format
"""

import unittest
from pathlib import Path

from finetune_format import SAMPLE_SENTINEL, split_samples


def turn(role: str, text: str) -> str:
    return f"<|im_start|>{role}\n{text}<|im_end|>"


PLAIN = "\n".join([turn("system", "You are a tenant law assistant."), turn("user", "Q1"), turn("assistant", "A1")])
RAG = "\n".join([
    turn("system", "Use the context.\n\nCONTEXT:\nParagraph one.\n\nParagraph two."),
    turn("user", "Q2"),
    turn("assistant", "A2 first paragraph.\n\nA2 second paragraph."),
])
INSTRUCTION = "\n".join([turn("user", "Q3"), turn("assistant", "A3")])
SAMPLES = [PLAIN, RAG, INSTRUCTION]


class SplitSamplesTest(unittest.TestCase):
    def test_sentinel_separated_files_round_trip(self):
        self.assertEqual(split_samples(SAMPLE_SENTINEL.join(SAMPLES)), SAMPLES)

    def test_blank_line_separated_files_split_on_turns_not_paragraphs(self):
        # Older files used "\n\n" between samples; RAG samples also contain "\n\n".
        self.assertEqual(split_samples("\n\n".join(SAMPLES)), SAMPLES)

    def test_tracked_training_files_split_into_every_sample(self):
        here = Path(__file__).parent
        train = split_samples((here / "train.txt").read_text(encoding="utf-8"))
        val = split_samples((here / "val.txt").read_text(encoding="utf-8"))
        self.assertEqual(len(train), 454)
        self.assertEqual(len(val), 50)
        for sample in train + val:
            self.assertIn("<|im_start|>assistant", sample)
            self.assertTrue(sample.endswith("<|im_end|>"))


class FineTuneImportTest(unittest.TestCase):
    def test_importing_the_training_script_has_no_side_effects(self):
        source = (Path(__file__).parent / "FineTune.py").read_text(encoding="utf-8")
        self.assertIn('if __name__ == "__main__":', source)
        module_level = [
            line for line in source.splitlines()
            if line.startswith(("shutil.rmtree", "for d in", "model, tokenizer", "trainer"))
        ]
        self.assertEqual(module_level, [], "training steps must not run at import time")


if __name__ == "__main__":
    unittest.main()

"""Training-data file format shared by prepare_finetune_data.py and FineTune.py."""

import re

# Written between samples. \n\n can't be used: RAG samples contain paragraph
# breaks inside the system message.
SAMPLE_SENTINEL = "\n<<<SAMPLE_BOUNDARY>>>\n"

_TURN = re.compile(r"<\|im_start\|>(system|user|assistant)\n.*?<\|im_end\|>", re.S)


def split_samples(raw: str) -> list[str]:
    """Split a train/val file into ChatML samples.

    Files with the sentinel split on it. Older files separated samples with
    blank lines, which also occur inside samples, so those are split on ChatML
    turns instead: each sample ends with its assistant turn.
    """
    if SAMPLE_SENTINEL.strip() in raw:
        return [s.strip() for s in raw.split(SAMPLE_SENTINEL) if s.strip()]

    samples: list[str] = []
    current: list[str] = []
    for match in _TURN.finditer(raw):
        current.append(match.group(0))
        if match.group(1) == "assistant":
            samples.append("\n".join(current))
            current = []
    if current:
        samples.append("\n".join(current))
    return samples

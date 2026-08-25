# Syllable Detection Python

Estimate syllable counts and split English words into syllable chunks with a sonority-based method.

## Requirements

- Python 3.14 or later
- [uv](https://docs.astral.sh/uv/)

## Setup

Install the project dependencies:

```bash
uv sync
```

## Usage

Run Python in the project environment:

```bash
uv run python
```

Then import the public functions:

```python
from sample_1 import count_syllables, split_syllables, text_to_sonority_features

count = count_syllables("syllable")
chunks = split_syllables("syllable")
features = text_to_sonority_features("syllable")
```

`count_syllables()` returns an estimated syllable count.

`split_syllables()` returns syllable chunks.

`text_to_sonority_features()` returns a fixed-length NumPy feature vector.

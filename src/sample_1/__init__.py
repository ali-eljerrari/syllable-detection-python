import random
from pathlib import Path

from sample_1.main import count_syllables, split_syllables, text_to_sonority_features

WORDLIST_PATH = Path(__file__).resolve().parents[2] / "data" / "words.csv"


def main() -> None:
    with open(WORDLIST_PATH) as f:
        all_words = [line.strip() for line in f if line.strip()]

    words = random.sample(all_words, 100)

    rows = []
    for word in words:
        features = text_to_sonority_features(word)
        syllable_count = count_syllables(word)
        chopped = "-".join(split_syllables(word))
        rows.append((word, syllable_count, chopped, features))

    # Column widths that adapt to the data
    w_word = max(len("Word"), max(len(r[0]) for r in rows))
    w_syl  = max(len("Syl"), 3)
    w_hyph = max(len("Hyphenation"), max(len(r[2]) for r in rows))
    w_feat = max(len("Sonority Features"), 39)  # "[x1 x2 … x10]" is ~39 chars

    def row_line(a, b, c, d, left="│", sep="│", right="│"):
        return f"{left} {a:<{w_word}} {sep} {b:>{w_syl}} {sep} {c:<{w_hyph}} {sep} {d:<{w_feat}} {right}"

    def border(left, mid, right, fill="─"):
        return (
            left
            + fill * (w_word + 2)
            + mid
            + fill * (w_syl + 2)
            + mid
            + fill * (w_hyph + 2)
            + mid
            + fill * (w_feat + 2)
            + right
        )

    print(border("┌", "┬", "┐"))
    print(row_line("Word", "Syl", "Hyphenation", "Sonority Features"))
    print(border("├", "┼", "┤"))
    for word, syllable_count, chopped, features in rows:
        feat_str = "[" + " ".join(f"{v:>2}" for v in features) + "]"
        print(row_line(word, syllable_count, chopped, feat_str))
    print(border("└", "┴", "┘"))

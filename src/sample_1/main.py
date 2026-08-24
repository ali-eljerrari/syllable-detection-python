import numpy as np

# Sonority Hierarchy scores (0 = unmapped, 6 = most sonorous)
SONORITY_HIERARCHY = {
    # Vowels (highest sonority)
    'a': 6, 'e': 6, 'i': 6, 'o': 6, 'u': 6,
    # Glides
    'w': 5, 'y': 5, 'j': 5,
    # Liquids
    'r': 4, 'l': 4,
    # Nasals
    'm': 3, 'n': 3, 'ng': 3,
    # Fricatives (including digraphs)
    's': 2, 'z': 2, 'f': 2, 'v': 2,
    'sh': 2, 'th': 2, 'ch': 2, 'ph': 2, 'wh': 2, 'gh': 2,
    'h': 2,
    # Stops / Plosives (lowest sonority)
    'p': 1, 't': 1, 'k': 1, 'b': 1, 'd': 1, 'g': 1,
    'c': 1, 'q': 1, 'x': 1, 'ck': 1,
    # 'qu' is an onset cluster (stop + glide); the 'u' is part of the onset
    'qu': 1,
}

# Digraphs matched greedily left-to-right (order matters for greedy scan)
_DIGRAPHS = frozenset({'sh', 'th', 'ch', 'ph', 'wh', 'gh', 'ck', 'ng', 'qu'})

_VOWELS = frozenset('aeiou')

# True diphthongs: vowel pairs that always form a single nucleus.
_DIPHTHONGS = frozenset({
    'ai', 'ay', 'au', 'aw',
    'ea', 'ee', 'ei', 'eu', 'ew',
    'oa', 'oe', 'oi', 'oo', 'ou', 'ow', 'oy',
    'ue', 'ui',
})

# Vowel pairs that are ONE nucleus only when preceded by a palatalising
# consonant (t, s, c, x produce a /ʃ/ or /tʃ/ onset that absorbs the
# following 'i' into the onset, e.g. -tion /ʃən/, -sion /ʃən/).
_PALATAL_FUSIONS = frozenset({'io', 'ia', 'ie', 'iu'})
_PALATAL_TRIGGERS = frozenset({'t', 's', 'c', 'x', 'ss', 'sc'})


def _tokenize(text: str) -> list[str]:
    """
    Convert a word into a list of phoneme tokens, collapsing common digraphs
    into single tokens before character-by-character processing.
    """
    text = text.lower().strip()
    tokens: list[str] = []
    i = 0
    while i < len(text):
        bigram = text[i : i + 2]
        if bigram in _DIGRAPHS:
            tokens.append(bigram)
            i += 2
        else:
            tokens.append(text[i])
            i += 1
    return tokens


def _drop_silent_terminal_e(tokens: list[str]) -> list[str]:
    """
    Remove a silent terminal 'e' following the magic-e rule:
    - The word ends in [consonant] + 'e' (optionally with a trailing 's').
    - At least one vowel nucleus exists before the candidate 'e'.

    Handles the common "e + s" pattern (e.g. "triptanes").
    """
    if not tokens:
        return tokens

    suffix: list[str] = []
    t = list(tokens)
    if t and t[-1] == 's':
        suffix = [t.pop()]

    if (
        len(t) >= 2
        and t[-1] == 'e'
        and t[-2] not in _VOWELS
        and any(tok in _VOWELS for tok in t[:-1])
    ):
        t = t[:-1]

    return t + suffix


def _get_active_tokens(tokens: list[str]) -> list[str]:
    """Return tokens with the silent terminal 'e' removed (if present)."""
    return _drop_silent_terminal_e(tokens)


def _is_diphthong(tokens: list[str], i: int) -> bool:
    """
    Return True if tokens[i] and tokens[i+1] should be treated as a single
    syllable nucleus (diphthong or palatal fusion).
    """
    pair = tokens[i] + tokens[i + 1]
    if pair in _DIPHTHONGS:
        return True
    if pair in _PALATAL_FUSIONS:
        # Fuse only when the preceding token is a palatalising consonant
        if i > 0 and tokens[i - 1] in _PALATAL_TRIGGERS:
            return True
    return False


def _find_peaks(tokens: list[str], scores: list[int]) -> list[int]:
    """
    Return token indices that are syllable nuclei.

    Rules:
    1. Only vowel-scored tokens (score == 6) or a word-final 'y' (glide acting
       as vowel /iː/) can be nuclei.
    2. A run of adjacent vowels is consumed greedily: diphthong/palatal pairs
       are fused into a single nucleus; any leftover vowels each become their
       own nucleus.
    3. All other tokens are not nuclei.
    """
    n = len(scores)
    # Build an effective score list where a word-final 'y' counts as a vowel
    eff = list(scores)
    if tokens and tokens[-1] == 'y':
        eff[-1] = 6

    peaks: list[int] = []
    i = 0
    while i < n:
        if eff[i] != 6:
            i += 1
            continue

        # Greedily consume a vowel run, fusing diphthong pairs
        # Check if this vowel fuses with the next
        if i + 1 < n and eff[i + 1] == 6 and _is_diphthong(tokens, i):
            peaks.append(i)  # fused nucleus
            i += 2
            # After a diphthong, if the very next token is also a vowel
            # and forms another diphthong with the last consumed token,
            # keep fusing (handles 'eau': ea+u where 'au' fuses after 'e')
            while i < n and eff[i] == 6:
                prev_idx = i - 1
                if _is_diphthong(tokens, prev_idx):
                    i += 1  # absorb into the same nucleus
                else:
                    break
        else:
            peaks.append(i)
            i += 1

    return peaks


def text_to_sonority_features(text: str, max_length: int = 10) -> np.ndarray:
    """
    Convert a word into a fixed-length numerical sonority feature vector for ML.

    Each element corresponds to one phoneme token (digraphs count as one).
    Silent terminal 'e' tokens are dropped before scoring.
    The vector is zero-padded or truncated to *max_length*.
    """
    tokens = _get_active_tokens(_tokenize(text))
    features = [SONORITY_HIERARCHY.get(t, 0) for t in tokens]

    if len(features) < max_length:
        features.extend([0] * (max_length - len(features)))
    else:
        features = features[:max_length]

    return np.array(features)


def count_syllables(text: str) -> int:
    """
    Estimate syllable count using the Sonority Sequencing Principle.

    Each vowel (treating diphthong pairs as one) in the per-phoneme sonority
    profile is a syllable nucleus.  Silent terminal 'e' is excluded.
    """
    tokens = _get_active_tokens(_tokenize(text))
    scores = [SONORITY_HIERARCHY.get(t, 0) for t in tokens]
    return max(len(_find_peaks(tokens, scores)), 1)


def split_syllables(text: str) -> list[str]:
    """
    Split a word into syllable chunks using the Sonority Sequencing Principle.

    Boundaries are placed at the lowest-sonority valley between adjacent nuclei
    (onset maximisation: the trough token opens the next syllable).
    Silent terminal 'e' is excluded from the nucleus search but remains
    attached to the final syllable in the output.
    """
    raw = text.lower().strip()
    tokens = _tokenize(raw)
    active = _get_active_tokens(tokens)

    scores = [SONORITY_HIERARCHY.get(t, 0) for t in active]
    peaks = _find_peaks(active, scores)

    if len(peaks) <= 1:
        return [raw]

    # Build a character-start index for every token in *raw*
    tok_char_start: list[int] = []
    pos = 0
    for tok in tokens:
        tok_char_start.append(pos)
        pos += len(tok)

    # Between each pair of adjacent peaks find the inter-syllable boundary
    boundaries: list[int] = []
    for p1, p2 in zip(peaks, peaks[1:]):
        valley = list(range(p1 + 1, p2))
        if not valley:
            boundaries.append(p2)
            continue
        min_score = min(scores[i] for i in valley)
        # Onset maximisation: last trough position opens the next syllable
        boundary = max(i for i in valley if scores[i] == min_score)
        boundaries.append(boundary)

    # Convert token-index boundaries → character-index boundaries in *raw*
    char_boundaries = [tok_char_start[b] for b in boundaries]

    parts: list[str] = []
    prev = 0
    for cb in char_boundaries:
        parts.append(raw[prev:cb])
        prev = cb
    parts.append(raw[prev:])

    return parts

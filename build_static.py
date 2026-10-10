"""Build the static (no server) version of the game into ./site.

Needs numberbatch_demo.py, static_page.html and the vectors file in the same folder.

Usage:
    python build_static.py                         # 500 puzzles drawn from the secret-word tier
    python build_static.py --count 1000
    python build_static.py --secrets my_words.txt  # hand-picked secrets, one word per line

Preview (the page must be served, it will not work opened as a file):
    python -m http.server 8001 --directory site
    then open http://127.0.0.1:8001

Output:
    site/index.html              the page
    site/data/manifest.json      vocabulary size, puzzle count, heat bins, hint rules, data version
    site/data/words.txt          every guessable word, one per line; line number = word index
    site/data/puzzles/0000.bin   one file per puzzle (see write_puzzle)
"""

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np

from numberbatch_demo import HEAT_EDGES, HEAT_LABELS, HINT_MIN_ZIPF, VOCAB_SIZE, load_vectors, read_vocab, secret_pool

SITE = Path("site")
PAGE_SOURCE = Path(__file__).with_name("static_page.html")
SHUFFLE_SEED = 0  # fixed, so rebuilding with the same inputs gives the same puzzle numbers


def choose_secrets(words: list[str], index: dict[str, int], count: int, secrets_file: Path | None) -> list[str]:
    if secrets_file:
        wanted = [line.strip().lower() for line in secrets_file.read_text(encoding="utf8").splitlines() if line.strip()]
        missing = [w for w in wanted if w not in index]
        if missing:
            print(f"Skipping {len(missing)} words that are not in the vocabulary: {', '.join(missing[:10])}")
        secrets = list(dict.fromkeys(w for w in wanted if w in index))  # drop repeats, keep order
    else:
        pool = secret_pool()
        secrets = random.Random(SHUFFLE_SEED).sample(pool, min(count, len(pool)))
    # Shuffle so a puzzle's number says nothing about its word.
    random.Random(SHUFFLE_SEED).shuffle(secrets)
    return secrets


def write_puzzle(path: Path, matrix: np.ndarray, target: int) -> None:
    """Write one puzzle: all word indexes closest-first, then their scores in the same order.

    Layout (little-endian): N unsigned 16-bit word indexes, then N signed 16-bit scores.
    Entry 0 is the answer itself. A stored score of 2214 means 22.14.
    """
    similarities = matrix @ matrix[target]
    similarities[target] = 2.0  # guarantee the answer sorts first, even against a duplicate vector
    order = np.argsort(-similarities, kind="stable")
    similarities[target] = 1.0
    scores = np.round(similarities[order] * 10_000)
    path.write_bytes(order.astype("<u2").tobytes() + scores.astype("<i2").tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=500, help="how many puzzles to generate from the secret pool")
    parser.add_argument("--secrets", type=Path, help="text file of hand-picked secret words; overrides --count")
    args = parser.parse_args()

    words, matrix = load_vectors(VOCAB_SIZE)
    words = [str(w) for w in words]
    if len(words) > 65_535:
        raise SystemExit("The puzzle files store word indexes in 16 bits, so the vocabulary must stay under 65,536.")
    index = {word: i for i, word in enumerate(words)}

    secrets = choose_secrets(words, index, args.count, args.secrets)
    if not secrets:
        raise SystemExit("No secret words to build puzzles from.")

    data = SITE / "data"
    if data.exists():
        shutil.rmtree(data)  # clear puzzles from an earlier build
    puzzles = data / "puzzles"
    puzzles.mkdir(parents=True)

    for number, secret in enumerate(secrets):
        write_puzzle(puzzles / f"{number:04d}.bin", matrix, index[secret])

    (data / "words.txt").write_text("\n".join(words), encoding="utf8", newline="\n")

    lows = [2] + [edge + 1 for edge in HEAT_EDGES]
    highs = HEAT_EDGES + [len(words)]
    # The version changes whenever words, secrets or bins change, so the page drops stale saved games.
    # Words are in frequency order, so "may be shown as a hint" is: among the first hintCount
    # words, and not one of the hidden (offensive) ones.
    vocab = read_vocab()[: len(words)]
    hint_count = sum(1 for _, zipf, _ in vocab if zipf >= HINT_MIN_ZIPF)
    hidden = [i for i, (_, _, offensive) in enumerate(vocab) if offensive]
    fingerprint = json.dumps([words, secrets, HEAT_EDGES, HEAT_LABELS, hint_count, hidden]).encode("utf8")
    manifest = {
        "version": hashlib.sha1(fingerprint).hexdigest()[:8],
        "total": len(words),
        "puzzles": len(secrets),
        "hintCount": hint_count,
        "hidden": hidden,
        "heats": [{"label": label, "from": low, "to": high} for label, low, high in zip(HEAT_LABELS, lows, highs)],
    }
    (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    shutil.copyfile(PAGE_SOURCE, SITE / "index.html")

    size_mb = sum(f.stat().st_size for f in SITE.rglob("*") if f.is_file()) / 1_000_000
    print(f"Wrote {len(secrets)} puzzles for {len(words):,} words to {SITE}/ ({size_mb:.0f} MB).")
    print("Preview: python -m http.server 8001 --directory site")


if __name__ == "__main__":
    main()
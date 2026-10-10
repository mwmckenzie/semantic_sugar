"""Cemantle-style guessing demo on ConceptNet Numberbatch vectors.

Setup:
    pip install numpy wordfreq nltk better-profanity
    Download numberbatch-en-19.08.txt.gz into this folder:
    https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz
    python audit_vocab.py      # builds the word list, vocab_kept.csv

Rerun audit_vocab.py whenever you edit vocab_keep.txt or vocab_drop.txt.

Usage:
    python numberbatch_demo.py                  # random puzzle
    python numberbatch_demo.py --code friday1   # same code = same puzzle for everyone
    python numberbatch_demo.py --neighbors cold # print a word's nearest neighbors and exit
"""

import argparse
import csv
import gzip
import hashlib
import random
from bisect import bisect_left
from functools import lru_cache
from pathlib import Path

import numpy as np

VECTORS_FILE = Path("numberbatch-en-19.08.txt.gz")
VOCAB_FILE = Path("vocab_kept.csv")   # written by audit_vocab.py, most frequent word first
VOCAB_SIZE = 60_000          # upper limit on guessable words; the cleaned list is about 50,500

# Three tiers, by Zipf frequency (3 = once per million words).
# Every word is guessable. Only fairly common, inoffensive words are shown as hints or
# in the give-up list. Only very common, inoffensive words can be the secret.
HINT_MIN_ZIPF = 2.5
SECRET_MIN_ZIPF = 4.0
MIN_SECRET_LENGTH = 4
TOP_COUNT = 10               # size of the "best guesses" list

# Heat bins on displayed rank (answer = 1). Each edge is the last rank inside its bin;
# anything past the final edge is "frozen solid".
HEAT_LABELS = ["on fire", "scorching", "hot", "warm", "lukewarm", "cool", "chilly", "cold", "freezing", "frozen solid"]
HEAT_EDGES = [10, 25, 60, 250, 600, 1_500, 3_000, 10_000, 25_000]


@lru_cache(maxsize=1)
def read_vocab() -> list[tuple[str, float, bool]]:
    """The cleaned word list as (word, Zipf frequency, offensive), most frequent first."""
    if not VOCAB_FILE.exists():
        raise SystemExit(f"Missing {VOCAB_FILE} - build it first with: python audit_vocab.py")
    with open(VOCAB_FILE, newline="", encoding="utf8") as f:
        rows = [(row["word"], float(row["zipf"]), row["offensive"] == "yes") for row in csv.DictReader(f)]
    return rows[:VOCAB_SIZE]


def load_vectors(vocab_size: int = VOCAB_SIZE) -> tuple[list[str], np.ndarray]:
    """Return (words in vocabulary order, unit-length mean-centered matrix)."""
    words = [word for word, _, _ in read_vocab()][:vocab_size]
    fingerprint = hashlib.sha1("\n".join(words).encode("utf8")).hexdigest()[:8]
    cache = Path(f"numberbatch_vocab_{fingerprint}.npz")  # a new word list gets a new cache file
    if cache.exists():
        return words, np.load(cache)["matrix"]

    if not VECTORS_FILE.exists():
        raise SystemExit(f"Missing {VECTORS_FILE} - see the download link at the top of this script.")

    print(f"Reading {VECTORS_FILE} (first run with this word list only, takes a minute)...")
    wanted = set(words)
    found: dict[str, np.ndarray] = {}
    with gzip.open(VECTORS_FILE, "rt", encoding="utf8") as f:
        next(f)  # header line: "<term count> <dimensions>"
        for line in f:
            term, _, values = line.partition(" ")
            if term in wanted:
                found[term] = np.array(values.split(), dtype=np.float32)

    missing = [word for word in words if word not in found]
    if missing:
        raise SystemExit(f"{len(missing)} words in {VOCAB_FILE} have no vector, e.g. {', '.join(missing[:5])}. "
                         "Rerun audit_vocab.py against this vectors file.")
    matrix = np.stack([found[word] for word in words])

    # Mean-centering removes the direction all words share, which spreads the scores out.
    matrix -= matrix.mean(axis=0)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

    np.savez_compressed(cache, matrix=matrix)
    return words, matrix


def hintable_mask() -> np.ndarray:
    """True for each word that may be shown to the player as a hint or in the give-up list."""
    return np.array([zipf >= HINT_MIN_ZIPF and not offensive for _, zipf, offensive in read_vocab()])


def secret_pool() -> list[str]:
    """Words that may be a puzzle's answer."""
    return [
        word for word, zipf, offensive in read_vocab()
        if zipf >= SECRET_MIN_ZIPF and not offensive and len(word) >= MIN_SECRET_LENGTH
    ]


def pick_secret(words: list[str], code: str | None) -> str:
    """Pick the secret word for a puzzle code. code=None gives a fresh random puzzle."""
    return random.Random(code).choice(secret_pool())


def rank_all(matrix: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (similarity of every word to the target, rank of every word; target = 0)."""
    similarities = matrix @ matrix[target]
    order = np.argsort(-similarities)
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(len(order))
    return similarities, ranks


def hint_index(by_rank: np.ndarray, hintable: np.ndarray, best_rank: int) -> int | None:
    """Word index for a hint: about halfway between the best guess so far and the answer.

    by_rank is every word index ordered closest-first (the answer at 0). best_rank is the
    0-based rank of the best guess so far, or len(by_rank) before any guess. Returns None
    when no showable word is closer than the best guess.
    """
    closer = by_rank[1:best_rank]
    closer = closer[hintable[closer]]
    if len(closer) == 0:
        return None
    halfway = max(1, best_rank // 2)
    position_of = {int(word): position for position, word in enumerate(by_rank[1:best_rank], start=1)}
    within = [int(word) for word in closer if position_of[int(word)] <= halfway]
    return within[-1] if within else int(closer[0])


def closest_showable(by_rank: np.ndarray, hintable: np.ndarray, count: int) -> np.ndarray:
    """The answer's nearest neighbors that may be shown, closest first."""
    near = by_rank[1:]
    return near[hintable[near]][:count]


def heat_label(rank: int) -> str:
    """Heat bin for a displayed rank (answer = 1)."""
    return HEAT_LABELS[min(bisect_left(HEAT_EDGES, rank), len(HEAT_LABELS) - 1)]


def show_neighbors(words: list[str], matrix: np.ndarray, index: dict[str, int], word: str, count: int) -> None:
    if word not in index:
        raise SystemExit(f"'{word}' is not in the vocabulary.")
    similarities, _ = rank_all(matrix, index[word])
    for rank, i in enumerate(np.argsort(-similarities)[1 : count + 1], start=1):
        print(f"{rank:>4}  {similarities[i] * 100:6.2f}  {words[i]}")


def play(words: list[str], matrix: np.ndarray, index: dict[str, int], code: str | None, secret: str | None) -> None:
    if secret is None:
        secret = pick_secret(words, code)
    elif secret not in index:
        raise SystemExit(f"'{secret}' is not in the vocabulary.")

    similarities, ranks = rank_all(matrix, index[secret])
    by_rank = np.argsort(ranks)
    hintable = hintable_mask()
    guesses: dict[str, tuple[int, int]] = {}  # word -> (guess number, displayed rank)
    best_rank = len(words)

    print(f"Guess the secret word ({len(words):,} words in play).")
    print("Commands: /hint  /give  /quit\n")

    while True:
        guess = input("> ").strip().lower()
        if not guess:
            continue
        if guess == "/quit":
            return
        if guess == "/give":
            print(f"The word was '{secret}'. Its nearest neighbors:")
            for i in closest_showable(by_rank, hintable, 10):
                print(f"    {similarities[i] * 100:6.2f}  {words[i]}")
            return
        if guess == "/hint":
            hint = hint_index(by_rank, hintable, best_rank)
            if hint is None:
                print("  no closer hint left")
                continue
            guess = words[hint]
            print(f"Hint: {guess}")
        if guess not in index:
            print("  not in the vocabulary")
            continue

        if guess in guesses:
            number, rank = guesses[guess]
            print(f"  already guessed (guess {number}: {heat_label(rank)}, rank {rank:,}) - not counted")
            continue

        i = index[guess]
        rank = int(ranks[i]) + 1  # displayed rank: the answer is #1, its nearest neighbor #2
        number = len(guesses) + 1
        guesses[guess] = (number, rank)
        best_rank = min(best_rank, rank - 1)

        if rank == 1:
            print(f"\nSolved in {number} guesses - the word was '{secret}'.")
            return

        print(f"  guess {number}: {heat_label(rank)}   score {similarities[i] * 100:6.2f}   rank {rank:,} of {len(words):,}")
        ordered = sorted(guesses.items(), key=lambda item: item[1][1])
        top = ordered[:TOP_COUNT]
        place = next(p for p, (w, _) in enumerate(top, start=1) if w == guess) if rank <= top[-1][1][1] else None
        # The first TOP_COUNT guesses always make the list, so only announce once one gets bumped.
        if place and len(ordered) > TOP_COUNT:
            print(f"  ** new top {TOP_COUNT} entry at #{place} - bumps '{ordered[TOP_COUNT][0]}' **")
        print(f"  top {TOP_COUNT}:")
        width = max(len(w) for w, _ in top)
        for p, (w, (_, r)) in enumerate(top, start=1):
            print(f"    {p:>2}. {w:<{width}}  {heat_label(r):<12}  rank {r:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--code", help="puzzle code; the same code always picks the same secret word")
    parser.add_argument("--secret", help="force a specific secret word (for testing)")
    parser.add_argument("--neighbors", metavar="WORD", help="print a word's nearest neighbors and exit")
    parser.add_argument("--count", type=int, default=30, help="how many neighbors to print")
    args = parser.parse_args()

    words, matrix = load_vectors(VOCAB_SIZE)
    index = {word: i for i, word in enumerate(words)}

    if args.neighbors:
        show_neighbors(words, matrix, index, args.neighbors.lower(), args.count)
    else:
        play(words, matrix, index, args.code, args.secret)


if __name__ == "__main__":
    main()
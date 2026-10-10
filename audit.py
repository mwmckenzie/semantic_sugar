"""Audit the vocabulary filters before changing the game's word list.

This only reports. It does not change the game or any of its files.

Setup:
    pip install nltk wordfreq better-profanity     (better-profanity is optional)

Usage:
    python audit_vocab.py

Reads numberbatch-en-19.08.txt.gz from this folder, walks every English entry, applies
each filter in turn, and prints what each one removes. It then ranks the survivors by
frequency so you can choose where to cut the list.

Optional files, one word per line (lines starting with # are ignored):
    vocab_drop.txt   words to remove even though they pass the filters
    vocab_keep.txt   words to keep even though a filter would remove them

Writes two files for browsing:
    vocab_kept.csv      rank, word, frequency, offensive flag, note
    vocab_removed.csv   word, frequency, reason, detail
"""

import csv
import gzip
from collections import Counter
from pathlib import Path

import nltk
from wordfreq import zipf_frequency

VECTORS_FILE = Path("numberbatch-en-19.08.txt.gz")
TERMS_CACHE = Path("numberbatch_terms.txt")
DROP_FILE = Path("vocab_drop.txt")
KEEP_FILE = Path("vocab_keep.txt")

MIN_LENGTH = 2
SAMPLES = 24                                        # examples printed per filter
CUTOFFS = [4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0]       # Zipf frequency floors to compare
MARKS = [5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000]

# Reasons, in the order the filters run.
SHAPE = "not a plain single word"
DROPPED = "on the drop list"
STOP = "stop word"
PROPER = "proper noun"
INFLECTED = "inflected form"
UNKNOWN = "not in the dictionary"
FORM_ONLY = "only a form of another word"
NO_FREQUENCY = "no frequency data"
REASONS = [SHAPE, DROPPED, STOP, PROPER, INFLECTED, UNKNOWN, FORM_ONLY, NO_FREQUENCY]

# Notes on kept words.
ING_FORM = "-ing form"
KEEP_LISTED = "keep list"


def read_terms() -> list[str]:
    """Every English entry in Numberbatch (the first token on each line)."""
    if TERMS_CACHE.exists():
        return TERMS_CACHE.read_text(encoding="utf8").split("\n")
    if not VECTORS_FILE.exists():
        raise SystemExit(f"Missing {VECTORS_FILE} - it should sit in this folder.")
    print(f"Reading entries from {VECTORS_FILE} (first run only, takes a minute)...")
    with gzip.open(VECTORS_FILE, "rt", encoding="utf8") as f:
        next(f)  # header line
        terms = [line.partition(" ")[0] for line in f]
    TERMS_CACHE.write_text("\n".join(terms), encoding="utf8", newline="\n")
    return terms


def read_word_file(path: Path) -> set[str]:
    if not path.exists():
        return set()
    lines = (line.strip().lower() for line in path.read_text(encoding="utf8").splitlines())
    return {line for line in lines if line and not line.startswith("#")}


def load_wordnet():
    """Return (wordnet, lowercase dictionary words, words that only appear capitalized, stop words)."""
    nltk.download("wordnet", quiet=True)
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords, wordnet

    lowercase, capitalized = set(), set()
    for synset in wordnet.all_synsets():
        for name in synset.lemma_names():
            if name.isalpha():  # skips phrases, hyphenated entries and anything with digits
                (lowercase if name.islower() else capitalized).add(name.lower())
    stops = {w for w in stopwords.words("english") if w.isalpha()}
    return wordnet, lowercase, capitalized - lowercase, stops


def load_offensive_check():
    try:
        from better_profanity import profanity
    except ImportError:
        return None
    profanity.load_censor_words()
    return profanity.contains_profanity


def base_forms(wordnet, word: str, parts_of_speech: str = "nva") -> set[str]:
    """Other dictionary words this word is a form of: 'cats' -> {'cat'}, 'saw' -> {'see'}."""
    bases = set()
    for part_of_speech in parts_of_speech:
        bases.update(wordnet._morphy(word, part_of_speech))
    bases.discard(word)
    return bases


def stands_alone(wordnet, word: str) -> bool:
    """True if the word is a dictionary entry in some part of speech where it is not an inflection.

    'thought' is a noun in its own right, so it stays even though it is also a form of 'think'.
    'days' is only ever the plural of 'day', so it goes.
    """
    for part_of_speech in "nvar":  # noun, verb, adjective, adverb
        forms = set(wordnet._morphy(word, part_of_speech))
        if forms == {word}:
            return True
    return False


def frequency(word: str) -> float:
    return zipf_frequency(word, "en")


def classify(word, wordnet, lowercase, proper, stops, drop, keep) -> tuple[str | None, str]:
    """Return (reason for removal or None, detail). For kept words the detail is a note."""
    if not (word.isascii() and word.isalpha() and len(word) >= MIN_LENGTH):
        return SHAPE, ""
    if word in keep:
        return None, KEEP_LISTED
    if word in drop:
        return DROPPED, ""
    if word in stops:
        return STOP, ""
    if word not in lowercase:
        if word in proper:
            return PROPER, ""
        # -ing forms stay: 'running' and 'thinking' mean something different from 'run' and 'think'.
        if word.endswith("ing") and base_forms(wordnet, word, "v"):
            return (None, ING_FORM) if frequency(word) > 0 else (NO_FREQUENCY, "")
        bases = base_forms(wordnet, word)
        if bases:
            return INFLECTED, " ".join(sorted(bases))
        return UNKNOWN, ""
    if not stands_alone(wordnet, word):
        return FORM_ONLY, " ".join(sorted(base_forms(wordnet, word)))
    if frequency(word) == 0:
        return NO_FREQUENCY, ""
    return None, ""


def show(words: list[str]) -> None:
    line = "    "
    for word in words:
        if len(line) + len(word) > 100:
            print(line.rstrip())
            line = "    "
        line += word + "  "
    if line.strip():
        print(line.rstrip())


def main() -> None:
    terms = read_terms()
    wordnet, lowercase, proper, stops = load_wordnet()
    is_offensive = load_offensive_check()
    drop, keep = read_word_file(DROP_FILE), read_word_file(KEEP_FILE)

    kept: list[tuple[str, float, bool, str]] = []       # word, frequency, offensive, note
    removed: list[tuple[str, float, str, str]] = []     # word, frequency, reason, detail
    for word in dict.fromkeys(terms):
        reason, detail = classify(word, wordnet, lowercase, proper, stops, drop, keep)
        if reason is None:
            kept.append((word, frequency(word), bool(is_offensive and is_offensive(word)), detail))
        else:
            # Phrases and symbols have no useful single-word frequency.
            removed.append((word, 0.0 if reason == SHAPE else frequency(word), reason, detail))

    kept.sort(key=lambda item: (-item[1], item[0]))
    removed.sort(key=lambda item: (-item[1], item[0]))
    counts = Counter(reason for _, _, reason, _ in removed)

    print(f"\nNumberbatch English entries: {len(terms):,}")
    print(f"Drop list: {len(drop):,} words.  Keep list: {len(keep):,} words "
          f"({len(keep - set(terms)):,} of them are not in Numberbatch, so they cannot be used).\n")
    print("Removed by each filter, in the order they run:")
    for reason in REASONS:
        print(f"  {counts[reason]:>8,}  {reason}")
    print(f"  {len(kept):>8,}  kept")
    print(f"\nOf the kept words, {sum(1 for k in kept if k[3] == ING_FORM):,} are -ing forms kept by the -ing rule.")
    if is_offensive:
        print(f"{sum(1 for k in kept if k[2]):,} kept words are flagged offensive. They stay guessable, "
              "and will be left out of secrets and hints.")
    else:
        print("The offensive-word flag was skipped: pip install better-profanity to include it.")

    print(f"\nThe {SAMPLES} most frequent words each filter removes:")
    for reason in REASONS[1:]:
        examples = [(w, d) for w, _, r, d in removed if r == reason][:SAMPLES]
        if not examples:
            continue
        print(f"\n  {reason}")
        with_base = reason in (INFLECTED, FORM_ONLY)
        show([f"{w}>{d.split()[0]}" if with_base and d else w for w, d in examples])

    print("\nHow many words survive at each frequency floor (Zipf scale; 3 = once per million words):")
    for cutoff in CUTOFFS:
        print(f"  Zipf >= {cutoff:<4} {sum(1 for k in kept if k[1] >= cutoff):>8,}")
    print(f"  any frequency  {len(kept):>7,}")

    print("\nWhat the list looks like at different depths (12 words starting at each rank):")
    for mark in MARKS:
        if mark >= len(kept):
            break
        print(f"\n  rank {mark:,}  (Zipf {kept[mark][1]:.2f})")
        show([k[0] for k in kept[mark : mark + 12]])

    with open("vocab_kept.csv", "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "word", "zipf", "offensive", "note"])
        writer.writerows(
            (rank, word, f"{z:.2f}", "yes" if flagged else "", note)
            for rank, (word, z, flagged, note) in enumerate(kept, start=1)
        )
    with open("vocab_removed.csv", "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "zipf", "reason", "detail"])
        writer.writerows((w, f"{z:.2f}", r, d) for w, z, r, d in removed if r != SHAPE)
    print("\nWrote vocab_kept.csv and vocab_removed.csv.")


if __name__ == "__main__":
    main()
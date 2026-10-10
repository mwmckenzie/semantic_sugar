"""Web version of the Numberbatch guessing game.

Keep this file, index.html and numberbatch_demo.py in the same folder as the vectors file.

Setup:
    pip install fastapi uvicorn      (plus numpy and wordfreq from before)

Run:
    uvicorn server:app --reload
    then open http://127.0.0.1:8000

The server keeps no per-player state. A puzzle code decides the secret word, and the
browser keeps the guess list, so anyone with the same code plays the same puzzle.
"""

import secrets
from bisect import bisect_left
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from numberbatch_demo import HEAT_EDGES, HEAT_LABELS, VOCAB_SIZE, load_vectors, pick_secret, rank_all

PAGE = Path(__file__).with_name("index.html")
NEIGHBORS_ON_REVEAL = 10

app = FastAPI(title="Warmer")
WORDS, MATRIX = load_vectors(VOCAB_SIZE)
INDEX = {word: i for i, word in enumerate(WORDS)}


class CodeRequest(BaseModel):
    code: str


class GuessRequest(BaseModel):
    code: str
    word: str


class HintRequest(BaseModel):
    code: str
    best_rank: int | None = None  # best displayed rank so far; None before the first guess


@lru_cache(maxsize=64)
def game(code: str) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """Return (secret, similarities, ranks, word indexes ordered by rank) for a puzzle code."""
    secret = pick_secret(WORDS, code)
    similarities, ranks = rank_all(MATRIX, INDEX[secret])
    return secret, similarities, ranks, np.argsort(ranks)


def heat_index(rank: int) -> int:
    return min(bisect_left(HEAT_EDGES, rank), len(HEAT_LABELS) - 1)


@app.get("/")
def page() -> FileResponse:
    return FileResponse(PAGE)


@app.get("/api/info")
def info() -> dict:
    """Vocabulary size and the heat scale, so the page never hard-codes the bins."""
    lows = [2] + [edge + 1 for edge in HEAT_EDGES]
    highs = HEAT_EDGES + [len(WORDS)]
    heats = [{"label": label, "from": low, "to": high} for label, low, high in zip(HEAT_LABELS, lows, highs)]
    return {"total": len(WORDS), "heats": heats}


@app.get("/api/new")
def new_code() -> dict:
    return {"code": secrets.token_hex(3)}


@app.post("/api/guess")
def guess(request: GuessRequest) -> dict:
    word = request.word.strip().lower()
    if word not in INDEX:
        return {"word": word, "known": False}

    _, similarities, ranks, _ = game(request.code)
    i = INDEX[word]
    rank = int(ranks[i]) + 1  # displayed rank: the answer is #1, its nearest neighbor #2
    return {
        "word": word,
        "known": True,
        "rank": rank,
        "score": round(float(similarities[i]) * 100, 2),
        "heat": heat_index(rank),
        "solved": rank == 1,
    }


@app.post("/api/hint")
def hint(request: HintRequest) -> dict:
    """Offer a word halfway between the best guess so far and the answer."""
    _, _, _, by_rank = game(request.code)
    best = (request.best_rank or len(WORDS)) - 1
    return {"word": str(WORDS[by_rank[max(1, best // 2)]])}


@app.post("/api/reveal")
def reveal(request: CodeRequest) -> dict:
    secret, similarities, _, by_rank = game(request.code)
    neighbors = [
        {"word": str(WORDS[i]), "score": round(float(similarities[i]) * 100, 2)}
        for i in by_rank[1 : NEIGHBORS_ON_REVEAL + 1]
    ]
    return {"secret": secret, "neighbors": neighbors}
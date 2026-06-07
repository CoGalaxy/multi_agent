from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RagDocument:
    id: str
    title: str
    text: str
    source: str | None = None


@dataclass(frozen=True)
class RagHit:
    document: RagDocument
    score: float

    def evidence_line(self) -> str:
        source = self.document.source or self.document.id
        snippet = self.document.text.strip().replace("\n", " ")
        if len(snippet) > 220:
            snippet = snippet[:217].rstrip() + "..."
        return f"rag:{self.document.id} score={self.score:.3f} source={source} :: {snippet}"


class SimpleRagRetriever:
    def __init__(self, corpus_path: Path, top_k: int = 4) -> None:
        self.corpus_path = corpus_path
        self.top_k = top_k
        self.documents = _load_documents(corpus_path)

    @property
    def available(self) -> bool:
        return bool(self.documents)

    def search(self, query: str) -> list[RagHit]:
        query_terms = _terms(query)
        if not query_terms:
            return []
        hits: list[RagHit] = []
        for document in self.documents:
            doc_terms = _terms(f"{document.title} {document.text}")
            overlap = query_terms & doc_terms
            if not overlap:
                continue
            score = len(overlap) / max(len(query_terms), 1)
            hits.append(RagHit(document=document, score=score))
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[: self.top_k]


def _load_documents(path: Path) -> list[RagDocument]:
    if not path.exists():
        return []
    documents: list[RagDocument] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            data = json.loads(line)
            documents.append(
                RagDocument(
                    id=str(data.get("id") or index),
                    title=str(data.get("title") or data.get("id") or index),
                    text=str(data.get("text") or ""),
                    source=data.get("source"),
                )
            )
    return documents


def _terms(text: str) -> set[str]:
    lowered = text.lower()
    latin_terms = set(re.findall(r"[a-z0-9_]{2,}", lowered))
    cjk_terms = set(re.findall(r"[\u4e00-\u9fff]{2,}", lowered))
    cjk_chars = {char for char in lowered if "\u4e00" <= char <= "\u9fff"}
    return latin_terms | cjk_terms | cjk_chars

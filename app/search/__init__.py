"""Search abstractions and the bounded authorized-corpus provider."""

from app.search.authorized_corpus import (
    DEFAULT_DEMO_CORPUS_PATH,
    AuthorizedCorpusRecord,
    AuthorizedCorpusSearchProvider,
    CorpusValidationError,
)
from app.search.interface import (
    ContentType,
    SearchCandidate,
    SearchProvider,
    SearchQuery,
)

__all__ = [
    "DEFAULT_DEMO_CORPUS_PATH",
    "AuthorizedCorpusRecord",
    "AuthorizedCorpusSearchProvider",
    "ContentType",
    "CorpusValidationError",
    "SearchCandidate",
    "SearchProvider",
    "SearchQuery",
]

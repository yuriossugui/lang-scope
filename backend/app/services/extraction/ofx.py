"""OFX (bank statement) extractor.

Prefers the ``ofxparse`` library when available and falls back to a small
regex-based parser over the raw SGML-like OFX text.
"""

import logging
import re
from typing import List, Optional

from app.services.extraction.base import DocumentExtractor
from app.models.document import DocumentChunk

logger = logging.getLogger("app")


def _tag(block: str, name: str) -> str:
    # OFX is SGML-like: tags are frequently unclosed (<TAG>value<next>).
    # Capture everything up to the next tag start.
    match = re.search(rf"<{name}>([^<]*)", block, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _try_ofxparse(file_path: str) -> Optional[List[DocumentChunk]]:
    try:
        import ofxparse as ofxparse_module  # type: ignore
        parse_fn = getattr(ofxparse_module, "parse", None)
        if parse_fn is None:
            return None
    except Exception:
        return None
    chunks: List[DocumentChunk] = []
    try:
        with open(file_path, "rb") as handle:
            ofx = parse_fn(handle)
    except Exception as exc:
        logger.warning("ofxparse failed for %s: %s", file_path, exc)
        return None
    for account in getattr(ofx, "accounts", []) or []:
        account_id = getattr(account, "account_id", "unknown")
        statement = getattr(account, "statement", None)
        for transaction in getattr(statement, "transactions", []) or []:
            content = (
                f"Transação {getattr(transaction, 'id', '')}: "
                f"{getattr(transaction, 'date', '')} "
                f"{getattr(transaction, 'payee', '')} "
                f"{getattr(transaction, 'amount', '')} "
                f"{getattr(transaction, 'type', '')}"
            )
            chunks.append(
                DocumentChunk(
                    content=content,
                    metadata={
                        "record": str(getattr(transaction, "id", "")),
                        "source_type": "record",
                        "account": account_id,
                    },
                )
            )
    return chunks


def _regex_fallback(file_path: str) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
        text = handle.read()
    blocks = re.split(r"<STMTTRN>", text)
    record = 0
    for block in blocks[1:]:
        record += 1
        fitid = _tag(block, "FITID") or str(record)
        content = (
            f"Transação {fitid}: "
            f"{_tag(block, 'DTPOSTED')} {_tag(block, 'PAYEE')} "
            f"{_tag(block, 'TRNAMT')} {_tag(block, 'TRNTYPE')}"
        )
        chunks.append(
            DocumentChunk(
                content=content,
                metadata={
                    "record": fitid,
                    "source_type": "record",
                },
            )
        )
    return chunks


class OFXExtractor(DocumentExtractor):
    def extract(self, file_path: str) -> List[DocumentChunk]:
        chunks = _try_ofxparse(file_path)
        if chunks is not None:
            return chunks
        return _regex_fallback(file_path)

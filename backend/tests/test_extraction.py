"""Tests for document content extraction (PDF, CSV, OFX)."""

import io
from unittest.mock import MagicMock, patch

import pytest

from app.services.extraction.csv import CSVExtractor
from app.services.extraction.ofx import OFXExtractor
from app.services.extraction.pdf import PDFExtractor


def test_csv_extractor(tmp_path):
    file_path = tmp_path / "dados.csv"
    file_path.write_text(
        "nome,valor,data\nJoao,100,2024-01-01\nMaria,200,2024-01-02\n",
        encoding="utf-8",
    )
    chunks = CSVExtractor().extract(str(file_path))
    assert len(chunks) == 2
    assert chunks[0].metadata["source_type"] == "line"
    assert "Joao" in chunks[0].content
    assert chunks[0].metadata["line"] == 2


def test_ofx_extractor_regex_fallback(tmp_path):
    file_path = tmp_path / "extrato.ofx"
    file_path.write_text(
        "<STMTTRN><TRNTYPE>DEBIT<DTPOSTED>20240101<FITID>1<PAYEE>Supermercado"
        "<TRNAMT>-50.00</STMTTRN><STMTTRN><TRNTYPE>CREDIT<DTPOSTED>20240102"
        "<FITID>2<PAYEE>Salario<TRNAMT>2000.00</STMTTRN>",
        encoding="utf-8",
    )
    chunks = OFXExtractor().extract(str(file_path))
    assert len(chunks) == 2
    assert chunks[0].metadata["source_type"] == "record"
    assert "Supermercado" in chunks[0].content
    assert chunks[1].metadata["record"] == "2"


@patch("app.services.extraction.pdf.PdfReader")
def test_pdf_extractor(mock_reader):
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Conteudo da pagina um."
    mock_reader.return_value.pages = [mock_page]
    chunks = PDFExtractor().extract("fake.pdf")
    assert len(chunks) == 1
    assert chunks[0].metadata["page"] == 1
    assert chunks[0].metadata["source_type"] == "page"
    assert "Conteudo da pagina um" in chunks[0].content

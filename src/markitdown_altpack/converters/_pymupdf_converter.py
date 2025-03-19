# markitdown-altpack
# Copyright (c) 2025 Arpan Biswas
# Licensed under the MIT License – see the LICENSE file in the root of the repository.

import typing as T

import pymupdf
import pymupdf4llm
from markitdown import DocumentConverter, DocumentConverterResult, StreamInfo

PYMUPDF_MIME_TYPE_PREFIXES = [
    "application/pdf",
    "application/x-pdf",
]

PYMUPDF_FILE_EXTENSIONS = [
    ".pdf",
]


class PyMuPDFConverter(DocumentConverter):
    def accepts(
        self,
        file_stream: T.BinaryIO,
        stream_info: StreamInfo,
        **kwargs: T.Any,  # Options to pass to the converter
    ) -> bool:
        mimetype = (stream_info.mimetype or "").lower()
        extension = (stream_info.extension or "").lower()

        if extension in PYMUPDF_FILE_EXTENSIONS:
            return True

        for prefix in PYMUPDF_MIME_TYPE_PREFIXES:
            if mimetype.startswith(prefix):
                return True

        return False

    def convert(
        self,
        file_stream: T.BinaryIO,
        stream_info: StreamInfo,
        **kwargs: T.Any,  # Options to pass to the converter
    ) -> DocumentConverterResult:
        doc = pymupdf.open(stream=file_stream.read())
        markdown_text = pymupdf4llm.to_markdown(doc)

        return DocumentConverterResult(markdown=markdown_text)

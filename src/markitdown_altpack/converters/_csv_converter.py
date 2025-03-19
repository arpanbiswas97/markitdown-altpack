# markitdown-altpack
# Copyright (c) 2025 Arpan Biswas
# Licensed under the MIT License – see the LICENSE file in the root of the repository.

import typing as T

import pandas as pd
from markitdown import DocumentConverter, DocumentConverterResult, StreamInfo

CSV_MIME_TYPE_PREFIXES = [
    "text/csv",
]
CSV_FILE_EXTENSIONS = [".csv"]


class CsvConverter(DocumentConverter):
    """Converts CSV files to markdown tables"""

    def accepts(
        self,
        file_stream: T.BinaryIO,
        stream_info: StreamInfo,
        **kwargs: T.Any,  # Options to pass to the converter
    ) -> bool:
        mimetype = (stream_info.mimetype or "").lower()
        extension = (stream_info.extension or "").lower()

        if extension in CSV_FILE_EXTENSIONS:
            return True

        for prefix in CSV_MIME_TYPE_PREFIXES:
            if mimetype.startswith(prefix):
                return True

    def convert(
        self,
        file_stream: T.BinaryIO,
        stream_info: StreamInfo,
        **kwargs: T.Any,  # Options to pass to the converter
    ) -> DocumentConverterResult:
        df = pd.read_csv(file_stream)
        markdown_text = df.fillna("").to_markdown(index=False)

        return DocumentConverterResult(markdown=markdown_text)

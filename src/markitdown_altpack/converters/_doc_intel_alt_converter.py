# markitdown-altpack
# Copyright (c) 2025 Arpan Biswas
# Licensed under the MIT License – see the LICENSE file in the root of the repository.

import os
import re
import typing as T
from io import StringIO

import pandas as pd
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, AnalyzeResult
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from markitdown import DocumentConverter, DocumentConverterResult, StreamInfo

DI_MIME_TYPE_PREFIXES = [
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml",
    "application/xhtml",
    "text/html",
    "application/pdf",
    "application/x-pdf",
    "image/",
]

# Note: Removed XLSX support from DI since XlsxAltConverter works better

DI_FILE_EXTENSIONS = [
    ".docx",
    ".pptx",
    ".html",
    ".htm",
    ".pdf",
    ".jpeg",
    ".jpg",
    ".png",
    ".bmp",
    ".tiff",
    ".heif",
]


class DocumentIntelligenceAltConverter(DocumentConverter):
    """Alternate Document Intelligence Converter that
    - Retains page metadata
    - Does not enable a few optional DI features
    - Converts html tables to markdown
    - Removed XLSX support from DI since XlsxAltConverter works better
    """

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str = None,
        api_version: str = "2024-07-31-preview",
    ):
        super().__init__()
        if endpoint is None:
            raise ValueError("No endpoint found for Azure Data Intelligence.")
        if api_key is None:
            # Try to fetch Document Intelligence API Key from environment
            api_key = os.getenv("DI_KEY", None)
        if api_key is None:
            api_key_obj = DefaultAzureCredential()
        else:
            api_key_obj = AzureKeyCredential(api_key)
        self._endpoint = endpoint
        self._api_version = api_version
        self._api_key = api_key_obj
        self._document_analysis_client = DocumentIntelligenceClient(
            endpoint=self._endpoint,
            credential=self._api_key,
            api_version=self._api_version,
        )

    def accepts(
        self,
        file_stream: T.BinaryIO,
        stream_info: StreamInfo,
        **kwargs: T.Any,  # Options to pass to the converter
    ) -> bool:
        mimetype = (stream_info.mimetype or "").lower()
        extension = (stream_info.extension or "").lower()

        if extension in DI_FILE_EXTENSIONS:
            return True

        for prefix in DI_MIME_TYPE_PREFIXES:
            if mimetype.startswith(prefix):
                return True

        return False

    def _convert_html_table_to_markdown(self, html_table: str) -> str:
        df = pd.read_html(StringIO(html_table), flavor="lxml")[0]
        return df.fillna("").to_markdown(index=False)

    def convert(
        self,
        file_stream: T.BinaryIO,
        stream_info: StreamInfo,
        **kwargs: T.Any,  # Options to pass to the converter
    ) -> DocumentConverterResult:
        poller = self._document_analysis_client.begin_analyze_document(
            model_id="prebuilt-layout",
            body=AnalyzeDocumentRequest(bytes_source=file_stream.read()),
            output_content_format="markdown",
        )
        result: AnalyzeResult = poller.result()

        # Convert all HTML tables to markdown
        table_regex = re.compile(
            r"<table\b[^>]*>(.*?)<\/table>", re.DOTALL | re.IGNORECASE
        )

        def _replace_table(match):
            html_table = match.group(0)
            try:
                return self._convert_html_table_to_markdown(html_table)
            except Exception as e:
                print(f"Error converting table into markdown - {e}")
                return html_table

        markdown_text = table_regex.sub(_replace_table, result.content)

        return DocumentConverterResult(markdown=markdown_text)

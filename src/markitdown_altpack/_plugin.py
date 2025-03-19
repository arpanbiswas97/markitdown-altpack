# markitdown-altpack
# Copyright (c) 2025 Arpan Biswas
# Licensed under the MIT License – see the LICENSE file in the root of the repository.

import os

from markitdown import MarkItDown

from markitdown_altpack.converters import (
    CsvConverter,
    DocumentIntelligenceAltConverter,
    PyMuPDFConverter,
    XlsAltConverter,
    XlsxAltConverter,
)

__plugin_interface_version__ = (
    1  # The version of the plugin interface that this plugin uses
)


def register_converters(markitdown: MarkItDown, **kwargs):
    """
    Called during construction of MarkItDown instances to register converters provided by plugins.
    """

    markitdown.register_converter(PyMuPDFConverter())
    markitdown.register_converter(CsvConverter())
    markitdown.register_converter(XlsAltConverter())
    markitdown.register_converter(XlsxAltConverter())

    docintel_endpoint = kwargs.get("docintel_endpoint", None)
    if docintel_endpoint is None:
        # Try to fetch Document Intelligence Endpoint from environment
        docintel_endpoint = os.getenv("DI_ENDPOINT", None)
    if docintel_endpoint is not None:
        markitdown.register_converter(
            DocumentIntelligenceAltConverter(endpoint=docintel_endpoint)
        )

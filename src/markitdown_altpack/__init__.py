# markitdown-altpack
# Copyright (c) 2025 Arpan Biswas
# Licensed under the MIT License – see the LICENSE file in the root of the repository.

from .__about__ import __version__
from ._plugin import (
    DocumentIntelligenceAltConverter,
    PyMuPDFConverter,
    __plugin_interface_version__,
    register_converters,
)

__all__ = [
    "__version__",
    "__plugin_interface_version__",
    "register_converters",
    "DocumentIntelligenceAltConverter",
    "PyMuPDFConverter"
]

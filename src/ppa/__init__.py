"""Unofficial reimplementation of Probabilistic Prompt Adaptation (PPA).

Hara & Otsuka, "Probabilistic Prompt Adaptation for Unified Image Aesthetics
and Quality Assessment", CVPR 2026. A faithful reproduction of the method only.
"""

from .config import Config
from .inference import PPAInferenceEngine
from .model import PPA

__version__ = "0.1.0"
__all__ = ["Config", "PPA", "PPAInferenceEngine"]


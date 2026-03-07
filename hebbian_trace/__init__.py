"""Hebbian Trace Memory: persistent cross-session memory for frozen LLMs."""

from .model import HebbianTraceModule, GPT2WithTrace

__all__ = ["HebbianTraceModule", "GPT2WithTrace"]

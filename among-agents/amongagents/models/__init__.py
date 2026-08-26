"""Model backends used by AmongAgents."""

from .local_transformers import LocalTransformersRuntime, is_local_model

__all__ = ["LocalTransformersRuntime", "is_local_model"]

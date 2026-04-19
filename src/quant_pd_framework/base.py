"""Abstract base class shared by all pipeline steps."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .context import PipelineContext


class BasePipelineStep(ABC):
    """Every quant-modeling stage inherits from this interface."""

    name = "base_step"

    def __call__(self, context: PipelineContext) -> PipelineContext:
        context.log(f"Starting step: {self.name}")
        updated_context = self.run(context)
        updated_context.log(f"Finished step: {self.name}")
        return updated_context

    @abstractmethod
    def run(self, context: PipelineContext) -> PipelineContext:
        """Performs the step-specific work and returns the updated context."""

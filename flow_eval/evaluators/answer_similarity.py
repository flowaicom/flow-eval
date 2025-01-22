import logging
from typing import Literal

import numpy as np
from sentence_transformers import SentenceTransformer

from flow_eval.eval_data_types import EvalInput, EvalOutput
from flow_eval.evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)


class AnswerSimilarityEvaluator(BaseEvaluator):
    """Answer similarity evaluator using embedding-based similarity.

    Example:
        evaluator = AnswerSimilarityEvaluator("all-MiniLM-L6-v2")

        # The strings to compare will be extracted
        # from EvalInput.output and EvalInput.expected_output
        result = evaluator.evaluate(eval_input)
    """

    def __init__(
        self,
        model_name: str,
        distance_fn: Literal["cosine", "euclidean", "dot"] = "cosine",
        normalize_embeddings: bool = True,
        output_dir: str | None = "output/",
    ):
        """Initialize EmbeddingEvaluator.

        Args:
            model_name: Name of the sentence-transformers model to use
            distance_fn: Distance function to use for similarity
            normalize_embeddings: Whether to normalize embeddings before comparison
            output_dir: Directory to save evaluation results
        """
        super().__init__(output_dir)
        self.model_name = model_name
        self.distance_fn = distance_fn
        self.normalize_embeddings = normalize_embeddings
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        # Get embeddings
        emb1 = self.model.encode(text1)
        emb2 = self.model.encode(text2)

        # Normalize if needed
        if self.normalize_embeddings:
            emb1 = emb1 / np.linalg.norm(emb1)
            emb2 = emb2 / np.linalg.norm(emb2)

        # Compute similarity
        if self.distance_fn == "cosine":
            return float(np.dot(emb1, emb2))
        elif self.distance_fn == "euclidean":
            return float(1 / (1 + np.linalg.norm(emb1 - emb2)))
        else:  # dot product
            return float(np.dot(emb1, emb2))

    def _extract_texts(self, eval_input: EvalInput) -> tuple[str, str]:
        """Extract the two texts to compare from EvalInput.

        Always compares output with expected_output. Both must contain exactly one value.
        """
        if not eval_input.output:
            raise ValueError("EvalInput.output is empty")
        if not eval_input.expected_output:
            raise ValueError("EvalInput.expected_output is empty")

        if len(eval_input.output) != 1:
            raise ValueError(
                f"EvalInput.output must contain exactly one value, got" f" {len(eval_input.output)}"
            )
        if len(eval_input.expected_output) != 1:
            raise ValueError(
                f"EvalInput.expected_output must contain exactly"
                f"one value, got {len(eval_input.expected_output)}"
            )

        text1 = next(iter(eval_input.output.values()))
        text2 = next(iter(eval_input.expected_output.values()))
        return text1, text2

    def _save_results(
        self, eval_inputs: list[EvalInput], eval_outputs: list[EvalOutput], append: bool = False
    ):
        """Save results to disk."""
        super()._save_results(
            eval_inputs,
            eval_outputs,
            {"model": self.model_name, "distance_fn": self.distance_fn},
            "embedding_similarity",
            append=append,
        )

    def evaluate(self, eval_input: EvalInput, save_results: bool = False) -> EvalOutput:
        """Evaluate a single EvalInput object."""
        try:
            # Extract texts to compare
            text1, text2 = self._extract_texts(eval_input)

            # Compute similarity
            similarity = self._compute_similarity(text1, text2)

            # Create evaluation output
            eval_output = EvalOutput(score=similarity)

            if save_results:
                self._save_results([eval_input], [eval_output])

            return eval_output
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    def batch_evaluate(
        self,
        eval_inputs: list[EvalInput],
        save_results: bool = True,
    ) -> list[EvalOutput]:
        """Batch evaluate a list of EvalInput objects."""
        try:
            eval_outputs = []
            for eval_input in eval_inputs:
                eval_output = self.evaluate(eval_input, save_results=False)
                eval_outputs.append(eval_output)

            if save_results:
                self._save_results(eval_inputs, eval_outputs)

            return eval_outputs
        except Exception as e:
            logger.error(f"Batch evaluation failed: {e}")
            raise

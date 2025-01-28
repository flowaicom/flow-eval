import logging
from typing import Literal

from sentence_transformers import SentenceTransformer, SimilarityFunction

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
        model_name: str = "all-MiniLM-L6-v2",
        model_type: Literal["sentence-transformer"] = "sentence-transformer",
        eval_name: str = "embedding_similarity",
        similarity_fn_name: Literal["cosine", "dot", "euclidean", "manhattan"] = "cosine",
        output_dir: str | None = "output/",
    ):
        """Initialize EmbeddingEvaluator."""
        super().__init__(output_dir)
        self.model_name = model_name
        self.model_type = model_type
        self.eval_name = eval_name
        self.similarity_fn_name = getattr(SimilarityFunction, similarity_fn_name.upper())
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy initialization of the model."""
        if self._model is None:
            self._model = SentenceTransformer(
                self.model_name, similarity_fn_name=self.similarity_fn_name
            )
        return self._model

    def _check_text_lengths(self, text1: str, text2: str) -> None:
        """Check if texts length exceeds model's maximum sequence length."""
        max_length = self.model.get_max_seq_length()

        for text, source in [(text1, "output"), (text2, "expected_output")]:
            # Use encode to get the token count with special tokens
            tokens = self.model.tokenizer.encode(text)
            tokenized_length = len(tokens)

            if tokenized_length > max_length:
                logger.warning(
                    f"{source} length ({tokenized_length} tokens) exceeds model's "
                    f"maximum sequence length ({max_length} tokens). Text will be truncated. "
                    "Consider using a model with longer sequence length for better results. "
                    "See available models at: https://www.sbert.net/docs/pretrained_models.html"
                )

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts using sentence-transformers similarity function."""
        # Check text lengths
        self._check_text_lengths(text1, text2)

        # Convert texts to embeddings
        emb1 = self.model.encode(text1)
        emb2 = self.model.encode(text2)

        # Use sentence-transformers built-in similarity function
        return float(self.model.similarity(emb1, emb2))

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
        self,
        eval_inputs: list[EvalInput],
        eval_outputs: list[EvalOutput],
        append: bool = False,
    ):
        """Save results to disk with embedding-specific metadata."""
        super()._save_results(
            eval_inputs,
            eval_outputs,
            {
                "model_id": self.model_name,
                "model_type": self.model_type,
                "similarity_fn_name": self.similarity_fn_name,
            },
            self.eval_name,
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

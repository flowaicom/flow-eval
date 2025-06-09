import logging
from typing import TYPE_CHECKING, Literal, Optional

from flow_eval.core import BaseEvaluator, EvalInput, EvalOutput

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnswerSimilarityEvaluator(BaseEvaluator):
    """Answer similarity evaluator using embedding-based similarity.

    It compares the output and expected output of an evaluation input.

    Example:
        evaluator = AnswerSimilarityEvaluator("all-mpnet-base-v2")

        # The strings to compare will be extracted
        # from EvalInput.output and EvalInput.expected_output
        result = evaluator.evaluate(eval_input)
    """

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
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
        self.similarity_fn_name_str = similarity_fn_name
        self._model: Optional["SentenceTransformer"] = None
        self._similarity_fn = None

    @property
    def similarity_fn_name(self):
        """Lazy initialization of similarity function."""
        if self._similarity_fn is None:
            try:
                from sentence_transformers import SimilarityFunction
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for AnswerSimilarityEvaluator. "
                    "Install it with: pip install flow-eval[similarity]"
                ) from e
            self._similarity_fn = getattr(SimilarityFunction, self.similarity_fn_name_str.upper())
        return self._similarity_fn

    @property
    def model(self) -> "SentenceTransformer":
        """Lazy initialization of the model."""
        if self._model is None:
            try:
                import torch
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "torch and sentence-transformers are required for AnswerSimilarityEvaluator. "
                    "Install them with: pip install flow-eval[similarity]"
                ) from e

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(
                self.model_name, similarity_fn_name=self.similarity_fn_name, device=device
            )
            logger.info(f"Model loaded on device: {device}")
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

        return next(iter(eval_input.output.values())), next(
            iter(eval_input.expected_output.values())
        )

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

    def _compute_batch_similarity(self, texts1: list[str], texts2: list[str]) -> list[float]:
        """Compute similarities between two lists of texts using batch processing."""
        # Check text lengths
        for text1, text2 in zip(texts1, texts2, strict=True):
            self._check_text_lengths(text1, text2)

        # Batch encode all texts
        emb1 = self.model.encode(texts1)
        emb2 = self.model.encode(texts2)

        # Compute similarities
        return [float(self.model.similarity(e1, e2)) for e1, e2 in zip(emb1, emb2, strict=True)]

    def evaluate(self, eval_input: EvalInput, save_results: bool = False) -> EvalOutput:
        """Evaluate a single EvalInput object."""
        try:
            # Use batch_evaluate for single input to avoid code duplication
            return self.batch_evaluate([eval_input], save_results)[0]
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    def batch_evaluate(
        self,
        eval_inputs: list[EvalInput],
        save_results: bool = True,
    ) -> list[EvalOutput]:
        """Batch evaluate a list of EvalInput objects."""
        if not eval_inputs:
            raise ValueError("Input list cannot be empty")

        try:
            # Extract all texts to compare and convert tuples to lists
            outputs, expected_outputs = map(
                list,
                zip(*[self._extract_texts(eval_input) for eval_input in eval_inputs], strict=True),
            )

            # Compute similarities in batch
            similarities = self._compute_batch_similarity(outputs, expected_outputs)
            eval_outputs = [EvalOutput(score=score) for score in similarities]

            if save_results:
                self._save_results(eval_inputs, eval_outputs)

            return eval_outputs
        except Exception as e:
            logger.error(f"Batch evaluation failed: {e}")
            raise

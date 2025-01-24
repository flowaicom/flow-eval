import logging
import re

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Module level storage for parsers
_PARSERS = {}


class EvalInput(BaseModel):
    """Input model for an Evaluator."""

    inputs: list[dict[str, str]] = Field(default_factory=list)
    output: dict[str, str]
    expected_output: dict[str, str]


class EvalOutput(BaseModel):
    """Output model for evaluation results supporting multiple score types."""

    feedback: str | None = Field(None, description="Optional feedback from the evaluation")
    score: int | float | bool | str | None = Field(
        None, description="Evaluation result (numeric, boolean, or categorical)"
    )

    @classmethod
    def register_parser(cls, name: str):
        """Decorator to register metric-specific parsers."""

        def decorator(func):
            _PARSERS[name] = func
            return func

        return decorator

    @classmethod
    def parse(
        cls, response: str, parser_name: str = "default", fail_on_parse_error: bool = False
    ) -> "EvalOutput":
        """Parse evaluation response using the specified parser."""
        parser = _PARSERS.get(parser_name)
        if not parser:
            raise ValueError(f"Parser '{parser_name}' not registered")

        try:
            return parser(response)
        except Exception as e:
            if fail_on_parse_error:
                raise ValueError(f"Parse error: {e}") from e
            logger.warning(f"Parse failed for response: {response}. Error: {e}")
            return cls(feedback="Error", score=None)


# FIXME: Move to parsers
@EvalOutput.register_parser("default")
def default_parser(response: str) -> EvalOutput:
    """Default parser for numeric scores with optional feedback."""
    score_match = re.search(r"<score>\s*([\d\.]+)\s*</score>", response)
    feedback_match = re.search(r"<feedback>\s*(.*?)\s*</feedback>", response, re.DOTALL)

    return EvalOutput(
        feedback=feedback_match.group(1).strip() if feedback_match else None,
        score=int(float(score_match.group(1))) if score_match else None,
    )


@EvalOutput.register_parser("pairwise")
def pairwise_parser(response: str) -> EvalOutput:
    """Parser for pairwise comparison results choosing between options A/B."""
    choice_match = re.search(r"<choice>\s*([AB])\s*</choice>", response, re.IGNORECASE)
    feedback_match = re.search(r"<feedback>\s*(.*?)\s*</feedback>", response, re.DOTALL)

    if not choice_match:
        raise ValueError("No valid choice (A/B) found in response")

    choice = choice_match.group(1).upper()

    return EvalOutput(
        feedback=feedback_match.group(1).strip() if feedback_match else None, score=choice
    )


@EvalOutput.register_parser("pairwise_bool")
def pairwise_bool_parser(response: str) -> EvalOutput:
    """Parser that converts A/B choices to boolean (A=True, B=False)."""
    choice_match = re.search(r"<choice>\s*([AB])\s*</choice>", response, re.IGNORECASE)
    feedback_match = re.search(r"<feedback>\s*(.*?)\s*</feedback>", response, re.DOTALL)

    if not choice_match:
        raise ValueError("No valid choice (A/B) found in response")

    choice = choice_match.group(1).upper()

    return EvalOutput(
        feedback=feedback_match.group(1).strip() if feedback_match else None,
        score=choice == "A",  # Returns True for A, False for B
    )

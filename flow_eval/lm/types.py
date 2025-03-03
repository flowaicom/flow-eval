from pydantic import BaseModel, Field

from flow_eval.core import EvalInput


class RubricItem(BaseModel):
    """Represents an item in the scoring rubric."""

    score: int
    description: str


class LMEval(BaseModel):
    """Specification for LM-based evaluation."""

    name: str
    input_columns: list[str] | None
    output_column: str
    expected_output_column: str | None = None
    criteria: str = Field(..., description="The evaluation criteria.")
    rubric: list[RubricItem] = Field(..., description="Scoring rubric for the evaluation.")

    def validate_input(self, eval_input: EvalInput) -> None:
        """Validate the input matches requirements."""
        # Validate required inputs
        if self.input_columns:
            for required_input in self.input_columns:
                found = False
                for input_dict in eval_input.inputs:
                    if required_input in input_dict:
                        found = True
                        break
                if not found:
                    raise ValueError(f"Required input '{required_input}' not found in EvalInput")

        # Validate required output
        if self.output_column not in eval_input.output:
            raise ValueError(f"Required output '{self.output_column}' not found in EvalInput")

        # Validate expected output if specified
        if self.expected_output_column and self.expected_output_column not in eval_input.output:
            raise ValueError(
                f"Required expected output '{self.expected_output_column}' not found in EvalInput"
            )

    def keys(self) -> dict[str, list[str] | str | None]:
        """Get the configured input and output columns.

        Returns:
            dict: A dictionary containing input_columns, output_column, and expected_output_column
        """
        return {
            "input_columns": self.input_columns,
            "output_column": self.output_column,
            "expected_output_column": self.expected_output_column,
        }

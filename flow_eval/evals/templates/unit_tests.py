"""Preset function evaluations."""

import re

from flow_eval.eval_data_types import EvalOutput


def exact_match_fn(response: str, reference: str) -> EvalOutput:
    """Check if response exactly matches reference."""
    matches = reference.strip().lower() == response.strip().lower()
    return EvalOutput(score=True if matches else False)


def email_validation_fn(response: str) -> EvalOutput:
    """Check if response matches email address pattern."""
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"

    matches = bool(re.match(pattern, response))
    return EvalOutput(score=True if matches else False)


def length_ratio_fn(response: str, reference: str) -> EvalOutput:
    """Calculate length ratio between response and reference."""
    ref_len = len(reference.split())
    resp_len = len(response.split())
    ratio = min(resp_len / ref_len if ref_len > 0 else 0.0, 1.0)

    return EvalOutput(score=ratio)


def keyword_presence_fn(response: str) -> EvalOutput:
    """Check presence of required keywords in response."""
    keywords = ["email", "phone", "address", "name"]
    tokenized_response = response.lower().split()

    found = [k for k in keywords if k in tokenized_response]

    if len(found) == 0:
        return EvalOutput(score=False)

    return EvalOutput(score=True)

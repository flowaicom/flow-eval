
# Tests for Flow Judge

This directory contains the test suite for the Flow Eval project.

## Test Coverage

Below is the current test coverage visualization for the Flow Eval project:

FIXME
<p align="center">
  <a href="https://codecov.io/gh/flowaicom/flow-eval" target="_blank">
    <img src="https://codecov.io/gh/flowaicom/flow-eval/branch/main/graphs/sunburst.svg?token=AEGC7W3DGE" alt="Codecov Sunburst Graph">
  </a>
</p>

## Running Tests

To run the entire test suite:
```sh
pytest
```
To run a specific test file:
```sh
pytest tests/unit/test_flow_eval.py
```
To run tests with coverage report:
```sh
pytest --cov=flow_eval --cov-report=term-missing
```

## Contributing

When adding new features or modifying existing ones, please make sure to add or update the corresponding tests. This helps maintain the project's reliability and makes it easier to catch potential issues early.

## Continuous Integration

Our CI pipeline automatically runs these tests on every pull request and push to the main branch. You can check the status of the latest runs in the GitHub Actions tab of the repository.

{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  pythonOlder,
  # build dependencies
  setuptools,
  setuptools-scm,
  wheel,
  # runtime dependencies
  pydantic,
  requests,
  hf-transfer,
  ipykernel,
  ipywidgets,
  tqdm,
  structlog,
  pyyaml,
  torch,
  sentence-transformers,
  openai ? null,
  aiohttp ? null,
  tenacity ? null,
}:
buildPythonPackage rec {
  pname = "flow-eval";
  version = "0.1.0";
  pyproject = true;

  disabled = pythonOlder "3.10";

  src = ./.;

  nativeBuildInputs = [
    setuptools
    setuptools-scm
    wheel
  ];

  propagatedBuildInputs =
    [
      pydantic
      requests
      hf-transfer
      ipykernel
      ipywidgets
      tqdm
      structlog
      aiohttp
      openai
      tenacity
      pyyaml
      torch
      sentence-transformers
    ];

  pythonImportsCheck = [
    "flow_eval"
    "flow_eval.core"
    "flow_eval.integrations"
    "flow_eval.lm"
  ];

  # Relax version constraints on dependencies if needed
  pythonRelaxDeps = [
    "pydantic"
    "requests"
    "hf-transfer"
    "ipykernel"
    "ipywidgets"
    "tqdm"
    "structlog"
    "torch"
  ];

  meta = with lib; {
    description = "Flow AI evals engine";
    homepage = "https://github.com/flowaicom/flow-eval";
    license = licenses.asl20;
    maintainers = with maintainers; [];
  };
}

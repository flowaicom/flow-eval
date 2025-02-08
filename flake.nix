{
  description = "Flow AI evaluation framework";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    flake-parts.inputs.nixpkgs.follows = "nixpkgs";
  };
  outputs = inputs @ {
    self,
    nixpkgs,
    flake-parts,
    ...
  }:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = ["x86_64-linux"];
      flake = {
        _module.args.self = self;
      };
      imports = [];
      perSystem = {
        system,
        self',
        ...
      }: let
        pkgs = import nixpkgs {
          system = system;
          config = {
            allowUnfree = true;
            allowBroken = true;
            allowUnfreePredicate = pkg: true;
            acceptLicense = true;
          };
        };
	ppkgs = pkgs.python311Packages;
        flow-eval = pkgs.callPackage ./default.nix {
          inherit (pkgs) lib fetchFromGitHub;
	  inherit (ppkgs)
	  buildPythonPackage
          pythonOlder
          setuptools
          setuptools-scm
          wheel
          pydantic
          requests
          hf-transfer
          ipykernel
          ipywidgets
          tqdm
          structlog
          openai
          aiohttp
	  pyyaml
	  torch
	  sentence-transformers
          tenacity;
        };
      in {
        packages = {
          flow-eval = flow-eval;
        };
        devShells = {
          default = pkgs.mkShell {
            name = "flow-eval-devshell";
            buildInputs = [
              pkgs.python311
              pkgs.python311Packages.setuptools
              pkgs.python311Packages.setuptools-scm
              pkgs.python311Packages.wheel
              pkgs.ruff
              pkgs.black
              pkgs.isort
              pkgs.pytest
            ];
          };
        };
      };
    };
}

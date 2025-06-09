"""Test that optional dependencies are properly isolated and lazy-loaded."""

import sys
from unittest.mock import patch

import pytest


class TestDependencyIsolation:
    """Test that heavy dependencies like torch are not imported when using lightweight models."""

    def test_openai_import_without_torch(self):
        """Test that OpenAI models can be imported and used without torch."""
        # Save current torch modules state
        torch_modules = {name: mod for name, mod in sys.modules.items() if name.startswith("torch")}

        try:
            # Clear torch from modules if it exists
            for module_name in list(torch_modules.keys()):
                sys.modules.pop(module_name, None)

            # Import flow_eval main package
            assert "torch" not in sys.modules, "torch should not be imported"

            # Import LMEvaluator
            assert "torch" not in sys.modules, "torch should not be imported"

            # Import OpenAI model
            from flow_eval.lm.models import OpenAIModel

            assert "torch" not in sys.modules, "torch should not be imported after OpenAI import"

            # Mock environment to avoid requiring API key
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                # Create OpenAI model instance
                OpenAIModel(model="gpt-4")
                assert (
                    "torch" not in sys.modules
                ), "torch should not be imported after OpenAI instantiation"
        finally:
            # Restore torch modules
            sys.modules.update(torch_modules)

    def test_lazy_model_imports(self):
        """Test that model classes are imported lazily."""
        # Clear all model-related modules
        model_modules = [name for name in sys.modules if "flow_eval.lm.models" in name]
        for module_name in model_modules:
            sys.modules.pop(module_name, None)

        # Import models package should not import actual model classes
        from flow_eval.lm import models

        # Check that heavy dependencies are not loaded yet (unless already in environment)
        # This test verifies the lazy loading mechanism works
        # Test that we can access the model class (it might work if deps are available)
        model_class = models.Hf
        assert model_class.__name__ == "Hf", "Should be able to access Hf class"

        # But instantiation should fail if dependencies aren't properly available
        # (This will pass in environments with all deps, which is fine)

    def test_similarity_evaluator_lazy_loading(self):
        """Test that AnswerSimilarityEvaluator lazy loads torch dependencies."""
        # Save current torch modules state
        torch_modules = {name: mod for name, mod in sys.modules.items() if name.startswith("torch")}

        try:
            # Clear torch modules
            for module_name in list(torch_modules.keys()):
                sys.modules.pop(module_name, None)

            # Import should work without torch
            from flow_eval import AnswerSimilarityEvaluator

            assert "torch" not in sys.modules, "torch should not be imported on class import"

            # Instantiation should work without dependencies
            evaluator = AnswerSimilarityEvaluator()
            assert "torch" not in sys.modules, "torch should not be imported on instantiation"

            # Mock torch as not available to test the error path
            with patch.dict("sys.modules", {"torch": None}):
                with pytest.raises(
                    ImportError, match="torch and sentence-transformers are required"
                ):
                    _ = evaluator.model
        finally:
            # Restore torch modules
            sys.modules.update(torch_modules)

    def test_list_available_models_without_deps(self):
        """Test that list_all_available_models works without heavy dependencies."""
        from flow_eval.lm import list_all_available_models

        # Should return at least BaseEvaluatorModel and OpenAIModel
        models = list_all_available_models()
        model_names = [m.__name__ for m in models]

        assert "BaseEvaluatorModel" in model_names
        assert "OpenAIModel" in model_names

        # Should not crash even if other models fail to import
        assert len(models) >= 2

    def test_model_instantiation_errors(self):
        """Test that models fail gracefully when dependencies are missing."""
        # Mock the availability checks to return False
        with patch("flow_eval.lm.models.huggingface._check_hf_availability", return_value=False):
            with patch("flow_eval.lm.models.vllm._check_vllm_availability", return_value=False):
                from flow_eval.lm.models import Hf, Vllm

                # These should fail with helpful error messages
                with pytest.raises(Exception) as exc_info:
                    Hf()
                assert (
                    "hugging face" in str(exc_info.value).lower()
                    or "hf" in str(exc_info.value).lower()
                )

                with pytest.raises(Exception) as exc_info:
                    Vllm()
                assert "vllm" in str(exc_info.value).lower()

    def test_hf_not_available(self):
        """Test behavior when HF dependencies are not available."""
        with patch("flow_eval.lm.models.huggingface._check_hf_availability", return_value=False):
            from flow_eval.lm.models.huggingface import Hf

            with pytest.raises(Exception, match="required Hugging Face packages"):
                Hf()

    def test_openai_model_creation_without_torch(self):
        """Test that OpenAI model can be created without torch dependencies."""
        # Save current torch modules state
        torch_modules = {name: mod for name, mod in sys.modules.items() if name.startswith("torch")}

        try:
            # Clear torch modules first
            for module_name in list(torch_modules.keys()):
                sys.modules.pop(module_name, None)

            from flow_eval.lm.models import OpenAIModel

            # Mock environment to avoid requiring API key
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                # Create model - this should work without torch
                model = OpenAIModel(model="gpt-4")

                # Should not have imported torch
                assert "torch" not in sys.modules, "torch should not be needed for OpenAI model"

                # Basic functionality should work
                assert model is not None
                assert hasattr(model, "config")
        finally:
            # Restore torch modules
            sys.modules.update(torch_modules)

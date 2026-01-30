"""Bayesian optimization for cmxflow workflows using Optuna."""

from pathlib import Path
from typing import Any

import optuna
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler

from cmxflow.block import ScoreBlock
from cmxflow.parameter import Categorical, Continuous, Integer, Parameter
from cmxflow.workflow import Workflow


class Optimizer:
    """Bayesian optimizer for cmxflow workflows using Optuna.

    Automatically optimizes workflow parameters to maximize or minimize
    a score computed by a ScoreBlock at the end of the workflow.

    Attributes:
        workflow: The workflow to optimize.
        input_path: Path to the input file for workflow execution.
        study: The Optuna study instance (available after optimize() is called).
    """

    def __init__(
        self,
        workflow: Workflow,
        input_path: Path | str,
        study_name: str | None = None,
        storage: str | None = None,
        sampler: BaseSampler | None = None,
        pruner: BasePruner | None = None,
    ) -> None:
        """Initialize the optimizer.

        Args:
            workflow: The workflow to optimize. Must end with a ScoreBlock.
            input_path: Path to the input file for workflow execution.
            study_name: Optional name for the Optuna study.
            storage: Optional storage URL for distributed optimization.
            sampler: Optional Optuna sampler for parameter suggestions.
            pruner: Optional Optuna pruner for early stopping.

        Raises:
            ValueError: If the workflow does not end with a ScoreBlock.
            ValueError: If the workflow has no optimizable parameters.
        """
        self._validate_workflow(workflow)
        self.workflow = workflow
        self.input_path = (
            Path(input_path) if isinstance(input_path, str) else input_path
        )
        self._study_name = study_name
        self._storage = storage
        self._sampler = sampler
        self._pruner = pruner
        self._study: optuna.Study | None = None
        self._params = workflow.get_params()

    def _validate_workflow(self, workflow: Workflow) -> None:
        """Validate that the workflow is suitable for optimization.

        Args:
            workflow: The workflow to validate.

        Raises:
            ValueError: If the workflow does not end with a ScoreBlock.
            ValueError: If the workflow has no optimizable parameters.
        """
        workflow.check()
        if not isinstance(workflow.blocks[-1], ScoreBlock):
            raise ValueError("Workflow must end with a ScoreBlock for optimization")
        if not workflow.get_params():
            raise ValueError("Workflow has no optimizable parameters")

    def _suggest_params(self, trial: optuna.Trial) -> None:
        """Suggest parameter values from Optuna and set them on the workflow.

        Args:
            trial: The current Optuna trial.
        """
        for param in self._params:
            value = self._suggest_param(trial, param)
            param.set(value)

    def _suggest_param(self, trial: optuna.Trial, param: Parameter) -> Any:
        """Suggest a value for a single parameter.

        Args:
            trial: The current Optuna trial.
            param: The parameter to suggest a value for.

        Returns:
            The suggested value.

        Raises:
            TypeError: If the parameter type is not supported.
        """
        if isinstance(param, Integer):
            return trial.suggest_int(param.name, param.low, param.high)
        elif isinstance(param, Continuous):
            return trial.suggest_float(param.name, param.low, param.high)
        elif isinstance(param, Categorical):
            return trial.suggest_categorical(param.name, param.choices)
        else:
            raise TypeError(f"Unsupported parameter type: {type(param)}")

    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization.

        Args:
            trial: The current Optuna trial.

        Returns:
            The score from the workflow execution.
        """
        # Reset block caches if necessary
        for block in self.workflow.blocks:
            block.reset_cache()
        self._suggest_params(trial)
        try:
            result = self.workflow.forward(self.input_path)
            if result is None:
                raise optuna.TrialPruned("Workflow returned None")
            score, _ = result
            return score
        except Exception as e:
            raise optuna.TrialPruned(f"Trial failed: {e}") from e

    def optimize(
        self,
        n_trials: int = 100,
        timeout: float | None = None,
        direction: str = "maximize",
        n_jobs: int = 1,
        show_progress_bar: bool = True,
        callbacks: list[Any] | None = None,
    ) -> optuna.Study:
        """Run the optimization.

        Args:
            n_trials: Number of trials to run.
            timeout: Maximum time in seconds for the optimization.
            direction: Optimization direction ("maximize" or "minimize").
            n_jobs: Number of parallel jobs (use -1 for all CPUs).
            show_progress_bar: Whether to show a progress bar.
            callbacks: Optional list of Optuna callback functions.

        Returns:
            The Optuna study containing optimization results.
        """
        self._study = optuna.create_study(
            study_name=self._study_name,
            storage=self._storage,
            sampler=self._sampler,
            pruner=self._pruner,
            direction=direction,
        )
        self._study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar,
            callbacks=callbacks,
        )
        return self._study

    @property
    def best_params(self) -> dict[str, Any]:
        """Get the best parameter values found during optimization.

        Returns:
            Dictionary mapping parameter names to their optimal values.

        Raises:
            RuntimeError: If optimize() has not been called yet.
        """
        if self._study is None:
            raise RuntimeError("No study available. Call optimize() first.")
        return dict(self._study.best_params)

    @property
    def best_score(self) -> float:
        """Get the best score achieved during optimization.

        Returns:
            The best score value.

        Raises:
            RuntimeError: If optimize() has not been called yet.
        """
        if self._study is None:
            raise RuntimeError("No study available. Call optimize() first.")
        return float(self._study.best_value)

    @property
    def study(self) -> optuna.Study:
        """Get the Optuna study instance.

        Returns:
            The Optuna study.

        Raises:
            RuntimeError: If optimize() has not been called yet.
        """
        if self._study is None:
            raise RuntimeError("No study available. Call optimize() first.")
        return self._study

    def set_best_params(self) -> None:
        """Set the workflow parameters to their optimal values.

        Raises:
            RuntimeError: If optimize() has not been called yet.
        """
        if self._study is None:
            raise RuntimeError("No study available. Call optimize() first.")
        best = self._study.best_params
        for param in self._params:
            if param.name in best:
                param.set(best[param.name])

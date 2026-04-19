"""Configuration objects that define how the framework should behave."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class DataStructure(StrEnum):
    """Supported structures determine how the dataset is split."""

    CROSS_SECTIONAL = "cross_sectional"
    TIME_SERIES = "time_series"
    PANEL = "panel"


class ModelType(StrEnum):
    """Supported model families exposed through the framework and GUI."""

    LOGISTIC_REGRESSION = "logistic_regression"
    ELASTIC_NET_LOGISTIC_REGRESSION = "elastic_net_logistic_regression"
    SCORECARD_LOGISTIC_REGRESSION = "scorecard_logistic_regression"
    PROBIT_REGRESSION = "probit_regression"
    LINEAR_REGRESSION = "linear_regression"
    BETA_REGRESSION = "beta_regression"
    TWO_STAGE_LGD_MODEL = "two_stage_lgd_model"
    PANEL_REGRESSION = "panel_regression"
    QUANTILE_REGRESSION = "quantile_regression"
    TOBIT_REGRESSION = "tobit_regression"
    XGBOOST = "xgboost"


class ExecutionMode(StrEnum):
    """Controls whether the pipeline fits a fresh model or scores an existing one."""

    FIT_NEW_MODEL = "fit_new_model"
    SCORE_EXISTING_MODEL = "score_existing_model"


class TargetMode(StrEnum):
    """Controls whether the framework should build a binary or continuous target."""

    BINARY = "binary"
    CONTINUOUS = "continuous"


class PresetName(StrEnum):
    """Named workflow presets for common model-development use cases."""

    PD_DEVELOPMENT = "pd_development"
    LIFETIME_PD_CECL = "lifetime_pd_cecl"
    LGD_SEVERITY = "lgd_severity"
    CCAR_FORECASTING = "ccar_forecasting"


class ScenarioShockOperation(StrEnum):
    """Supported scenario shock operations."""

    ADD = "add"
    MULTIPLY = "multiply"
    SET = "set"


class ColumnRole(StrEnum):
    """Roles control how each column is treated across the pipeline."""

    FEATURE = "feature"
    TARGET_SOURCE = "target_source"
    DATE = "date"
    IDENTIFIER = "identifier"
    IGNORE = "ignore"


class MissingValuePolicy(StrEnum):
    """Supported missing-value strategies available in the column designer."""

    INHERIT_DEFAULT = "inherit_default"
    NONE = "none"
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"


@dataclass(slots=True)
class ColumnSpec:
    """
    Describes a single column the user wants to control.

    This lets the user:
    - toggle an existing column on or off,
    - rename an input column into a framework-standard column,
    - create a new column with a default value,
    - and force a column into a specific dtype.
    """

    name: str
    source_name: str | None = None
    enabled: bool = True
    dtype: str | None = None
    role: ColumnRole = ColumnRole.FEATURE
    missing_value_policy: MissingValuePolicy = MissingValuePolicy.INHERIT_DEFAULT
    missing_value_fill_value: Any = None
    create_if_missing: bool = False
    default_value: Any = None
    keep_source: bool = False

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("ColumnSpec.name cannot be blank.")
        if self.dtype is not None and self.dtype.strip().lower() not in {
            "string",
            "str",
            "text",
            "category",
            "categorical",
            "float",
            "float64",
            "double",
            "int",
            "int64",
            "integer",
            "bool",
            "boolean",
            "datetime",
            "datetime64",
            "date",
        }:
            raise ValueError(
                f"Unsupported dtype '{self.dtype}' for column '{self.name}'. "
                "Use one of string/category/float/int/bool/datetime."
            )
        if (
            self.enabled
            and
            self.missing_value_policy == MissingValuePolicy.CONSTANT
            and self.missing_value_fill_value is None
        ):
            raise ValueError(
                f"Column '{self.name}' uses constant imputation but does not define "
                "missing_value_fill_value."
            )


@dataclass(slots=True)
class SchemaConfig:
    """Controls how the incoming dataframe is standardized."""

    column_specs: list[ColumnSpec] = field(default_factory=list)
    pass_through_unconfigured_columns: bool = True

    def validate(self) -> None:
        seen_names: set[str] = set()
        for spec in self.column_specs:
            spec.validate()
            if spec.name in seen_names:
                raise ValueError(f"Duplicate configured output column '{spec.name}'.")
            seen_names.add(spec.name)


@dataclass(slots=True)
class CleaningConfig:
    """Options for basic data hygiene before model fitting."""

    trim_string_columns: bool = True
    blank_strings_as_null: bool = True
    drop_duplicate_rows: bool = True
    drop_rows_with_missing_target: bool = True
    drop_all_null_feature_columns: bool = True


@dataclass(slots=True)
class FeatureEngineeringConfig:
    """Feature creation rules that remain simple and extensible in v1."""

    derive_date_parts: bool = True
    drop_raw_date_columns: bool = True
    date_parts: list[str] = field(default_factory=lambda: ["year", "month", "quarter", "dayofweek"])

    def validate(self) -> None:
        supported_parts = {"year", "month", "quarter", "day", "dayofweek"}
        unsupported_parts = [part for part in self.date_parts if part not in supported_parts]
        if unsupported_parts:
            raise ValueError(
                "Unsupported date parts: "
                + ", ".join(sorted(unsupported_parts))
                + ". Supported values are year/month/quarter/day/dayofweek."
            )


@dataclass(slots=True)
class TargetConfig:
    """How the framework should build the binary default flag."""

    source_column: str
    mode: TargetMode = TargetMode.BINARY
    output_column: str = "default_flag"
    positive_values: list[Any] | None = None
    drop_source_column: bool = False

    def validate(self) -> None:
        if not self.source_column.strip():
            raise ValueError("TargetConfig.source_column cannot be blank.")
        if not self.output_column.strip():
            raise ValueError("TargetConfig.output_column cannot be blank.")


@dataclass(slots=True)
class SplitConfig:
    """Defines the train/validation/test partition strategy."""

    data_structure: DataStructure = DataStructure.CROSS_SECTIONAL
    train_size: float = 0.6
    validation_size: float = 0.2
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    date_column: str | None = None
    entity_column: str | None = None

    def validate(self) -> None:
        total = self.train_size + self.validation_size + self.test_size
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "Split sizes must add to 1.0. "
                f"Received {self.train_size} + {self.validation_size} + {self.test_size} = {total}."
            )
        for split_name, split_value in {
            "train_size": self.train_size,
            "validation_size": self.validation_size,
            "test_size": self.test_size,
        }.items():
            if split_value <= 0:
                raise ValueError(f"{split_name} must be greater than 0. Received {split_value}.")


@dataclass(slots=True)
class ModelConfig:
    """Hyperparameters for the supported model families."""

    model_type: ModelType = ModelType.LOGISTIC_REGRESSION
    max_iter: int = 1000
    C: float = 1.0
    solver: str = "lbfgs"
    l1_ratio: float = 0.5
    class_weight: str | dict[str, float] | None = "balanced"
    threshold: float = 0.5
    scorecard_bins: int = 5
    beta_clip_epsilon: float = 1e-4
    lgd_positive_threshold: float = 1e-6
    quantile_alpha: float = 0.5
    xgboost_n_estimators: int = 300
    xgboost_learning_rate: float = 0.05
    xgboost_max_depth: int = 4
    xgboost_subsample: float = 0.9
    xgboost_colsample_bytree: float = 0.9
    tobit_left_censoring: float | None = 0.0
    tobit_right_censoring: float | None = None

    def validate(self, target_mode: TargetMode) -> None:
        if self.max_iter <= 0:
            raise ValueError("ModelConfig.max_iter must be greater than 0.")
        if self.C <= 0:
            raise ValueError("ModelConfig.C must be greater than 0.")
        if not 0 <= self.l1_ratio <= 1:
            raise ValueError("ModelConfig.l1_ratio must be in [0, 1].")
        if not 0 < self.threshold < 1:
            raise ValueError("ModelConfig.threshold must be strictly between 0 and 1.")
        if self.scorecard_bins < 2:
            raise ValueError("ModelConfig.scorecard_bins must be at least 2.")
        if not 0 < self.beta_clip_epsilon < 0.5:
            raise ValueError("ModelConfig.beta_clip_epsilon must be in (0, 0.5).")
        if self.lgd_positive_threshold < 0:
            raise ValueError("ModelConfig.lgd_positive_threshold cannot be negative.")
        if not 0 < self.quantile_alpha < 1:
            raise ValueError("ModelConfig.quantile_alpha must be strictly between 0 and 1.")
        if self.xgboost_n_estimators <= 0:
            raise ValueError("ModelConfig.xgboost_n_estimators must be greater than 0.")
        if not 0 < self.xgboost_learning_rate <= 1:
            raise ValueError("ModelConfig.xgboost_learning_rate must be in (0, 1].")
        if self.xgboost_max_depth <= 0:
            raise ValueError("ModelConfig.xgboost_max_depth must be greater than 0.")
        if not 0 < self.xgboost_subsample <= 1:
            raise ValueError("ModelConfig.xgboost_subsample must be in (0, 1].")
        if not 0 < self.xgboost_colsample_bytree <= 1:
            raise ValueError("ModelConfig.xgboost_colsample_bytree must be in (0, 1].")
        if (
            self.tobit_left_censoring is not None
            and self.tobit_right_censoring is not None
            and self.tobit_left_censoring > self.tobit_right_censoring
        ):
            raise ValueError("Tobit left censoring cannot be greater than right censoring.")
        if target_mode == TargetMode.BINARY and self.model_type in {
            ModelType.TOBIT_REGRESSION,
            ModelType.BETA_REGRESSION,
            ModelType.TWO_STAGE_LGD_MODEL,
            ModelType.PANEL_REGRESSION,
            ModelType.QUANTILE_REGRESSION,
        }:
            raise ValueError(f"{self.model_type.value} is only supported for continuous targets.")
        if target_mode == TargetMode.CONTINUOUS and self.model_type in {
            ModelType.LOGISTIC_REGRESSION,
            ModelType.ELASTIC_NET_LOGISTIC_REGRESSION,
            ModelType.SCORECARD_LOGISTIC_REGRESSION,
            ModelType.PROBIT_REGRESSION,
        }:
            raise ValueError(
                f"{self.model_type.value} requires a binary target. "
                "Use linear, beta, two-stage LGD, panel, quantile, Tobit, or XGBoost "
                "for continuous targets."
            )


@dataclass(slots=True)
class ComparisonConfig:
    """Controls optional challenger-model training during development runs."""

    enabled: bool = False
    challenger_model_types: list[ModelType] = field(default_factory=list)
    ranking_metric: str | None = None
    ranking_split: str = "validation"

    def validate(self, primary_model_type: ModelType, target_mode: TargetMode) -> None:
        if self.ranking_split not in {"validation", "test"}:
            raise ValueError("ComparisonConfig.ranking_split must be 'validation' or 'test'.")
        if not self.enabled:
            return
        if not self.challenger_model_types:
            raise ValueError(
                "ComparisonConfig.enabled=True requires at least one challenger model type."
            )
        for challenger in self.challenger_model_types:
            if challenger == primary_model_type:
                raise ValueError("Comparison challengers cannot repeat the primary model type.")
            ModelConfig(model_type=challenger).validate(target_mode)


@dataclass(slots=True)
class FeaturePolicyConfig:
    """Feature-governance rules used during development and documentation."""

    enabled: bool = False
    required_features: list[str] = field(default_factory=list)
    excluded_features: list[str] = field(default_factory=list)
    max_missing_pct: float | None = None
    max_vif: float | None = None
    minimum_information_value: float | None = None
    expected_signs: dict[str, str] = field(default_factory=dict)
    monotonic_features: dict[str, str] = field(default_factory=dict)
    error_on_violation: bool = False

    def validate(self) -> None:
        if self.max_missing_pct is not None and not 0 <= self.max_missing_pct <= 100:
            raise ValueError("FeaturePolicyConfig.max_missing_pct must be in [0, 100].")
        if self.max_vif is not None and self.max_vif <= 0:
            raise ValueError("FeaturePolicyConfig.max_vif must be greater than 0.")
        if self.minimum_information_value is not None and self.minimum_information_value < 0:
            raise ValueError(
                "FeaturePolicyConfig.minimum_information_value cannot be negative."
            )
        allowed_signs = {"positive", "negative", "nonnegative", "nonpositive"}
        invalid_signs = {
            feature: sign
            for feature, sign in self.expected_signs.items()
            if sign not in allowed_signs
        }
        if invalid_signs:
            raise ValueError(
                "FeaturePolicyConfig.expected_signs only supports "
                "positive/negative/nonnegative/nonpositive."
            )
        allowed_monotonicity = {"increasing", "decreasing"}
        invalid_monotonicity = {
            feature: direction
            for feature, direction in self.monotonic_features.items()
            if direction not in allowed_monotonicity
        }
        if invalid_monotonicity:
            raise ValueError(
                "FeaturePolicyConfig.monotonic_features only supports "
                "increasing/decreasing."
            )


@dataclass(slots=True)
class ExplainabilityConfig:
    """Controls explainability outputs generated during diagnostics."""

    enabled: bool = True
    permutation_importance: bool = True
    feature_effect_curves: bool = True
    coefficient_breakdown: bool = True
    top_n_features: int = 5
    grid_points: int = 12
    sample_size: int = 2000

    def validate(self) -> None:
        if self.top_n_features <= 0:
            raise ValueError("ExplainabilityConfig.top_n_features must be greater than 0.")
        if self.grid_points < 3:
            raise ValueError("ExplainabilityConfig.grid_points must be at least 3.")
        if self.sample_size <= 0:
            raise ValueError("ExplainabilityConfig.sample_size must be greater than 0.")


@dataclass(slots=True)
class ScenarioFeatureShock:
    """One feature-level change applied inside a scenario test."""

    feature_name: str
    operation: ScenarioShockOperation
    value: Any

    def validate(self) -> None:
        if not self.feature_name.strip():
            raise ValueError("ScenarioFeatureShock.feature_name cannot be blank.")


@dataclass(slots=True)
class ScenarioConfig:
    """A reusable scenario made up of one or more feature shocks."""

    name: str
    description: str = ""
    feature_shocks: list[ScenarioFeatureShock] = field(default_factory=list)
    enabled: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("ScenarioConfig.name cannot be blank.")
        if self.enabled and not self.feature_shocks:
            raise ValueError(
                "ScenarioConfig.enabled=True requires at least one feature shock."
            )
        for shock in self.feature_shocks:
            shock.validate()


@dataclass(slots=True)
class ScenarioTestConfig:
    """Controls optional what-if scenario scoring on held-out data."""

    enabled: bool = False
    evaluation_split: str = "test"
    scenarios: list[ScenarioConfig] = field(default_factory=list)

    def validate(self) -> None:
        if self.evaluation_split not in {"train", "validation", "test"}:
            raise ValueError(
                "ScenarioTestConfig.evaluation_split must be train, validation, or test."
            )
        if self.enabled and not self.scenarios:
            raise ValueError(
                "ScenarioTestConfig.enabled=True requires at least one configured scenario."
            )
        for scenario in self.scenarios:
            scenario.validate()


@dataclass(slots=True)
class ExecutionConfig:
    """Controls whether the run trains a model or reuses an existing artifact."""

    mode: ExecutionMode = ExecutionMode.FIT_NEW_MODEL
    existing_model_path: Path | None = None
    existing_config_path: Path | None = None

    def validate(self) -> None:
        if self.mode == ExecutionMode.SCORE_EXISTING_MODEL and self.existing_model_path is None:
            raise ValueError(
                "ExecutionConfig.existing_model_path is required when mode='score_existing_model'."
            )


@dataclass(slots=True)
class DiagnosticConfig:
    """Controls which diagnostics, charts, and export helpers are enabled."""

    data_quality: bool = True
    descriptive_statistics: bool = True
    missingness_analysis: bool = True
    correlation_analysis: bool = True
    vif_analysis: bool = True
    woe_iv_analysis: bool = True
    psi_analysis: bool = True
    adf_analysis: bool = True
    calibration_analysis: bool = True
    threshold_analysis: bool = True
    lift_gain_analysis: bool = True
    segment_analysis: bool = True
    residual_analysis: bool = True
    quantile_analysis: bool = True
    qq_analysis: bool = True
    interactive_visualizations: bool = True
    static_image_exports: bool = True
    export_excel_workbook: bool = True
    top_n_features: int = 15
    top_n_categories: int = 10
    max_plot_rows: int = 20000
    quantile_bucket_count: int = 10
    default_segment_column: str | None = None

    def validate(self) -> None:
        if self.top_n_features <= 0:
            raise ValueError("DiagnosticConfig.top_n_features must be greater than 0.")
        if self.top_n_categories <= 0:
            raise ValueError("DiagnosticConfig.top_n_categories must be greater than 0.")
        if self.max_plot_rows <= 0:
            raise ValueError("DiagnosticConfig.max_plot_rows must be greater than 0.")
        if self.quantile_bucket_count < 2:
            raise ValueError("DiagnosticConfig.quantile_bucket_count must be at least 2.")


@dataclass(slots=True)
class ArtifactConfig:
    """Where pipeline outputs are written."""

    output_root: Path = Path("artifacts")
    model_file_name: str = "quant_model.joblib"
    metrics_file_name: str = "metrics.json"
    input_snapshot_file_name: str = "input_snapshot.csv"
    predictions_file_name: str = "predictions.csv"
    feature_importance_file_name: str = "feature_importance.csv"
    backtest_file_name: str = "backtest_summary.csv"
    report_file_name: str = "run_report.md"
    interactive_report_file_name: str = "interactive_report.html"
    config_file_name: str = "run_config.json"
    statistical_tests_file_name: str = "statistical_tests.json"
    workbook_file_name: str = "analysis_workbook.xlsx"
    model_summary_file_name: str = "model_summary.txt"
    manifest_file_name: str = "artifact_manifest.json"
    step_manifest_file_name: str = "step_manifest.json"
    runner_script_file_name: str = "generated_run.py"
    rerun_readme_file_name: str = "HOW_TO_RERUN.md"
    tables_directory_name: str = "tables"
    figures_directory_name: str = "figures"
    html_directory_name: str = "html"
    png_directory_name: str = "png"
    json_directory_name: str = "json"
    code_snapshot_directory_name: str = "code_snapshot"
    export_input_snapshot: bool = True
    export_code_snapshot: bool = True

    def validate(self) -> None:
        if not self.output_root:
            raise ValueError("ArtifactConfig.output_root cannot be empty.")
        required_names = {
            "model_file_name": self.model_file_name,
            "metrics_file_name": self.metrics_file_name,
            "predictions_file_name": self.predictions_file_name,
            "feature_importance_file_name": self.feature_importance_file_name,
            "backtest_file_name": self.backtest_file_name,
            "report_file_name": self.report_file_name,
            "interactive_report_file_name": self.interactive_report_file_name,
            "config_file_name": self.config_file_name,
        }
        for field_name, value in required_names.items():
            if not value.strip():
                raise ValueError(f"ArtifactConfig.{field_name} cannot be blank.")


@dataclass(slots=True)
class FrameworkConfig:
    """Aggregates all pipeline controls into one object."""

    schema: SchemaConfig
    cleaning: CleaningConfig
    feature_engineering: FeatureEngineeringConfig
    target: TargetConfig
    split: SplitConfig
    preset_name: PresetName | None = None
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    feature_policy: FeaturePolicyConfig = field(default_factory=FeaturePolicyConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    scenario_testing: ScenarioTestConfig = field(default_factory=ScenarioTestConfig)
    diagnostics: DiagnosticConfig = field(default_factory=DiagnosticConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)

    def validate(self) -> None:
        self.schema.validate()
        self.feature_engineering.validate()
        self.target.validate()
        self.split.validate()
        self.execution.validate()
        self.model.validate(self.target.mode)
        self.comparison.validate(self.model.model_type, self.target.mode)
        self.feature_policy.validate()
        self.explainability.validate()
        self.scenario_testing.validate()
        self.diagnostics.validate()
        self.artifacts.validate()

    def to_dict(self) -> dict[str, Any]:
        """Serializes dataclasses, enums, and paths into JSON-friendly objects."""

        def convert(value: Any) -> Any:
            if isinstance(value, StrEnum):
                return value.value
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, list):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if hasattr(value, "__dataclass_fields__"):
                return {key: convert(item) for key, item in asdict(value).items()}
            return value

        return convert(self)

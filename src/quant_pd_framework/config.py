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
    DISCRETE_TIME_HAZARD_MODEL = "discrete_time_hazard_model"
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


class CalibrationStrategy(StrEnum):
    """Supported binning strategies for calibration diagnostics."""

    QUANTILE = "quantile"
    UNIFORM = "uniform"


class CalibrationRankingMetric(StrEnum):
    """Supported ranking metrics for choosing a calibration method."""

    BRIER_SCORE = "brier_score"
    LOG_LOSS = "log_loss"
    EXPECTED_CALIBRATION_ERROR = "expected_calibration_error"


class ScorecardMonotonicity(StrEnum):
    """Supported monotonic binning directions for scorecard development."""

    NONE = "none"
    AUTO = "auto"
    INCREASING = "increasing"
    DECREASING = "decreasing"


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


class TransformationType(StrEnum):
    """Supported governed feature-transformation types."""

    WINSORIZE = "winsorize"
    LOG1P = "log1p"
    YEO_JOHNSON = "yeo_johnson"
    CAPPED_ZSCORE = "capped_zscore"
    RATIO = "ratio"
    INTERACTION = "interaction"
    LAG = "lag"
    ROLLING_MEAN = "rolling_mean"
    PCT_CHANGE = "pct_change"
    MANUAL_BINS = "manual_bins"


class FeatureReviewDecisionType(StrEnum):
    """Supported manual-review decisions for feature selection."""

    APPROVE = "approve"
    REJECT = "reject"
    FORCE_INCLUDE = "force_include"
    FORCE_EXCLUDE = "force_exclude"


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
    missing_value_group_columns: list[str] = field(default_factory=list)
    create_missing_indicator: bool = False
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
            and self.missing_value_policy == MissingValuePolicy.CONSTANT
            and self.missing_value_fill_value is None
        ):
            raise ValueError(
                f"Column '{self.name}' uses constant imputation but does not define "
                "missing_value_fill_value."
            )
        normalized_group_columns = [
            str(column_name).strip()
            for column_name in self.missing_value_group_columns
            if str(column_name).strip()
        ]
        if len(normalized_group_columns) != len(set(normalized_group_columns)):
            raise ValueError(
                f"Column '{self.name}' has duplicate missing_value_group_columns entries."
            )
        if (
            self.enabled
            and (self.create_missing_indicator or normalized_group_columns)
            and self.role != ColumnRole.FEATURE
        ):
            raise ValueError(
                f"Column '{self.name}' can only use advanced imputation options when "
                "its role is 'feature'."
            )
        if self.name in normalized_group_columns:
            raise ValueError(
                f"Column '{self.name}' cannot use itself as an imputation grouping column."
            )
        if normalized_group_columns and self.missing_value_policy in {
            MissingValuePolicy.NONE,
            MissingValuePolicy.CONSTANT,
            MissingValuePolicy.FORWARD_FILL,
            MissingValuePolicy.BACKWARD_FILL,
        }:
            raise ValueError(
                f"Column '{self.name}' uses missing_value_group_columns with "
                f"policy '{self.missing_value_policy.value}', which is not supported."
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
            ModelType.DISCRETE_TIME_HAZARD_MODEL,
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
            raise ValueError("FeaturePolicyConfig.minimum_information_value cannot be negative.")
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
                "FeaturePolicyConfig.monotonic_features only supports increasing/decreasing."
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
class CalibrationConfig:
    """Controls development-grade calibration diagnostics and recalibration methods."""

    bin_count: int = 10
    strategy: CalibrationStrategy = CalibrationStrategy.QUANTILE
    platt_scaling: bool = True
    isotonic_calibration: bool = True
    ranking_metric: CalibrationRankingMetric = CalibrationRankingMetric.BRIER_SCORE

    def validate(self) -> None:
        if self.bin_count < 2:
            raise ValueError("CalibrationConfig.bin_count must be at least 2.")


@dataclass(slots=True)
class ScorecardConfig:
    """Controls scorecard-specific binning, scaling, and reason-code outputs."""

    monotonicity: ScorecardMonotonicity = ScorecardMonotonicity.AUTO
    min_bin_share: float = 0.05
    base_score: int = 600
    points_to_double_odds: int = 50
    odds_reference: float = 20.0
    reason_code_count: int = 3

    def validate(self) -> None:
        if not 0 < self.min_bin_share < 0.5:
            raise ValueError("ScorecardConfig.min_bin_share must be in (0, 0.5).")
        if self.points_to_double_odds <= 0:
            raise ValueError("ScorecardConfig.points_to_double_odds must be greater than 0.")
        if self.odds_reference <= 0:
            raise ValueError("ScorecardConfig.odds_reference must be greater than 0.")
        if self.reason_code_count <= 0:
            raise ValueError("ScorecardConfig.reason_code_count must be greater than 0.")


@dataclass(slots=True)
class ScorecardWorkbenchConfig:
    """Controls scorecard-specific workbench outputs and visual focus."""

    enabled: bool = True
    max_features: int = 6
    include_score_distribution: bool = True
    include_reason_code_analysis: bool = True

    def validate(self) -> None:
        if self.max_features <= 0:
            raise ValueError("ScorecardWorkbenchConfig.max_features must be greater than 0.")


@dataclass(slots=True)
class ImputationSensitivityConfig:
    """Controls what-if comparisons across alternate missing-value treatments."""

    enabled: bool = False
    evaluation_split: str = "test"
    alternative_policies: list[MissingValuePolicy] = field(
        default_factory=lambda: [
            MissingValuePolicy.MEAN,
            MissingValuePolicy.MEDIAN,
            MissingValuePolicy.MODE,
        ]
    )
    selected_features: list[str] = field(default_factory=list)
    max_features: int = 5
    min_missing_count: int = 1
    max_features_with_detail: int = 3

    def validate(self) -> None:
        if self.evaluation_split not in {"train", "validation", "test"}:
            raise ValueError(
                "ImputationSensitivityConfig.evaluation_split must be train, validation, or test."
            )
        if self.max_features <= 0:
            raise ValueError("ImputationSensitivityConfig.max_features must be greater than 0.")
        if self.min_missing_count <= 0:
            raise ValueError(
                "ImputationSensitivityConfig.min_missing_count must be greater than 0."
            )
        if self.max_features_with_detail <= 0:
            raise ValueError(
                "ImputationSensitivityConfig.max_features_with_detail must be greater than 0."
            )
        allowed_policies = {
            MissingValuePolicy.MEAN,
            MissingValuePolicy.MEDIAN,
            MissingValuePolicy.MODE,
        }
        unsupported = [
            policy.value for policy in self.alternative_policies if policy not in allowed_policies
        ]
        if unsupported:
            raise ValueError(
                "ImputationSensitivityConfig.alternative_policies only supports "
                f"mean/median/mode. Received: {', '.join(sorted(unsupported))}."
            )


@dataclass(slots=True)
class RobustnessConfig:
    """Controls repeated-resample robustness and stability diagnostics."""

    enabled: bool = False
    resample_count: int = 12
    sample_fraction: float = 0.8
    sample_with_replacement: bool = True
    evaluation_split: str = "test"
    metric_stability: bool = True
    coefficient_stability: bool = True
    random_state: int = 42

    def validate(self) -> None:
        if self.resample_count < 2:
            raise ValueError("RobustnessConfig.resample_count must be at least 2.")
        if not 0 < self.sample_fraction <= 1:
            raise ValueError("RobustnessConfig.sample_fraction must be in (0, 1].")
        if self.evaluation_split not in {"train", "validation", "test"}:
            raise ValueError(
                "RobustnessConfig.evaluation_split must be train, validation, or test."
            )
        if self.enabled and not (self.metric_stability or self.coefficient_stability):
            raise ValueError(
                "RobustnessConfig.enabled=True requires at least one output type to be enabled."
            )


@dataclass(slots=True)
class VariableSelectionConfig:
    """Controls the optional train-split variable-selection workflow."""

    enabled: bool = False
    max_features: int | None = None
    min_univariate_score: float | None = None
    correlation_threshold: float | None = 0.8
    locked_include_features: list[str] = field(default_factory=list)
    locked_exclude_features: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.max_features is not None and self.max_features <= 0:
            raise ValueError(
                "VariableSelectionConfig.max_features must be greater than 0 when set."
            )
        if self.min_univariate_score is not None and self.min_univariate_score < 0:
            raise ValueError("VariableSelectionConfig.min_univariate_score cannot be negative.")
        if self.correlation_threshold is not None and not 0 < self.correlation_threshold <= 1:
            raise ValueError("VariableSelectionConfig.correlation_threshold must be in (0, 1].")


@dataclass(slots=True)
class FeatureDictionaryEntry:
    """Business and governance metadata attached to a modeled feature."""

    feature_name: str
    business_name: str = ""
    definition: str = ""
    source_system: str = ""
    unit: str = ""
    allowed_range: str = ""
    missingness_meaning: str = ""
    expected_sign: str = ""
    inclusion_rationale: str = ""
    notes: str = ""
    enabled: bool = True

    def validate(self) -> None:
        if not self.feature_name.strip():
            raise ValueError("FeatureDictionaryEntry.feature_name cannot be blank.")


@dataclass(slots=True)
class FeatureDictionaryConfig:
    """Controls the optional feature dictionary exported with each run."""

    enabled: bool = False
    entries: list[FeatureDictionaryEntry] = field(default_factory=list)
    require_documentation_for_selected_features: bool = False

    def validate(self) -> None:
        seen_features: set[str] = set()
        for entry in self.entries:
            entry.validate()
            if entry.feature_name in seen_features:
                raise ValueError(f"Duplicate feature dictionary entry '{entry.feature_name}'.")
            seen_features.add(entry.feature_name)


@dataclass(slots=True)
class TransformationSpec:
    """One governed feature transformation to fit and apply reproducibly."""

    transform_type: TransformationType
    source_feature: str
    output_feature: str | None = None
    secondary_feature: str | None = None
    categorical_value: str | None = None
    lower_quantile: float | None = None
    upper_quantile: float | None = None
    parameter_value: float | None = None
    window_size: int | None = None
    lag_periods: int | None = None
    bin_edges: list[float] = field(default_factory=list)
    enabled: bool = True
    generated_automatically: bool = False
    notes: str = ""

    def validate(self) -> None:
        if not self.source_feature.strip():
            raise ValueError("TransformationSpec.source_feature cannot be blank.")
        if self.transform_type == TransformationType.WINSORIZE:
            lower = 0.01 if self.lower_quantile is None else self.lower_quantile
            upper = 0.99 if self.upper_quantile is None else self.upper_quantile
            if not 0 <= lower < upper <= 1:
                raise ValueError("Winsorization quantiles must satisfy 0 <= lower < upper <= 1.")
        if (
            self.transform_type
            in {
                TransformationType.RATIO,
                TransformationType.INTERACTION,
            }
            and not (self.secondary_feature or "").strip()
        ):
            raise ValueError(
                f"{self.transform_type.value} transformations require secondary_feature."
            )
        if self.transform_type == TransformationType.CAPPED_ZSCORE:
            z_cap = 3.0 if self.parameter_value is None else self.parameter_value
            if z_cap <= 0:
                raise ValueError("capped_zscore parameter_value must be greater than 0.")
        if self.transform_type in {
            TransformationType.LAG,
            TransformationType.PCT_CHANGE,
        }:
            lag_periods = 1 if self.lag_periods is None else self.lag_periods
            if lag_periods <= 0:
                raise ValueError(
                    f"{self.transform_type.value} transformations require lag_periods > 0."
                )
        if self.transform_type == TransformationType.ROLLING_MEAN:
            window_size = 3 if self.window_size is None else self.window_size
            if window_size <= 1:
                raise ValueError("rolling_mean transformations require window_size >= 2.")
        if self.transform_type == TransformationType.MANUAL_BINS:
            if not self.bin_edges:
                raise ValueError("manual_bins transformations require at least one bin edge.")
            if sorted(self.bin_edges) != list(self.bin_edges):
                raise ValueError("manual_bins bin_edges must be sorted in ascending order.")


@dataclass(slots=True)
class TransformationConfig:
    """Controls governed, exportable feature transformations."""

    enabled: bool = False
    transformations: list[TransformationSpec] = field(default_factory=list)
    error_on_failure: bool = True
    auto_interactions_enabled: bool = False
    include_numeric_numeric_interactions: bool = True
    include_categorical_numeric_interactions: bool = False
    max_auto_interactions: int = 5
    max_categorical_levels: int = 3
    min_interaction_score: float = 0.0

    def validate(self) -> None:
        seen_outputs: set[str] = set()
        for transformation in self.transformations:
            transformation.validate()
            output_name = self._resolve_output_name(transformation)
            if output_name in seen_outputs:
                raise ValueError(f"Multiple governed transformations write to '{output_name}'.")
            seen_outputs.add(output_name)
        if self.max_auto_interactions <= 0:
            raise ValueError("TransformationConfig.max_auto_interactions must be greater than 0.")
        if self.max_categorical_levels <= 0:
            raise ValueError("TransformationConfig.max_categorical_levels must be greater than 0.")
        if self.min_interaction_score < 0:
            raise ValueError("TransformationConfig.min_interaction_score cannot be negative.")

    def _resolve_output_name(self, transformation: TransformationSpec) -> str:
        configured_output = (transformation.output_feature or "").strip()
        if configured_output:
            return configured_output
        if transformation.transform_type == TransformationType.MANUAL_BINS:
            return f"{transformation.source_feature}_binned"
        if transformation.transform_type == TransformationType.RATIO:
            return f"{transformation.source_feature}_over_{transformation.secondary_feature}"
        if transformation.transform_type == TransformationType.INTERACTION:
            if (transformation.categorical_value or "").strip():
                category_token = (
                    transformation.categorical_value.strip().replace(" ", "_").replace("/", "_")
                )
                return (
                    f"{transformation.source_feature}_x_"
                    f"{transformation.secondary_feature}_{category_token}"
                )
            return f"{transformation.source_feature}_x_{transformation.secondary_feature}"
        if transformation.transform_type == TransformationType.YEO_JOHNSON:
            return f"{transformation.source_feature}_yeo_johnson"
        if transformation.transform_type == TransformationType.CAPPED_ZSCORE:
            return f"{transformation.source_feature}_zscore"
        if transformation.transform_type == TransformationType.LAG:
            lag_periods = 1 if transformation.lag_periods is None else transformation.lag_periods
            return f"{transformation.source_feature}_lag_{lag_periods}"
        if transformation.transform_type == TransformationType.ROLLING_MEAN:
            window_size = 3 if transformation.window_size is None else transformation.window_size
            return f"{transformation.source_feature}_rollmean_{window_size}"
        if transformation.transform_type == TransformationType.PCT_CHANGE:
            lag_periods = 1 if transformation.lag_periods is None else transformation.lag_periods
            return f"{transformation.source_feature}_pct_change_{lag_periods}"
        return transformation.source_feature


@dataclass(slots=True)
class FeatureReviewDecision:
    """Manual review decision attached to a candidate feature."""

    feature_name: str
    decision: FeatureReviewDecisionType
    rationale: str = ""

    def validate(self) -> None:
        if not self.feature_name.strip():
            raise ValueError("FeatureReviewDecision.feature_name cannot be blank.")


@dataclass(slots=True)
class ScorecardBinOverride:
    """Manual scorecard bin override for one numeric feature."""

    feature_name: str
    bin_edges: list[float] = field(default_factory=list)
    rationale: str = ""

    def validate(self) -> None:
        if not self.feature_name.strip():
            raise ValueError("ScorecardBinOverride.feature_name cannot be blank.")
        if not self.bin_edges:
            raise ValueError("ScorecardBinOverride.bin_edges cannot be empty.")
        if sorted(self.bin_edges) != list(self.bin_edges):
            raise ValueError("ScorecardBinOverride.bin_edges must be sorted in ascending order.")


@dataclass(slots=True)
class ManualReviewConfig:
    """Captures human review of selected variables and scorecard bin overrides."""

    enabled: bool = False
    reviewer_name: str = ""
    require_review_complete: bool = False
    feature_decisions: list[FeatureReviewDecision] = field(default_factory=list)
    scorecard_bin_overrides: list[ScorecardBinOverride] = field(default_factory=list)

    def validate(self) -> None:
        seen_features: set[str] = set()
        for decision in self.feature_decisions:
            decision.validate()
            if decision.feature_name in seen_features:
                raise ValueError(
                    f"Duplicate manual feature review entry '{decision.feature_name}'."
                )
            seen_features.add(decision.feature_name)

        seen_overrides: set[str] = set()
        for override in self.scorecard_bin_overrides:
            override.validate()
            if override.feature_name in seen_overrides:
                raise ValueError(f"Duplicate scorecard bin override '{override.feature_name}'.")
            seen_overrides.add(override.feature_name)


@dataclass(slots=True)
class SuitabilityCheckConfig:
    """Controls pre-fit assumption and model-suitability checks."""

    enabled: bool = True
    min_events_per_feature: float | None = 10.0
    min_class_rate: float | None = 0.01
    max_class_rate: float | None = 0.99
    max_dominant_category_share: float | None = 0.98
    min_non_null_target_rows: int = 30
    error_on_failure: bool = False

    def validate(self) -> None:
        if self.min_events_per_feature is not None and self.min_events_per_feature <= 0:
            raise ValueError(
                "SuitabilityCheckConfig.min_events_per_feature must be greater than 0."
            )
        for field_name, value in {
            "min_class_rate": self.min_class_rate,
            "max_class_rate": self.max_class_rate,
            "max_dominant_category_share": self.max_dominant_category_share,
        }.items():
            if value is not None and not 0 < value < 1:
                raise ValueError(f"SuitabilityCheckConfig.{field_name} must be in (0, 1).")
        if (
            self.min_class_rate is not None
            and self.max_class_rate is not None
            and self.min_class_rate >= self.max_class_rate
        ):
            raise ValueError(
                "SuitabilityCheckConfig.min_class_rate must be less than max_class_rate."
            )
        if self.min_non_null_target_rows <= 0:
            raise ValueError(
                "SuitabilityCheckConfig.min_non_null_target_rows must be greater than 0."
            )


@dataclass(slots=True)
class WorkflowGuardrailConfig:
    """Controls preset-aware workflow readiness checks before model execution."""

    enabled: bool = True
    fail_on_error: bool = True
    enforce_documentation_requirements: bool = True

    def validate(self) -> None:
        if self.enabled and not self.fail_on_error and not self.enforce_documentation_requirements:
            return


@dataclass(slots=True)
class ReproducibilityConfig:
    """Controls run-manifest metadata used for auditability and reruns."""

    enabled: bool = True
    capture_git_metadata: bool = True
    package_names: list[str] = field(
        default_factory=lambda: [
            "quant-pd-framework",
            "pandas",
            "numpy",
            "scikit-learn",
            "statsmodels",
            "xgboost",
            "plotly",
            "streamlit",
            "joblib",
            "openpyxl",
        ]
    )

    def validate(self) -> None:
        package_names = [package_name.strip() for package_name in self.package_names]
        if any(not package_name for package_name in package_names):
            raise ValueError("ReproducibilityConfig.package_names cannot contain blanks.")
        if len(package_names) != len(set(package_names)):
            raise ValueError("ReproducibilityConfig.package_names must be unique.")


@dataclass(slots=True)
class DocumentationConfig:
    """Captures development metadata for the exported documentation pack."""

    enabled: bool = True
    model_name: str = "Quant Studio Model"
    model_owner: str = ""
    business_purpose: str = ""
    portfolio_name: str = ""
    segment_name: str = ""
    horizon_definition: str = ""
    target_definition: str = ""
    loss_definition: str = ""
    assumptions: list[str] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    reviewer_notes: str = ""

    def validate(self) -> None:
        if self.enabled and not self.model_name.strip():
            raise ValueError("DocumentationConfig.model_name cannot be blank.")


@dataclass(slots=True)
class RegulatoryReportConfig:
    """Controls committee-ready and validation-ready report exports."""

    enabled: bool = True
    export_docx: bool = True
    export_pdf: bool = True
    committee_template_name: str = "committee_standard"
    validation_template_name: str = "validation_standard"
    include_assumptions_section: bool = True
    include_challenger_section: bool = True
    include_scenario_section: bool = True
    include_appendix_section: bool = True

    def validate(self) -> None:
        if not self.enabled:
            return
        if not (self.export_docx or self.export_pdf):
            raise ValueError("RegulatoryReportConfig.enabled=True requires DOCX or PDF export.")
        if not self.committee_template_name.strip():
            raise ValueError("RegulatoryReportConfig.committee_template_name cannot be blank.")
        if not self.validation_template_name.strip():
            raise ValueError("RegulatoryReportConfig.validation_template_name cannot be blank.")


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
            raise ValueError("ScenarioConfig.enabled=True requires at least one feature shock.")
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
    model_specification_tests: bool = True
    forecasting_statistical_tests: bool = True
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
class CreditRiskDiagnosticConfig:
    """Controls credit-risk-specific development diagnostics and plots."""

    enabled: bool = True
    vintage_analysis: bool = True
    migration_analysis: bool = True
    delinquency_transition_analysis: bool = True
    cohort_pd_analysis: bool = True
    lgd_segment_analysis: bool = True
    recovery_analysis: bool = True
    macro_sensitivity_analysis: bool = True
    top_macro_features: int = 5
    top_segments: int = 8
    shock_std_multiplier: float = 1.0

    def validate(self) -> None:
        if self.top_macro_features <= 0:
            raise ValueError(
                "CreditRiskDiagnosticConfig.top_macro_features must be greater than 0."
            )
        if self.top_segments <= 0:
            raise ValueError("CreditRiskDiagnosticConfig.top_segments must be greater than 0.")
        if self.shock_std_multiplier <= 0:
            raise ValueError(
                "CreditRiskDiagnosticConfig.shock_std_multiplier must be greater than 0."
            )


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
    documentation_pack_file_name: str = "model_documentation_pack.md"
    validation_pack_file_name: str = "validation_pack.md"
    committee_report_docx_file_name: str = "committee_report.docx"
    validation_report_docx_file_name: str = "validation_report.docx"
    committee_report_pdf_file_name: str = "committee_report.pdf"
    validation_report_pdf_file_name: str = "validation_report.pdf"
    reproducibility_manifest_file_name: str = "reproducibility_manifest.json"
    template_workbook_file_name: str = "configuration_template.xlsx"
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
            "documentation_pack_file_name": self.documentation_pack_file_name,
            "validation_pack_file_name": self.validation_pack_file_name,
            "committee_report_docx_file_name": self.committee_report_docx_file_name,
            "validation_report_docx_file_name": self.validation_report_docx_file_name,
            "committee_report_pdf_file_name": self.committee_report_pdf_file_name,
            "validation_report_pdf_file_name": self.validation_report_pdf_file_name,
            "reproducibility_manifest_file_name": self.reproducibility_manifest_file_name,
            "template_workbook_file_name": self.template_workbook_file_name,
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
    feature_dictionary: FeatureDictionaryConfig = field(default_factory=FeatureDictionaryConfig)
    transformations: TransformationConfig = field(default_factory=TransformationConfig)
    manual_review: ManualReviewConfig = field(default_factory=ManualReviewConfig)
    suitability_checks: SuitabilityCheckConfig = field(default_factory=SuitabilityCheckConfig)
    workflow_guardrails: WorkflowGuardrailConfig = field(default_factory=WorkflowGuardrailConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    scorecard: ScorecardConfig = field(default_factory=ScorecardConfig)
    scorecard_workbench: ScorecardWorkbenchConfig = field(default_factory=ScorecardWorkbenchConfig)
    imputation_sensitivity: ImputationSensitivityConfig = field(
        default_factory=ImputationSensitivityConfig
    )
    variable_selection: VariableSelectionConfig = field(default_factory=VariableSelectionConfig)
    documentation: DocumentationConfig = field(default_factory=DocumentationConfig)
    regulatory_reporting: RegulatoryReportConfig = field(default_factory=RegulatoryReportConfig)
    scenario_testing: ScenarioTestConfig = field(default_factory=ScenarioTestConfig)
    diagnostics: DiagnosticConfig = field(default_factory=DiagnosticConfig)
    credit_risk: CreditRiskDiagnosticConfig = field(default_factory=CreditRiskDiagnosticConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)
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
        self.feature_dictionary.validate()
        self.transformations.validate()
        self.manual_review.validate()
        self.suitability_checks.validate()
        self.workflow_guardrails.validate()
        self.explainability.validate()
        self.calibration.validate()
        self.scorecard.validate()
        self.scorecard_workbench.validate()
        self.imputation_sensitivity.validate()
        self.variable_selection.validate()
        self.documentation.validate()
        self.regulatory_reporting.validate()
        self.scenario_testing.validate()
        self.diagnostics.validate()
        self.credit_risk.validate()
        self.robustness.validate()
        self.reproducibility.validate()
        self.artifacts.validate()
        if self.workflow_guardrails.enabled:
            from .workflow_guardrails import (
                evaluate_workflow_guardrails,
                has_blocking_guardrails,
                summarize_guardrail_findings,
            )

            findings = evaluate_workflow_guardrails(self)
            if self.workflow_guardrails.fail_on_error and has_blocking_guardrails(findings):
                raise ValueError(
                    "Workflow guardrails failed for the selected preset:\n"
                    + summarize_guardrail_findings(findings)
                )

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

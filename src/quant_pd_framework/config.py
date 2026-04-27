"""Configuration objects that define how the framework should behave."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .config_serialization import framework_config_to_dict


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
    SEARCH_FEATURE_SUBSETS = "search_feature_subsets"


class ExportProfile(StrEnum):
    """Controls how much artifact packaging work is performed after a run."""

    FAST = "fast"
    STANDARD = "standard"
    AUDIT = "audit"


class TabularOutputFormat(StrEnum):
    """Controls file formats for large tabular artifacts."""

    CSV = "csv"
    PARQUET = "parquet"
    BOTH = "both"


class LargeDataExportPolicy(StrEnum):
    """Controls whether large tabular exports are full, sampled, or metadata-only."""

    FULL = "full"
    SAMPLED = "sampled"
    METADATA_ONLY = "metadata_only"


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


class CrossValidationStrategy(StrEnum):
    """Supported validation resampling strategies."""

    AUTO = "auto"
    KFOLD = "kfold"
    STRATIFIED_KFOLD = "stratified_kfold"
    TIME_SERIES = "time_series"


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
    KNN = "knn"
    ITERATIVE = "iterative"


class TransformationType(StrEnum):
    """Supported governed feature-transformation types."""

    WINSORIZE = "winsorize"
    LOG1P = "log1p"
    BOX_COX = "box_cox"
    NATURAL_SPLINE = "natural_spline"
    YEO_JOHNSON = "yeo_johnson"
    CAPPED_ZSCORE = "capped_zscore"
    PIECEWISE_LINEAR = "piecewise_linear"
    RATIO = "ratio"
    INTERACTION = "interaction"
    LAG = "lag"
    DIFFERENCE = "difference"
    EWMA = "ewma"
    ROLLING_MEAN = "rolling_mean"
    ROLLING_MEDIAN = "rolling_median"
    ROLLING_MIN = "rolling_min"
    ROLLING_MAX = "rolling_max"
    ROLLING_STD = "rolling_std"
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
            MissingValuePolicy.KNN,
            MissingValuePolicy.ITERATIVE,
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
class FeatureSubsetSearchConfig:
    """Controls exhaustive feature-subset comparison for a fixed model family."""

    enabled: bool = False
    candidate_feature_names: list[str] = field(default_factory=list)
    locked_include_features: list[str] = field(default_factory=list)
    locked_exclude_features: list[str] = field(default_factory=list)
    min_subset_size: int = 1
    max_subset_size: int | None = 4
    max_candidate_features: int = 12
    ranking_split: str = "validation"
    ranking_metric: str = "roc_auc"
    top_candidate_count: int = 25
    top_curve_count: int = 5
    include_significance_tests: bool = True

    def validate(self) -> None:
        if self.min_subset_size <= 0:
            raise ValueError("FeatureSubsetSearchConfig.min_subset_size must be greater than 0.")
        if self.max_subset_size is not None and self.max_subset_size < self.min_subset_size:
            raise ValueError(
                "FeatureSubsetSearchConfig.max_subset_size must be greater than or equal "
                "to min_subset_size."
            )
        if self.max_candidate_features <= 1:
            raise ValueError(
                "FeatureSubsetSearchConfig.max_candidate_features must be greater than 1."
            )
        if self.ranking_split not in {"validation", "test"}:
            raise ValueError(
                "FeatureSubsetSearchConfig.ranking_split must be 'validation' or 'test'."
            )
        if self.ranking_metric not in {
            "roc_auc",
            "ks_statistic",
            "average_precision",
            "brier_score",
            "log_loss",
        }:
            raise ValueError(
                "FeatureSubsetSearchConfig.ranking_metric only supports roc_auc, "
                "ks_statistic, average_precision, brier_score, or log_loss."
            )
        if self.top_candidate_count <= 0:
            raise ValueError(
                "FeatureSubsetSearchConfig.top_candidate_count must be greater than 0."
            )
        if self.top_curve_count <= 0:
            raise ValueError("FeatureSubsetSearchConfig.top_curve_count must be greater than 0.")
        for field_name, values in {
            "candidate_feature_names": self.candidate_feature_names,
            "locked_include_features": self.locked_include_features,
            "locked_exclude_features": self.locked_exclude_features,
        }.items():
            normalized = [value.strip() for value in values if value and value.strip()]
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"FeatureSubsetSearchConfig.{field_name} must be unique.")
        overlap = set(self.locked_include_features) & set(self.locked_exclude_features)
        if overlap:
            raise ValueError(
                "FeatureSubsetSearchConfig.locked_include_features and "
                "locked_exclude_features cannot overlap."
            )


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
    partial_dependence: bool = True
    ice_curves: bool = True
    centered_ice_curves: bool = True
    accumulated_local_effects: bool = True
    two_way_effects: bool = True
    effect_confidence_bands: bool = True
    monotonicity_diagnostics: bool = True
    segmented_effects: bool = True
    effect_stability: bool = True
    marginal_effects: bool = True
    interaction_strength: bool = True
    effect_calibration: bool = True
    coefficient_breakdown: bool = True
    top_n_features: int = 5
    grid_points: int = 12
    sample_size: int = 2000
    ice_sample_size: int = 250
    effect_band_resamples: int = 20
    two_way_grid_points: int = 6
    max_effect_segments: int = 4

    def validate(self) -> None:
        if self.top_n_features <= 0:
            raise ValueError("ExplainabilityConfig.top_n_features must be greater than 0.")
        if self.grid_points < 3:
            raise ValueError("ExplainabilityConfig.grid_points must be at least 3.")
        if self.sample_size <= 0:
            raise ValueError("ExplainabilityConfig.sample_size must be greater than 0.")
        if self.ice_sample_size <= 0:
            raise ValueError("ExplainabilityConfig.ice_sample_size must be greater than 0.")
        if self.effect_band_resamples < 2:
            raise ValueError("ExplainabilityConfig.effect_band_resamples must be at least 2.")
        if self.two_way_grid_points < 3:
            raise ValueError("ExplainabilityConfig.two_way_grid_points must be at least 3.")
        if self.max_effect_segments <= 0:
            raise ValueError("ExplainabilityConfig.max_effect_segments must be greater than 0.")


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
            MissingValuePolicy.KNN,
            MissingValuePolicy.ITERATIVE,
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
            MissingValuePolicy.KNN,
            MissingValuePolicy.ITERATIVE,
        }
        unsupported = [
            policy.value for policy in self.alternative_policies if policy not in allowed_policies
        ]
        if unsupported:
            raise ValueError(
                "ImputationSensitivityConfig.alternative_policies only supports "
                "mean/median/mode/knn/iterative. "
                f"Received: {', '.join(sorted(unsupported))}."
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
class CrossValidationConfig:
    """Controls optional fold-based validation without changing the final model fit."""

    enabled: bool = False
    fold_count: int = 5
    strategy: CrossValidationStrategy = CrossValidationStrategy.AUTO
    shuffle: bool = True
    metric_stability: bool = True
    coefficient_stability: bool = True
    random_state: int = 42

    def validate(self) -> None:
        if self.fold_count < 2:
            raise ValueError("CrossValidationConfig.fold_count must be at least 2.")
        if self.enabled and not (self.metric_stability or self.coefficient_stability):
            raise ValueError(
                "CrossValidationConfig.enabled=True requires at least one output type."
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
        if self.transform_type == TransformationType.NATURAL_SPLINE:
            spline_df = 4 if self.parameter_value is None else int(self.parameter_value)
            if spline_df < 3:
                raise ValueError(
                    "natural_spline transformations require parameter_value >= 3 "
                    "to define the spline degrees of freedom."
                )
        if self.transform_type == TransformationType.PIECEWISE_LINEAR:
            if self.parameter_value is None:
                raise ValueError(
                    "piecewise_linear transformations require parameter_value "
                    "to define the hinge point."
                )
        if self.transform_type in {
            TransformationType.LAG,
            TransformationType.DIFFERENCE,
            TransformationType.PCT_CHANGE,
        }:
            lag_periods = 1 if self.lag_periods is None else self.lag_periods
            if lag_periods <= 0:
                raise ValueError(
                    f"{self.transform_type.value} transformations require lag_periods > 0."
                )
        if self.transform_type in {
            TransformationType.EWMA,
            TransformationType.ROLLING_MEAN,
            TransformationType.ROLLING_MEDIAN,
            TransformationType.ROLLING_MIN,
            TransformationType.ROLLING_MAX,
            TransformationType.ROLLING_STD,
        }:
            window_size = 3 if self.window_size is None else self.window_size
            if window_size <= 1:
                raise ValueError(
                    f"{self.transform_type.value} transformations require window_size >= 2."
                )
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
        if transformation.transform_type == TransformationType.BOX_COX:
            return f"{transformation.source_feature}_box_cox"
        if transformation.transform_type == TransformationType.NATURAL_SPLINE:
            spline_df = (
                4 if transformation.parameter_value is None else int(transformation.parameter_value)
            )
            return f"{transformation.source_feature}_spline_df_{spline_df}"
        if transformation.transform_type == TransformationType.CAPPED_ZSCORE:
            return f"{transformation.source_feature}_zscore"
        if transformation.transform_type == TransformationType.PIECEWISE_LINEAR:
            hinge_point = transformation.parameter_value
            hinge_token = "hinge" if hinge_point is None else str(hinge_point).replace(".", "_")
            return f"{transformation.source_feature}_piecewise_{hinge_token}"
        if transformation.transform_type == TransformationType.LAG:
            lag_periods = 1 if transformation.lag_periods is None else transformation.lag_periods
            return f"{transformation.source_feature}_lag_{lag_periods}"
        if transformation.transform_type == TransformationType.DIFFERENCE:
            lag_periods = 1 if transformation.lag_periods is None else transformation.lag_periods
            return f"{transformation.source_feature}_diff_{lag_periods}"
        if transformation.transform_type == TransformationType.EWMA:
            window_size = 3 if transformation.window_size is None else transformation.window_size
            return f"{transformation.source_feature}_ewma_{window_size}"
        if transformation.transform_type == TransformationType.ROLLING_MEAN:
            window_size = 3 if transformation.window_size is None else transformation.window_size
            return f"{transformation.source_feature}_rollmean_{window_size}"
        if transformation.transform_type == TransformationType.ROLLING_MEDIAN:
            window_size = 3 if transformation.window_size is None else transformation.window_size
            return f"{transformation.source_feature}_rollmedian_{window_size}"
        if transformation.transform_type == TransformationType.ROLLING_MIN:
            window_size = 3 if transformation.window_size is None else transformation.window_size
            return f"{transformation.source_feature}_rollmin_{window_size}"
        if transformation.transform_type == TransformationType.ROLLING_MAX:
            window_size = 3 if transformation.window_size is None else transformation.window_size
            return f"{transformation.source_feature}_rollmax_{window_size}"
        if transformation.transform_type == TransformationType.ROLLING_STD:
            window_size = 3 if transformation.window_size is None else transformation.window_size
            return f"{transformation.source_feature}_rollstd_{window_size}"
        if transformation.transform_type == TransformationType.PCT_CHANGE:
            lag_periods = 1 if transformation.lag_periods is None else transformation.lag_periods
            return f"{transformation.source_feature}_pct_change_{lag_periods}"
        return transformation.source_feature


@dataclass(slots=True)
class AdvancedImputationConfig:
    """Controls model-based and framework-level missing-value features."""

    enabled: bool = True
    knn_neighbors: int = 5
    iterative_max_iter: int = 10
    iterative_random_state: int = 42
    iterative_sample_posterior: bool = False
    max_auxiliary_numeric_features: int = 25
    minimum_complete_rows: int = 20
    multiple_imputation_enabled: bool = False
    multiple_imputation_datasets: int = 5
    multiple_imputation_evaluation_split: str = "test"
    multiple_imputation_top_features: int = 20

    def validate(self) -> None:
        if self.knn_neighbors <= 0:
            raise ValueError("AdvancedImputationConfig.knn_neighbors must be greater than 0.")
        if self.iterative_max_iter <= 0:
            raise ValueError("AdvancedImputationConfig.iterative_max_iter must be greater than 0.")
        if self.max_auxiliary_numeric_features <= 0:
            raise ValueError(
                "AdvancedImputationConfig.max_auxiliary_numeric_features must be greater than 0."
            )
        if self.minimum_complete_rows <= 0:
            raise ValueError(
                "AdvancedImputationConfig.minimum_complete_rows must be greater than 0."
            )
        if self.multiple_imputation_datasets < 2:
            raise ValueError(
                "AdvancedImputationConfig.multiple_imputation_datasets must be at least 2."
            )
        if self.multiple_imputation_evaluation_split not in {"train", "validation", "test"}:
            raise ValueError(
                "AdvancedImputationConfig.multiple_imputation_evaluation_split must be "
                "train, validation, or test."
            )
        if self.multiple_imputation_top_features <= 0:
            raise ValueError(
                "AdvancedImputationConfig.multiple_imputation_top_features must be greater than 0."
            )


@dataclass(slots=True)
class DistributionDiagnosticConfig:
    """Controls distributional testing and shift diagnostics."""

    enabled: bool = True
    include_normality_tests: bool = True
    include_shift_tests: bool = True
    top_features: int = 8
    minimum_rows: int = 30

    def validate(self) -> None:
        if self.top_features <= 0:
            raise ValueError("DistributionDiagnosticConfig.top_features must be greater than 0.")
        if self.minimum_rows <= 0:
            raise ValueError("DistributionDiagnosticConfig.minimum_rows must be greater than 0.")


@dataclass(slots=True)
class ResidualDiagnosticConfig:
    """Controls residual-bias, heteroskedasticity, and autocorrelation checks."""

    enabled: bool = True
    heteroskedasticity_tests: bool = True
    segment_bias_analysis: bool = True
    autocorrelation_tests: bool = True
    minimum_rows: int = 30

    def validate(self) -> None:
        if self.minimum_rows <= 0:
            raise ValueError("ResidualDiagnosticConfig.minimum_rows must be greater than 0.")


@dataclass(slots=True)
class OutlierDiagnosticConfig:
    """Controls influence and outlier flagging thresholds."""

    enabled: bool = True
    zscore_threshold: float = 3.0
    leverage_multiplier: float = 2.0
    cooks_distance_multiplier: float = 4.0
    max_rows: int = 50

    def validate(self) -> None:
        if self.zscore_threshold <= 0:
            raise ValueError("OutlierDiagnosticConfig.zscore_threshold must be positive.")
        if self.leverage_multiplier <= 0:
            raise ValueError("OutlierDiagnosticConfig.leverage_multiplier must be positive.")
        if self.cooks_distance_multiplier <= 0:
            raise ValueError("OutlierDiagnosticConfig.cooks_distance_multiplier must be positive.")
        if self.max_rows <= 0:
            raise ValueError("OutlierDiagnosticConfig.max_rows must be greater than 0.")


@dataclass(slots=True)
class DependencyDiagnosticConfig:
    """Controls multicollinearity and dependency-clustering diagnostics."""

    enabled: bool = True
    clustering_correlation_threshold: float = 0.7
    maximum_features: int = 12
    condition_index_warning: float = 30.0

    def validate(self) -> None:
        if not 0 < self.clustering_correlation_threshold <= 1:
            raise ValueError(
                "DependencyDiagnosticConfig.clustering_correlation_threshold must be in (0, 1]."
            )
        if self.maximum_features <= 1:
            raise ValueError("DependencyDiagnosticConfig.maximum_features must be greater than 1.")
        if self.condition_index_warning <= 0:
            raise ValueError("DependencyDiagnosticConfig.condition_index_warning must be positive.")


@dataclass(slots=True)
class TimeSeriesDiagnosticConfig:
    """Controls the deeper econometric and time-series testing layer."""

    enabled: bool = True
    maximum_lag: int = 5
    seasonal_period: int = 4
    minimum_series_length: int = 12

    def validate(self) -> None:
        if self.maximum_lag <= 0:
            raise ValueError("TimeSeriesDiagnosticConfig.maximum_lag must be positive.")
        if self.seasonal_period <= 1:
            raise ValueError("TimeSeriesDiagnosticConfig.seasonal_period must be at least 2.")
        if self.minimum_series_length <= 0:
            raise ValueError("TimeSeriesDiagnosticConfig.minimum_series_length must be positive.")


@dataclass(slots=True)
class StructuralBreakConfig:
    """Controls structural-break and regime-shift diagnostics."""

    enabled: bool = True
    candidate_break_count: int = 3
    minimum_segment_size: int = 12
    rolling_window_fraction: float = 0.25

    def validate(self) -> None:
        if self.candidate_break_count <= 0:
            raise ValueError("StructuralBreakConfig.candidate_break_count must be positive.")
        if self.minimum_segment_size <= 2:
            raise ValueError("StructuralBreakConfig.minimum_segment_size must be at least 3.")
        if not 0 < self.rolling_window_fraction <= 0.5:
            raise ValueError("StructuralBreakConfig.rolling_window_fraction must be in (0, 0.5].")


@dataclass(slots=True)
class FeatureWorkbenchConfig:
    """Controls the constructed-feature workbench and preview outputs."""

    enabled: bool = True
    max_features: int = 12
    include_preview_statistics: bool = True
    include_target_association: bool = True

    def validate(self) -> None:
        if self.max_features <= 0:
            raise ValueError("FeatureWorkbenchConfig.max_features must be greater than 0.")


@dataclass(slots=True)
class PresetRecommendationConfig:
    """Controls preset-aligned transformation and test recommendation outputs."""

    enabled: bool = True
    include_imputation_recommendations: bool = True
    include_transformation_recommendations: bool = True
    include_test_recommendations: bool = True

    def validate(self) -> None:
        if not self.enabled:
            return


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
            "pyarrow",
        ]
    )

    def validate(self) -> None:
        package_names = [package_name.strip() for package_name in self.package_names]
        if any(not package_name for package_name in package_names):
            raise ValueError("ReproducibilityConfig.package_names cannot contain blanks.")
        if len(package_names) != len(set(package_names)):
            raise ValueError("ReproducibilityConfig.package_names must be unique.")


@dataclass(slots=True)
class PerformanceConfig:
    """Controls lightweight safeguards for large uploads, diagnostics, and reports."""

    enabled: bool = True
    large_data_mode: bool = False
    upload_warning_mb: int = 250
    dataframe_warning_rows: int = 200_000
    dataframe_warning_columns: int = 150
    ui_preview_rows: int = 50
    html_table_preview_rows: int = 12
    html_max_figures_per_section: int = 6
    html_max_tables_per_section: int = 6
    diagnostic_sample_rows: int = 20_000
    multiple_imputation_row_cap: int = 25_000
    lazy_html_figures: bool = True
    lazy_streamlit_results: bool = True
    optimize_dtypes: bool = False
    downcast_numeric: bool = True
    convert_low_cardinality_strings: bool = True
    category_max_unique_values: int = 500
    category_max_unique_ratio: float = 0.5
    convert_csv_to_parquet: bool = False
    csv_conversion_chunk_rows: int = 100_000
    large_data_training_sample_rows: int = 250_000
    large_data_score_chunk_rows: int = 100_000
    large_data_project_columns: bool = True
    large_data_auto_stage_parquet: bool = True
    memory_limit_gb: float | None = None
    memory_estimate_file_multiplier: float = 6.0
    memory_estimate_dataframe_multiplier: float = 4.0

    def validate(self) -> None:
        for field_name, value in {
            "upload_warning_mb": self.upload_warning_mb,
            "dataframe_warning_rows": self.dataframe_warning_rows,
            "dataframe_warning_columns": self.dataframe_warning_columns,
            "ui_preview_rows": self.ui_preview_rows,
            "html_table_preview_rows": self.html_table_preview_rows,
            "html_max_figures_per_section": self.html_max_figures_per_section,
            "html_max_tables_per_section": self.html_max_tables_per_section,
            "diagnostic_sample_rows": self.diagnostic_sample_rows,
            "multiple_imputation_row_cap": self.multiple_imputation_row_cap,
            "category_max_unique_values": self.category_max_unique_values,
            "csv_conversion_chunk_rows": self.csv_conversion_chunk_rows,
            "large_data_training_sample_rows": self.large_data_training_sample_rows,
            "large_data_score_chunk_rows": self.large_data_score_chunk_rows,
        }.items():
            if value <= 0:
                raise ValueError(f"PerformanceConfig.{field_name} must be greater than 0.")
        if not 0 < self.category_max_unique_ratio <= 1:
            raise ValueError("PerformanceConfig.category_max_unique_ratio must be in (0, 1].")
        if self.memory_limit_gb is not None and self.memory_limit_gb <= 0:
            raise ValueError("PerformanceConfig.memory_limit_gb must be positive when set.")
        if self.memory_estimate_file_multiplier <= 0:
            raise ValueError("PerformanceConfig.memory_estimate_file_multiplier must be positive.")
        if self.memory_estimate_dataframe_multiplier <= 0:
            raise ValueError(
                "PerformanceConfig.memory_estimate_dataframe_multiplier must be positive."
            )


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
        if self.mode == ExecutionMode.SEARCH_FEATURE_SUBSETS and (
            self.existing_model_path is not None or self.existing_config_path is not None
        ):
            raise ValueError(
                "Feature subset search does not load an existing model or prior run config."
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
    input_snapshot_parquet_file_name: str = "input_snapshot.parquet"
    predictions_file_name: str = "predictions.csv"
    predictions_parquet_file_name: str = "predictions.parquet"
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
    decision_summary_file_name: str = "decision_summary.md"
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
    code_snapshot_directory_name: str = "code_snapshot"
    include_enhanced_report_visuals: bool = True
    include_advanced_visual_analytics: bool = False
    export_individual_figure_files: bool = False
    export_input_snapshot: bool = True
    export_code_snapshot: bool = True
    export_profile: ExportProfile = ExportProfile.STANDARD
    tabular_output_format: TabularOutputFormat = TabularOutputFormat.CSV
    large_data_export_policy: LargeDataExportPolicy = LargeDataExportPolicy.FULL
    large_data_sample_rows: int = 50_000
    parquet_compression: str = "snappy"
    run_debug_trace_file_name: str = "run_debug_trace.json"

    def validate(self) -> None:
        if not self.output_root:
            raise ValueError("ArtifactConfig.output_root cannot be empty.")
        required_names = {
            "model_file_name": self.model_file_name,
            "metrics_file_name": self.metrics_file_name,
            "input_snapshot_file_name": self.input_snapshot_file_name,
            "input_snapshot_parquet_file_name": self.input_snapshot_parquet_file_name,
            "predictions_file_name": self.predictions_file_name,
            "predictions_parquet_file_name": self.predictions_parquet_file_name,
            "feature_importance_file_name": self.feature_importance_file_name,
            "backtest_file_name": self.backtest_file_name,
            "report_file_name": self.report_file_name,
            "interactive_report_file_name": self.interactive_report_file_name,
            "config_file_name": self.config_file_name,
            "decision_summary_file_name": self.decision_summary_file_name,
            "documentation_pack_file_name": self.documentation_pack_file_name,
            "validation_pack_file_name": self.validation_pack_file_name,
            "committee_report_docx_file_name": self.committee_report_docx_file_name,
            "validation_report_docx_file_name": self.validation_report_docx_file_name,
            "committee_report_pdf_file_name": self.committee_report_pdf_file_name,
            "validation_report_pdf_file_name": self.validation_report_pdf_file_name,
            "reproducibility_manifest_file_name": self.reproducibility_manifest_file_name,
            "template_workbook_file_name": self.template_workbook_file_name,
            "run_debug_trace_file_name": self.run_debug_trace_file_name,
        }
        for field_name, value in required_names.items():
            if not value.strip():
                raise ValueError(f"ArtifactConfig.{field_name} cannot be blank.")
        if self.large_data_sample_rows <= 0:
            raise ValueError("ArtifactConfig.large_data_sample_rows must be greater than 0.")
        if not self.parquet_compression.strip():
            raise ValueError("ArtifactConfig.parquet_compression cannot be blank.")


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
    subset_search: FeatureSubsetSearchConfig = field(default_factory=FeatureSubsetSearchConfig)
    feature_policy: FeaturePolicyConfig = field(default_factory=FeaturePolicyConfig)
    feature_dictionary: FeatureDictionaryConfig = field(default_factory=FeatureDictionaryConfig)
    advanced_imputation: AdvancedImputationConfig = field(default_factory=AdvancedImputationConfig)
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
    distribution_diagnostics: DistributionDiagnosticConfig = field(
        default_factory=DistributionDiagnosticConfig
    )
    residual_diagnostics: ResidualDiagnosticConfig = field(default_factory=ResidualDiagnosticConfig)
    outlier_diagnostics: OutlierDiagnosticConfig = field(default_factory=OutlierDiagnosticConfig)
    dependency_diagnostics: DependencyDiagnosticConfig = field(
        default_factory=DependencyDiagnosticConfig
    )
    time_series_diagnostics: TimeSeriesDiagnosticConfig = field(
        default_factory=TimeSeriesDiagnosticConfig
    )
    structural_breaks: StructuralBreakConfig = field(default_factory=StructuralBreakConfig)
    feature_workbench: FeatureWorkbenchConfig = field(default_factory=FeatureWorkbenchConfig)
    preset_recommendations: PresetRecommendationConfig = field(
        default_factory=PresetRecommendationConfig
    )
    credit_risk: CreditRiskDiagnosticConfig = field(default_factory=CreditRiskDiagnosticConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    cross_validation: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)

    def validate(self) -> None:
        self.schema.validate()
        self.feature_engineering.validate()
        self.target.validate()
        self.split.validate()
        self.execution.validate()
        self.model.validate(self.target.mode)
        self.comparison.validate(self.model.model_type, self.target.mode)
        self.subset_search.validate()
        self.feature_policy.validate()
        self.feature_dictionary.validate()
        self.advanced_imputation.validate()
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
        self.distribution_diagnostics.validate()
        self.residual_diagnostics.validate()
        self.outlier_diagnostics.validate()
        self.dependency_diagnostics.validate()
        self.time_series_diagnostics.validate()
        self.structural_breaks.validate()
        self.feature_workbench.validate()
        self.preset_recommendations.validate()
        self.credit_risk.validate()
        self.robustness.validate()
        self.cross_validation.validate()
        self.reproducibility.validate()
        self.performance.validate()
        self.artifacts.validate()
        if self.execution.mode == ExecutionMode.SEARCH_FEATURE_SUBSETS:
            if self.target.mode != TargetMode.BINARY:
                raise ValueError(
                    "Feature subset search is currently only supported for binary targets."
                )
            if self.model.model_type not in {
                ModelType.LOGISTIC_REGRESSION,
                ModelType.DISCRETE_TIME_HAZARD_MODEL,
                ModelType.ELASTIC_NET_LOGISTIC_REGRESSION,
                ModelType.SCORECARD_LOGISTIC_REGRESSION,
                ModelType.PROBIT_REGRESSION,
                ModelType.XGBOOST,
            }:
                raise ValueError(
                    "Feature subset search currently supports logistic, discrete-time hazard, "
                    "elastic-net logistic, scorecard logistic, probit, and XGBoost models."
                )
            if not self.subset_search.enabled:
                raise ValueError(
                    "FeatureSubsetSearchConfig.enabled must be True when using "
                    "mode='search_feature_subsets'."
                )
            if self.comparison.enabled:
                raise ValueError(
                    "Disable challenger-model comparison when running feature subset search."
                )
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

        return framework_config_to_dict(self)

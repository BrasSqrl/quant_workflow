"""Runs pre-fit suitability and assumption checks for the chosen model flow."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..base import BasePipelineStep
from ..config import DataStructure, ModelType, TargetMode
from ..context import PipelineContext

STATUS_LABELS = {
    "pass": "Pass",
    "warn": "Watch",
    "fail": "Fail",
}

CHECK_LABELS = {
    "non_null_target_rows": "Non-null target rows",
    "positive_class_rate": "Positive class rate",
    "events_per_feature": "Events per feature",
    "continuous_target_range": "Continuous target range",
    "bounded_target_unit_interval": "Bounded target range",
    "positive_loss_observations": "Positive loss observations",
    "left_censoring_share": "Left-censoring share",
    "right_censoring_share": "Right-censoring share",
    "duplicate_entity_date_pairs": "Duplicate entity-date pairs",
    "max_date_gap_days": "Maximum date gap",
    "dominant_category_share": "Dominant category share",
}

CHECK_EXPLANATIONS = {
    "non_null_target_rows": {
        "pass": "The training split meets the minimum labeled-row requirement.",
        "fail": "The training split has fewer non-missing target rows than required.",
        "why": "Too few labeled rows can make model estimates and validation metrics unstable.",
        "action": (
            "Add labeled observations, revise the split, or lower this minimum only "
            "with documented justification."
        ),
    },
    "positive_class_rate": {
        "pass": "The positive class rate is within the configured acceptable range.",
        "fail": "The positive class rate is outside the configured acceptable range.",
        "why": (
            "Extreme class imbalance can make discrimination, calibration, and "
            "threshold metrics unstable."
        ),
        "action": (
            "Review the target definition and sample window, add event or non-event "
            "observations, or adjust class-rate guardrails with documented rationale."
        ),
    },
    "events_per_feature": {
        "pass": "The training split has enough target events for the current feature count.",
        "fail": "The training split has fewer target events per feature than required.",
        "why": (
            "Low events per feature increases overfitting risk and makes coefficients "
            "less reliable."
        ),
        "action": (
            "Reduce selected features, add more event observations, change the split, "
            "use regularization, or lower the threshold with documented model-risk acceptance."
        ),
    },
    "continuous_target_range": {
        "pass": "The continuous target range was recorded for review.",
        "why": (
            "The observed range helps confirm that the selected model family matches the target."
        ),
        "action": "Confirm the range matches the business target definition.",
    },
    "bounded_target_unit_interval": {
        "pass": (
            "The target values are inside the unit interval required by the selected model family."
        ),
        "fail": (
            "The target values fall outside the unit interval required by the selected "
            "model family."
        ),
        "why": "Beta and two-stage LGD workflows assume outcomes bounded between 0 and 1.",
        "action": (
            "Correct the target scaling, choose a model family that supports the observed "
            "target range, or document why the value range is acceptable."
        ),
    },
    "positive_loss_observations": {
        "pass": "The training sample has enough positive-loss observations for the LGD stage.",
        "fail": "The training sample has too few positive-loss observations for the LGD stage.",
        "why": (
            "Two-stage LGD models need enough positive-loss cases to estimate the severity stage."
        ),
        "action": (
            "Add positive-loss observations, simplify the model, or use an alternate "
            "LGD specification."
        ),
    },
    "left_censoring_share": {
        "pass": "The target includes observations at the configured left-censoring bound.",
        "warn": "No observations appear at the configured left-censoring bound.",
        "why": (
            "A Tobit model is intended for censored outcomes; absent censoring may "
            "make OLS more appropriate."
        ),
        "action": (
            "Confirm the censoring bound, target definition, and whether Tobit is "
            "still appropriate."
        ),
    },
    "right_censoring_share": {
        "pass": "The target includes observations at the configured right-censoring bound.",
        "warn": "No observations appear at the configured right-censoring bound.",
        "why": (
            "A Tobit model is intended for censored outcomes; absent censoring may "
            "make OLS more appropriate."
        ),
        "action": (
            "Confirm the censoring bound, target definition, and whether Tobit is "
            "still appropriate."
        ),
    },
    "duplicate_entity_date_pairs": {
        "pass": "No duplicate entity-date records were found for the panel structure.",
        "fail": "Duplicate entity-date records were found in the panel structure.",
        "why": (
            "Duplicate panel keys can distort time splits, backtests, and grouped "
            "validation outputs."
        ),
        "action": (
            "Deduplicate the source data or revise the entity/date column selections "
            "before relying on results."
        ),
    },
    "max_date_gap_days": {
        "pass": "The largest date gap was recorded for time-series review.",
        "why": "Large or irregular gaps can affect temporal backtests and trend interpretation.",
        "action": "Review whether the observed gap is expected for the dataset and split design.",
    },
    "dominant_category_share": {
        "pass": "The most common category is within the configured concentration limit.",
        "fail": "The most common category exceeds the configured concentration limit.",
        "why": (
            "Highly concentrated categorical features may add little signal and can "
            "create unstable category effects."
        ),
        "action": (
            "Collapse sparse categories, remove the feature, revise the category definition, "
            "or document why the concentration is acceptable."
        ),
    },
}


class AssumptionCheckStep(BasePipelineStep):
    """
    Records model-suitability checks before fitting.

    These checks are meant to catch common development mistakes early and to
    make the run folder easier to audit later.
    """

    name = "assumption_checks"

    def run(self, context: PipelineContext) -> PipelineContext:
        config = context.config.suitability_checks
        if not config.enabled:
            return context
        if not context.split_frames or context.target_column is None:
            raise ValueError("Assumption checks require populated splits and a target column.")

        train_frame = context.split_frames.get("train")
        dataframe = context.working_data
        if train_frame is None or dataframe is None:
            raise ValueError("Assumption checks require train and working dataframes.")

        rows: list[dict[str, Any]] = []
        target_column = context.target_column
        labels_available = bool(context.metadata.get("labels_available", False))
        train_target = (
            train_frame[target_column] if target_column in train_frame.columns else pd.Series()
        )

        non_null_target = train_target.dropna()
        rows.append(
            self._row(
                check_name="non_null_target_rows",
                subject="train_split",
                observed_value=int(len(non_null_target)),
                threshold=config.min_non_null_target_rows,
                passed=len(non_null_target) >= config.min_non_null_target_rows,
            )
        )

        if labels_available and len(non_null_target) > 0:
            rows.extend(self._build_target_checks(context, non_null_target))
        rows.extend(self._build_structure_checks(context, dataframe))
        rows.extend(self._build_feature_distribution_checks(context, train_frame))

        table = pd.DataFrame(rows)
        status_rank = table["status"].map({"fail": 0, "warn": 1, "pass": 2}).fillna(3)
        table = (
            table.assign(_status_rank=status_rank)
            .sort_values(
                ["_status_rank", "check_name", "subject"],
                ascending=[True, True, True],
                kind="stable",
            )
            .drop(columns=["_status_rank"])
        )
        context.diagnostics_tables["assumption_checks"] = table

        failures = table.loc[table["status"] == "fail"]
        warnings = table.loc[table["status"] == "warn"]
        context.metadata["assumption_check_summary"] = {
            "row_count": int(len(table)),
            "fail_count": int(len(failures)),
            "warn_count": int(len(warnings)),
        }
        if not warnings.empty:
            context.warn(f"Suitability checks recorded {len(warnings)} warning conditions.")
        if not failures.empty:
            context.warn(self._failure_warning_message(failures))
            if config.error_on_failure:
                preview = ", ".join(
                    failures.head(5).apply(
                        lambda row: f"{row['check_name']}:{row['subject']}",
                        axis=1,
                    )
                )
                raise ValueError(f"Suitability checks failed: {preview}.")

        return context

    def _build_target_checks(
        self,
        context: PipelineContext,
        non_null_target: pd.Series,
    ) -> list[dict[str, Any]]:
        config = context.config.suitability_checks
        target_mode = context.config.target.mode
        rows: list[dict[str, Any]] = []

        if target_mode == TargetMode.BINARY:
            positive_rate = float(pd.to_numeric(non_null_target, errors="coerce").mean())
            positive_events = int(pd.to_numeric(non_null_target, errors="coerce").sum())
            feature_count = max(len(context.feature_columns), 1)
            events_per_feature = positive_events / feature_count
            rows.append(
                self._row(
                    check_name="positive_class_rate",
                    subject="train_target",
                    observed_value=positive_rate,
                    threshold=f"[{config.min_class_rate}, {config.max_class_rate}]",
                    passed=(config.min_class_rate is None or positive_rate >= config.min_class_rate)
                    and (config.max_class_rate is None or positive_rate <= config.max_class_rate),
                )
            )
            if config.min_events_per_feature is not None:
                rows.append(
                    self._row(
                        check_name="events_per_feature",
                        subject="train_target",
                        observed_value=events_per_feature,
                        threshold=config.min_events_per_feature,
                        passed=events_per_feature >= config.min_events_per_feature,
                    )
                )
            return rows

        numeric_target = pd.to_numeric(non_null_target, errors="coerce")
        target_min = float(numeric_target.min())
        target_max = float(numeric_target.max())
        rows.append(
            self._row(
                check_name="continuous_target_range",
                subject="train_target",
                observed_value=f"[{target_min:.4f}, {target_max:.4f}]",
                threshold="observed",
                passed=True,
            )
        )

        model_type = context.config.model.model_type
        if model_type in {ModelType.BETA_REGRESSION, ModelType.TWO_STAGE_LGD_MODEL}:
            rows.append(
                self._row(
                    check_name="bounded_target_unit_interval",
                    subject="train_target",
                    observed_value=f"[{target_min:.4f}, {target_max:.4f}]",
                    threshold="[0, 1]",
                    passed=0.0 <= target_min and target_max <= 1.0,
                )
            )
        if model_type == ModelType.TWO_STAGE_LGD_MODEL:
            positive_loss_count = int(
                (numeric_target > context.config.model.lgd_positive_threshold).sum()
            )
            rows.append(
                self._row(
                    check_name="positive_loss_observations",
                    subject="train_target",
                    observed_value=positive_loss_count,
                    threshold=10,
                    passed=positive_loss_count >= 10,
                )
            )
        if model_type == ModelType.TOBIT_REGRESSION:
            lower = context.config.model.tobit_left_censoring
            upper = context.config.model.tobit_right_censoring
            if lower is not None:
                lower_share = float((numeric_target <= lower).mean())
                rows.append(
                    self._row(
                        check_name="left_censoring_share",
                        subject="train_target",
                        observed_value=lower_share,
                        threshold="> 0 preferred",
                        passed=lower_share > 0,
                        status_if_failed="warn",
                    )
                )
            if upper is not None:
                upper_share = float((numeric_target >= upper).mean())
                rows.append(
                    self._row(
                        check_name="right_censoring_share",
                        subject="train_target",
                        observed_value=upper_share,
                        threshold="> 0 preferred",
                        passed=upper_share > 0,
                        status_if_failed="warn",
                    )
                )
        return rows

    def _build_structure_checks(
        self,
        context: PipelineContext,
        dataframe: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        split_config = context.config.split
        date_column = split_config.date_column
        entity_column = split_config.entity_column

        if (
            split_config.data_structure == DataStructure.PANEL
            and date_column
            and entity_column
            and date_column in dataframe.columns
            and entity_column in dataframe.columns
        ):
            duplicate_pairs = int(dataframe.duplicated([entity_column, date_column]).sum())
            rows.append(
                self._row(
                    check_name="duplicate_entity_date_pairs",
                    subject=f"{entity_column}+{date_column}",
                    observed_value=duplicate_pairs,
                    threshold=0,
                    passed=duplicate_pairs == 0,
                )
            )

        if (
            split_config.data_structure in {DataStructure.TIME_SERIES, DataStructure.PANEL}
            and date_column
            and date_column in dataframe.columns
        ):
            ordered = dataframe.copy(deep=True)
            ordered[date_column] = pd.to_datetime(ordered[date_column], errors="coerce")
            if entity_column and entity_column in ordered.columns:
                gap_series = (
                    ordered.sort_values([entity_column, date_column])
                    .groupby(entity_column, dropna=False)[date_column]
                    .diff()
                )
            else:
                gap_series = ordered.sort_values(date_column)[date_column].diff()
            gap_days = gap_series.dropna().dt.days
            if not gap_days.empty:
                rows.append(
                    self._row(
                        check_name="max_date_gap_days",
                        subject=date_column,
                        observed_value=int(gap_days.max()),
                        threshold="review",
                        passed=True,
                    )
                )
        return rows

    def _build_feature_distribution_checks(
        self,
        context: PipelineContext,
        train_frame: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        config = context.config.suitability_checks
        rows: list[dict[str, Any]] = []
        if config.max_dominant_category_share is None:
            return rows

        for feature_name in context.categorical_features:
            if feature_name not in train_frame.columns:
                continue
            distribution = (
                train_frame[feature_name]
                .astype("object")
                .fillna("Missing")
                .astype(str)
                .value_counts(normalize=True)
            )
            if distribution.empty:
                continue
            dominant_share = float(distribution.iloc[0])
            dominant_value = str(distribution.index[0])
            rows.append(
                self._row(
                    check_name="dominant_category_share",
                    subject=feature_name,
                    observed_value=dominant_share,
                    threshold=config.max_dominant_category_share,
                    passed=dominant_share <= config.max_dominant_category_share,
                    details=f"Top category: {dominant_value}",
                )
            )
        return rows

    def _row(
        self,
        *,
        check_name: str,
        subject: str,
        observed_value: Any,
        threshold: Any,
        passed: bool,
        details: str = "",
        status_if_failed: str = "fail",
    ) -> dict[str, Any]:
        status = "pass" if passed else status_if_failed
        return {
            "check_name": check_name,
            "check_label": CHECK_LABELS.get(check_name, check_name.replace("_", " ").title()),
            "subject": subject,
            "status": status,
            "status_label": STATUS_LABELS.get(status, status.title()),
            "observed_value": observed_value,
            "threshold": threshold,
            "details": details,
            "interpretation": self._interpretation(check_name, status),
            "why_it_matters": self._why_it_matters(check_name),
            "recommended_action": self._recommended_action(check_name),
        }

    def _interpretation(self, check_name: str, status: str) -> str:
        explanation = CHECK_EXPLANATIONS.get(check_name, {})
        if status in explanation:
            return explanation[status]
        if status == "warn":
            return "This check produced a watch condition that should be reviewed before model use."
        if status == "fail":
            return "This check failed against the configured suitability threshold."
        return "This check passed against the configured suitability threshold."

    def _why_it_matters(self, check_name: str) -> str:
        return CHECK_EXPLANATIONS.get(check_name, {}).get(
            "why",
            (
                "Suitability checks help identify data or configuration conditions "
                "that can weaken model evidence."
            ),
        )

    def _recommended_action(self, check_name: str) -> str:
        return CHECK_EXPLANATIONS.get(check_name, {}).get(
            "action",
            "Review the observed value, threshold, source data, and model configuration.",
        )

    def _failure_warning_message(self, failures: pd.DataFrame) -> str:
        details = []
        for _, row in failures.head(3).iterrows():
            details.append(
                f"{row['check_label']} on {row['subject']} failed "
                f"({self._format_value(row['observed_value'])} observed vs "
                f"{self._format_value(row['threshold'])} required). "
                f"{row['recommended_action']}"
            )
        suffix = ""
        if len(failures) > len(details):
            suffix = f" {len(failures) - len(details)} additional suitability failures recorded."
        return "Suitability check failure: " + " ".join(details) + suffix

    def _format_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

"""Selects a development feature set using train-split screening rules."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import roc_auc_score

from ..base import BasePipelineStep
from ..config import ExecutionMode, FeatureReviewDecisionType, TargetMode
from ..context import PipelineContext


class VariableSelectionStep(BasePipelineStep):
    """
    Applies a narrow, documented feature-screening workflow on the train split.

    The goal is not automated feature search. It is a transparent development
    screen that keeps the selected feature set reproducible and exportable.
    """

    name = "variable_selection"

    def run(self, context: PipelineContext) -> PipelineContext:
        config = context.config.variable_selection
        manual_review = context.config.manual_review
        if not config.enabled:
            if manual_review.enabled and manual_review.feature_decisions:
                selection_table = self._build_manual_review_only_table(context)
                selection_table, selected_features = self._apply_manual_review(
                    context,
                    selection_table,
                    set(context.feature_columns),
                )
                self._apply_selected_features(context, selected_features)
                context.diagnostics_tables["variable_selection"] = selection_table
                self._record_manual_review_outputs(context)
                return context
            return context
        if context.config.execution.mode == ExecutionMode.SCORE_EXISTING_MODEL:
            context.warn(
                "Variable selection was skipped because existing-model scoring must preserve "
                "the trained model feature contract."
            )
            return context
        if not context.feature_columns or context.target_column is None:
            raise ValueError("Variable selection requires resolved feature columns and a target.")

        train_frame = context.split_frames.get("train")
        if train_frame is None:
            raise ValueError("Variable selection requires an imputed training split.")

        y_train = train_frame[context.target_column]
        labels_available = bool(context.metadata.get("labels_available", False))
        if not labels_available or y_train.nunique(dropna=True) < 2:
            context.warn(
                "Variable selection was skipped because the training split does not contain "
                "enough target variation."
            )
            return context

        selection_table = self._build_selection_table(context, train_frame, y_train)
        selected_features = self._resolve_selected_features(context, selection_table)
        selection_table, selected_features = self._apply_manual_review(
            context,
            selection_table,
            selected_features,
        )
        if not selected_features:
            raise ValueError(
                "Variable selection removed every candidate feature. Relax the selection "
                "thresholds or locked exclusions."
            )

        self._apply_selected_features(context, selected_features)
        context.diagnostics_tables["variable_selection"] = selection_table
        self._record_manual_review_outputs(context)
        context.metadata["variable_selection_summary"] = {
            "enabled": True,
            "selected_feature_count": len(context.feature_columns),
            "selected_features": context.feature_columns,
        }
        context.log(
            "Applied train-split variable selection and retained "
            f"{len(context.feature_columns)} features."
        )
        return context

    def _build_manual_review_only_table(self, context: PipelineContext) -> pd.DataFrame:
        rows = []
        for feature_name in context.feature_columns:
            rows.append(
                {
                    "feature_name": feature_name,
                    "feature_type": (
                        "numeric" if feature_name in context.numeric_features else "categorical"
                    ),
                    "univariate_score": None,
                    "selection_status": "selected",
                    "selection_reason": "manual_review_base",
                    "selected": True,
                }
            )
        return pd.DataFrame(rows)

    def _build_selection_table(
        self,
        context: PipelineContext,
        train_frame: pd.DataFrame,
        y_train: pd.Series,
    ) -> pd.DataFrame:
        config = context.config.variable_selection
        target_mode = context.config.target.mode
        rows: list[dict[str, Any]] = []

        for feature_name in context.feature_columns:
            series = train_frame[feature_name]
            feature_type = (
                "numeric" if feature_name in context.numeric_features else "categorical"
            )
            univariate_score = self._compute_univariate_score(
                series=series,
                y_train=y_train,
                feature_type=feature_type,
                target_mode=target_mode,
            )
            rows.append(
                {
                    "feature_name": feature_name,
                    "feature_type": feature_type,
                    "univariate_score": univariate_score,
                    "selection_status": "candidate",
                    "selection_reason": "candidate_pool",
                }
            )

        table = pd.DataFrame(rows).sort_values(
            ["univariate_score", "feature_name"],
            ascending=[False, True],
            kind="stable",
        )
        if table.empty:
            return table

        table = table.reset_index(drop=True)
        status_map = {
            row["feature_name"]: {
                "selection_status": row["selection_status"],
                "selection_reason": row["selection_reason"],
            }
            for _, row in table.iterrows()
        }

        locked_includes = list(dict.fromkeys(config.locked_include_features))
        locked_excludes = set(config.locked_exclude_features)
        available_features = set(table["feature_name"].tolist())
        for feature_name in locked_includes:
            if feature_name not in available_features:
                context.warn(
                    f"Locked include feature '{feature_name}' was not available after "
                    "feature engineering."
                )
        for feature_name in locked_excludes:
            if feature_name not in available_features:
                context.warn(
                    f"Locked exclude feature '{feature_name}' was not available after "
                    "feature engineering."
                )
                continue
            status_map[feature_name] = {
                "selection_status": "excluded",
                "selection_reason": "locked_exclude",
            }

        for feature_name in table["feature_name"]:
            if feature_name in locked_excludes:
                continue
            feature_rows = table.loc[table["feature_name"] == feature_name, "univariate_score"]
            score = float(feature_rows.iloc[0])
            if (
                config.min_univariate_score is not None
                and score < config.min_univariate_score
                and feature_name not in locked_includes
            ):
                status_map[feature_name] = {
                    "selection_status": "excluded",
                    "selection_reason": "below_univariate_threshold",
                }

        self._apply_correlation_filter(context, train_frame, table, status_map, locked_includes)
        self._apply_max_feature_cap(table, status_map, locked_includes, context)
        if context.config.variable_selection.max_features is None:
            for feature_name in table["feature_name"]:
                if status_map[feature_name]["selection_status"] != "excluded":
                    status_map[feature_name] = {
                        "selection_status": "selected",
                        "selection_reason": "screen_passed",
                    }

        selected_by_rule = {
            feature_name
            for feature_name, payload in status_map.items()
            if payload["selection_status"] == "selected"
        }
        for feature_name in locked_includes:
            if feature_name not in available_features or feature_name in locked_excludes:
                continue
            selected_by_rule.add(feature_name)
            status_map[feature_name] = {
                "selection_status": "selected",
                "selection_reason": "locked_include",
            }

        for feature_name in table["feature_name"]:
            is_candidate_reason = (
                status_map[feature_name]["selection_reason"] == "candidate_pool"
            )
            if feature_name in selected_by_rule and is_candidate_reason:
                status_map[feature_name] = {
                    "selection_status": "selected",
                    "selection_reason": "screen_passed",
                }
            elif (
                feature_name not in selected_by_rule
                and status_map[feature_name]["selection_status"] == "candidate"
            ):
                status_map[feature_name] = {
                    "selection_status": "excluded",
                    "selection_reason": "screen_not_selected",
                }

        enriched = table.copy(deep=True)
        enriched["selection_status"] = enriched["feature_name"].map(
            lambda feature_name: status_map[feature_name]["selection_status"]
        )
        enriched["selection_reason"] = enriched["feature_name"].map(
            lambda feature_name: status_map[feature_name]["selection_reason"]
        )
        enriched["selected"] = enriched["selection_status"].eq("selected")
        return enriched.sort_values(
            ["selected", "univariate_score", "feature_name"],
            ascending=[False, False, True],
            kind="stable",
        ).reset_index(drop=True)

    def _compute_univariate_score(
        self,
        *,
        series: pd.Series,
        y_train: pd.Series,
        feature_type: str,
        target_mode: TargetMode,
    ) -> float:
        predictor = self._encode_feature(series=series, y_train=y_train, feature_type=feature_type)
        if predictor.nunique(dropna=True) < 2:
            return 0.0
        if target_mode == TargetMode.BINARY:
            try:
                auc_value = float(
                    roc_auc_score(y_train.astype(int), predictor.to_numpy(dtype=float))
                )
            except ValueError:
                return 0.0
            return float(abs(auc_value - 0.5) * 2.0)

        target_array = pd.to_numeric(y_train, errors="coerce")
        correlation = predictor.corr(target_array)
        if pd.isna(correlation):
            return 0.0
        return float(abs(correlation))

    def _encode_feature(
        self,
        *,
        series: pd.Series,
        y_train: pd.Series,
        feature_type: str,
    ) -> pd.Series:
        if feature_type == "numeric":
            numeric_series = pd.to_numeric(series, errors="coerce")
            return numeric_series.fillna(numeric_series.median())

        bucket = series.astype("object").fillna("Missing").astype(str)
        target_numeric = pd.to_numeric(y_train, errors="coerce")
        mean_map = target_numeric.groupby(bucket, dropna=False).mean().to_dict()
        fallback = float(target_numeric.mean())
        return bucket.map(mean_map).fillna(fallback).astype(float)

    def _apply_correlation_filter(
        self,
        context: PipelineContext,
        train_frame: pd.DataFrame,
        selection_table: pd.DataFrame,
        status_map: dict[str, dict[str, str]],
        locked_includes: list[str],
    ) -> None:
        threshold = context.config.variable_selection.correlation_threshold
        if threshold is None or len(context.numeric_features) < 2:
            return

        eligible_features = [
            feature_name
            for feature_name in context.numeric_features
            if status_map.get(feature_name, {}).get("selection_status") != "excluded"
        ]
        if len(eligible_features) < 2:
            return

        correlation_frame = train_frame[eligible_features].copy(deep=True)
        correlation_frame = correlation_frame.apply(pd.to_numeric, errors="coerce")
        correlation_frame = correlation_frame.fillna(correlation_frame.median(numeric_only=True))
        correlation_matrix = correlation_frame.corr().abs()
        ranked_features = selection_table.set_index("feature_name")["univariate_score"].to_dict()
        kept_features: list[str] = []
        for feature_name in selection_table["feature_name"]:
            if feature_name not in eligible_features:
                continue
            if feature_name in locked_includes:
                kept_features.append(feature_name)
                continue
            correlated_with_kept = [
                kept_feature
                for kept_feature in kept_features
                if correlation_matrix.loc[feature_name, kept_feature] >= threshold
            ]
            if correlated_with_kept:
                best_kept = max(
                    correlated_with_kept,
                    key=lambda candidate: ranked_features.get(candidate, 0.0),
                )
                status_map[feature_name] = {
                    "selection_status": "excluded",
                    "selection_reason": f"correlated_with:{best_kept}",
                }
                continue
            kept_features.append(feature_name)

    def _apply_max_feature_cap(
        self,
        selection_table: pd.DataFrame,
        status_map: dict[str, dict[str, str]],
        locked_includes: list[str],
        context: PipelineContext,
    ) -> None:
        max_features = context.config.variable_selection.max_features
        if max_features is None:
            return

        selected_features = [
            feature_name
            for feature_name in selection_table["feature_name"]
            if status_map[feature_name]["selection_status"] != "excluded"
        ]
        if len(selected_features) <= max_features:
            for feature_name in selected_features:
                if status_map[feature_name]["selection_status"] != "selected":
                    status_map[feature_name] = {
                        "selection_status": "selected",
                        "selection_reason": "screen_passed",
                    }
            return

        locked_feature_set = {
            feature_name
            for feature_name in locked_includes
            if feature_name in selected_features
        }
        retained_features = list(locked_feature_set)
        for feature_name in selection_table["feature_name"]:
            if feature_name not in selected_features or feature_name in locked_feature_set:
                continue
            if len(retained_features) >= max_features:
                status_map[feature_name] = {
                    "selection_status": "excluded",
                    "selection_reason": "max_feature_cap",
                }
                continue
            retained_features.append(feature_name)
            status_map[feature_name] = {
                "selection_status": "selected",
                "selection_reason": "screen_passed",
            }

        for feature_name in locked_feature_set:
            status_map[feature_name] = {
                "selection_status": "selected",
                "selection_reason": "locked_include",
            }

    def _resolve_selected_features(
        self,
        context: PipelineContext,
        selection_table: pd.DataFrame,
    ) -> set[str]:
        if selection_table.empty:
            return set(context.feature_columns)
        selected_features = set(
            selection_table.loc[selection_table["selected"], "feature_name"].tolist()
        )
        locked_includes = {
            feature_name
            for feature_name in context.config.variable_selection.locked_include_features
            if feature_name in selection_table["feature_name"].tolist()
        }
        return selected_features | locked_includes

    def _apply_manual_review(
        self,
        context: PipelineContext,
        selection_table: pd.DataFrame,
        selected_features: set[str],
    ) -> tuple[pd.DataFrame, set[str]]:
        manual_review = context.config.manual_review
        if selection_table.empty:
            return selection_table, selected_features

        reviewed = selection_table.copy(deep=True)
        reviewed["review_decision"] = ""
        reviewed["review_rationale"] = ""
        decision_map = {
            decision.feature_name: decision for decision in manual_review.feature_decisions
        }

        for feature_name, decision in decision_map.items():
            feature_mask = reviewed["feature_name"] == feature_name
            if not feature_mask.any():
                context.warn(
                    f"Manual review decision references unavailable feature '{feature_name}'."
                )
                continue
            reviewed.loc[feature_mask, "review_decision"] = decision.decision.value
            reviewed.loc[feature_mask, "review_rationale"] = decision.rationale
            if decision.decision in {
                FeatureReviewDecisionType.APPROVE,
                FeatureReviewDecisionType.FORCE_INCLUDE,
            }:
                selected_features.add(feature_name)
                reviewed.loc[feature_mask, "selection_status"] = "selected"
                reviewed.loc[feature_mask, "selection_reason"] = (
                    "manual_approve"
                    if decision.decision == FeatureReviewDecisionType.APPROVE
                    else "manual_force_include"
                )
            else:
                selected_features.discard(feature_name)
                reviewed.loc[feature_mask, "selection_status"] = "excluded"
                reviewed.loc[feature_mask, "selection_reason"] = (
                    "manual_reject"
                    if decision.decision == FeatureReviewDecisionType.REJECT
                    else "manual_force_exclude"
                )

        reviewed["selected"] = reviewed["feature_name"].isin(selected_features)
        if manual_review.enabled and manual_review.require_review_complete:
            missing_reviews = reviewed.loc[
                reviewed["review_decision"] == "",
                "feature_name",
            ].tolist()
            if missing_reviews:
                preview = ", ".join(missing_reviews[:10])
                raise ValueError(
                    "Manual review is marked as required, but some features do not have a "
                    f"review decision: {preview}."
                )
        return reviewed.sort_values(
            ["selected", "feature_name"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True), selected_features

    def _apply_selected_features(
        self,
        context: PipelineContext,
        selected_features: set[str],
    ) -> None:
        context.feature_columns = [
            feature_name
            for feature_name in context.feature_columns
            if feature_name in selected_features
        ]
        context.numeric_features = [
            feature_name
            for feature_name in context.numeric_features
            if feature_name in selected_features
        ]
        context.categorical_features = [
            feature_name
            for feature_name in context.categorical_features
            if feature_name in selected_features
        ]

    def _record_manual_review_outputs(self, context: PipelineContext) -> None:
        manual_review = context.config.manual_review
        if not manual_review.enabled and not manual_review.feature_decisions:
            return
        context.diagnostics_tables["manual_review_feature_decisions"] = pd.DataFrame(
            [
                {
                    "feature_name": decision.feature_name,
                    "decision": decision.decision.value,
                    "rationale": decision.rationale,
                    "reviewer_name": manual_review.reviewer_name,
                }
                for decision in manual_review.feature_decisions
            ]
        )

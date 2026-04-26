"""Chunked full-file scoring for file-backed Large Data Mode runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..base import BasePipelineStep
from ..config import ExecutionMode, TargetMode
from ..context import PipelineContext
from ..export_layout import build_export_path_layout
from ..large_data import iter_dataset_batches
from .cleaning import CleaningStep
from .feature_engineering import FeatureEngineeringStep
from .imputation import ImputationRule, ImputationStep
from .schema import SchemaManagementStep
from .target import TargetConstructionStep
from .transformations import ResolvedTransformation, TransformationStep


class LargeDataFullScoringStep(BasePipelineStep):
    """Scores the full file in chunks after the sample-development model is fitted."""

    name = "large_data_full_scoring"

    def run(self, context: PipelineContext) -> PipelineContext:
        if not context.config.performance.large_data_mode:
            return context
        if context.config.execution.mode == ExecutionMode.SEARCH_FEATURE_SUBSETS:
            context.warn(
                "Large Data Mode full-file scoring is skipped for feature subset search runs."
            )
            return context
        if context.large_data_handle is None:
            return context
        if context.model is None:
            raise ValueError("Large-data full scoring requires a fitted or loaded model.")
        if not context.feature_columns:
            raise ValueError("Large-data full scoring requires resolved model features.")

        output_root = context.config.artifacts.output_root / context.run_id
        layout = build_export_path_layout(context.config.artifacts, output_root)
        scoring_dir = layout.data_dir / "full_data_scoring"
        metadata_dir = layout.metadata_dir / "large_data"
        scoring_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        predictions_path = scoring_dir / "predictions.parquet"
        progress_path = metadata_dir / "large_data_full_scoring_progress.json"
        projected_columns = context.metadata.get("large_data_sample", {}).get("projected_columns")
        if not isinstance(projected_columns, list):
            projected_columns = None

        writer = _ChunkedParquetWriter(
            predictions_path,
            compression=context.config.artifacts.parquet_compression,
        )
        summary = _RunningScoreSummary(target_mode=context.config.target.mode)
        row_count = 0
        chunk_count = 0
        imputation_rules = context.metadata.get("imputation_rule_objects", [])
        transformation_objects = context.metadata.get("resolved_transformation_objects", [])

        try:
            for chunk in iter_dataset_batches(
                context.large_data_handle,
                batch_rows=context.config.performance.large_data_score_chunk_rows,
                columns=projected_columns,
            ):
                chunk_count += 1
                try:
                    scored_chunk = self._score_chunk(
                        context=context,
                        chunk=chunk,
                        chunk_id=chunk_count,
                        imputation_rules=imputation_rules,
                        transformation_objects=transformation_objects,
                    )
                    row_count += int(len(scored_chunk))
                    summary.update(scored_chunk, context)
                    writer.write(scored_chunk)
                    self._write_progress(
                        progress_path=progress_path,
                        chunk_count=chunk_count,
                        row_count=row_count,
                        predictions_path=predictions_path,
                    )
                except Exception as exc:
                    self._write_progress(
                        progress_path=progress_path,
                        chunk_count=chunk_count,
                        row_count=row_count,
                        predictions_path=predictions_path,
                        status="failed",
                        error_message=str(exc),
                    )
                    raise RuntimeError(
                        "Large-data full scoring failed while processing chunk "
                        f"{chunk_count}. Review {progress_path} for the last completed "
                        "row count and chunk status."
                    ) from exc
        finally:
            writer.close()

        self._write_progress(
            progress_path=progress_path,
            chunk_count=chunk_count,
            row_count=row_count,
            predictions_path=predictions_path,
            status="completed",
        )
        summary_table = summary.to_frame(row_count=row_count, chunk_count=chunk_count)
        context.diagnostics_tables["large_data_full_scoring_summary"] = summary_table
        score_distribution = summary.score_distribution_frame()
        if not score_distribution.empty:
            context.diagnostics_tables["large_data_full_score_distribution"] = score_distribution

        metadata = {
            "enabled": True,
            "source_path": str(context.large_data_handle.path),
            "active_path": str(context.large_data_handle.active_path),
            "predictions_path": str(predictions_path),
            "progress_path": str(progress_path),
            "scoring_directory": str(scoring_dir),
            "metadata_directory": str(metadata_dir),
            "chunk_rows": int(context.config.performance.large_data_score_chunk_rows),
            "chunk_count": int(chunk_count),
            "row_count": int(row_count),
            "projected_columns": projected_columns,
        }
        metadata_path = metadata_dir / "large_data_full_scoring.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, default=str),
            encoding="utf-8",
        )
        context.metadata["large_data_full_scoring"] = metadata
        context.artifacts["full_data_scoring_dir"] = scoring_dir
        context.artifacts["full_data_predictions"] = predictions_path
        context.artifacts["large_data_metadata_dir"] = metadata_dir
        context.artifacts["large_data_full_scoring_progress"] = progress_path
        context.artifacts["large_data_full_scoring_metadata"] = metadata_path
        return context

    def _write_progress(
        self,
        *,
        progress_path: Path,
        chunk_count: int,
        row_count: int,
        predictions_path: Path,
        status: str = "running",
        error_message: str = "",
    ) -> None:
        progress = {
            "status": status,
            "completed_chunks": int(chunk_count),
            "completed_rows": int(row_count),
            "predictions_path": str(predictions_path),
            "error_message": error_message,
        }
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

    def _score_chunk(
        self,
        *,
        context: PipelineContext,
        chunk: pd.DataFrame,
        chunk_id: int,
        imputation_rules: Any,
        transformation_objects: Any,
    ) -> pd.DataFrame:
        temp_context = PipelineContext(
            config=context.config,
            run_id=context.run_id,
            raw_input=chunk,
            raw_data=chunk.copy(deep=True),
            working_data=chunk.copy(deep=True),
        )
        temp_context.metadata["labels_available"] = context.metadata.get(
            "labels_available",
            True,
        )

        for step in (
            SchemaManagementStep(),
            TargetConstructionStep(),
            CleaningStep(),
            FeatureEngineeringStep(),
        ):
            temp_context = step(temp_context)

        if isinstance(imputation_rules, list):
            temp_context.working_data = ImputationStep().apply_rules_to_frame(
                context=temp_context,
                frame=temp_context.working_data,
                rules=[rule for rule in imputation_rules if isinstance(rule, ImputationRule)],
            )
            generated = context.metadata.get("generated_missing_indicator_columns", [])
            if isinstance(generated, list):
                temp_context.feature_columns = list(
                    dict.fromkeys([*temp_context.feature_columns, *generated])
                )
                temp_context.numeric_features = list(
                    dict.fromkeys([*temp_context.numeric_features, *generated])
                )

        if isinstance(transformation_objects, list):
            transformer = TransformationStep()
            for resolved in transformation_objects:
                if isinstance(resolved, ResolvedTransformation):
                    temp_context.working_data = transformer._apply_transformation(
                        temp_context.working_data,
                        resolved,
                        context=temp_context,
                    )

        missing_features = [
            feature_name
            for feature_name in context.feature_columns
            if feature_name not in temp_context.working_data.columns
        ]
        if missing_features:
            raise ValueError(
                "Large-data chunk preprocessing did not produce required model features: "
                + ", ".join(missing_features[:20])
            )

        feature_frame = temp_context.working_data.loc[:, context.feature_columns]
        scored = temp_context.working_data.copy(deep=True).reset_index(drop=True)
        scored["large_data_chunk_id"] = chunk_id
        if context.config.target.mode == TargetMode.BINARY:
            probability = np.asarray(context.model.predict_score(feature_frame))
            predicted_class = np.asarray(
                context.model.predict_class(feature_frame, context.config.model.threshold)
            )
            scored["predicted_probability"] = probability
            scored["predicted_class"] = predicted_class
        else:
            scored["predicted_value"] = np.asarray(context.model.predict_score(feature_frame))
        for column_name, values in context.model.get_prediction_outputs(feature_frame).items():
            scored[column_name] = values
        return scored


class _RunningScoreSummary:
    def __init__(self, target_mode: TargetMode) -> None:
        self.target_mode = target_mode
        self.prediction_sum = 0.0
        self.prediction_sum_sq = 0.0
        self.prediction_count = 0
        self.predicted_positive_sum = 0
        self.target_sum = 0.0
        self.target_count = 0
        self.score_bins = np.zeros(10, dtype=int)

    def update(self, scored_chunk: pd.DataFrame, context: PipelineContext) -> None:
        score_column = (
            "predicted_probability" if self.target_mode == TargetMode.BINARY else "predicted_value"
        )
        scores = pd.to_numeric(scored_chunk[score_column], errors="coerce").dropna()
        self.prediction_sum += float(scores.sum())
        self.prediction_sum_sq += float((scores**2).sum())
        self.prediction_count += int(len(scores))
        if self.target_mode == TargetMode.BINARY and "predicted_class" in scored_chunk.columns:
            self.predicted_positive_sum += int(scored_chunk["predicted_class"].sum())
            clipped = scores.clip(0, 0.999999)
            bin_ids = np.floor(clipped.to_numpy(dtype=float) * 10).astype(int)
            for bin_id in bin_ids:
                self.score_bins[int(bin_id)] += 1
        if context.target_column and context.target_column in scored_chunk.columns:
            targets = pd.to_numeric(scored_chunk[context.target_column], errors="coerce").dropna()
            self.target_sum += float(targets.sum())
            self.target_count += int(len(targets))

    def to_frame(self, *, row_count: int, chunk_count: int) -> pd.DataFrame:
        average_prediction = (
            self.prediction_sum / self.prediction_count if self.prediction_count else None
        )
        prediction_std = None
        if self.prediction_count:
            mean_value = self.prediction_sum / self.prediction_count
            variance = max(0.0, self.prediction_sum_sq / self.prediction_count - mean_value**2)
            prediction_std = float(np.sqrt(variance))
        row = {
            "row_count": int(row_count),
            "chunk_count": int(chunk_count),
            "score_count": int(self.prediction_count),
            "average_prediction": average_prediction,
            "prediction_std": prediction_std,
            "target_count": int(self.target_count),
            "average_target": self.target_sum / self.target_count if self.target_count else None,
        }
        if self.target_mode == TargetMode.BINARY:
            row["predicted_positive_rate"] = (
                self.predicted_positive_sum / self.prediction_count
                if self.prediction_count
                else None
            )
        return pd.DataFrame([row])

    def score_distribution_frame(self) -> pd.DataFrame:
        if self.target_mode != TargetMode.BINARY:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "score_bin": f"{index / 10:.1f}-{(index + 1) / 10:.1f}",
                    "observation_count": int(count),
                }
                for index, count in enumerate(self.score_bins)
            ]
        )


class _ChunkedParquetWriter:
    def __init__(self, path: Path, *, compression: str) -> None:
        self.path = path
        self.compression = compression
        self._writer = None
        self._schema = None

    def write(self, dataframe: pd.DataFrame) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("Chunked full-data scoring export requires `pyarrow`.") from exc

        safe_frame = self._safe_frame(dataframe)
        if self._schema is not None:
            safe_frame = self._coerce_frame_to_schema(safe_frame, pa)
        table = pa.Table.from_pandas(safe_frame, preserve_index=False)
        if self._writer is None:
            self._schema = table.schema
            self._writer = pq.ParquetWriter(
                self.path,
                self._schema,
                compression=self.compression,
            )
        else:
            table = table.cast(self._schema, safe=False)
        self._writer.write_table(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()

    def _coerce_frame_to_schema(self, dataframe: pd.DataFrame, pa: Any) -> pd.DataFrame:
        if self._schema is None:
            return dataframe
        coerced = pd.DataFrame(index=dataframe.index)
        for field in self._schema:
            if field.name in dataframe.columns:
                series = dataframe[field.name]
            else:
                series = pd.Series([None] * len(dataframe), index=dataframe.index)

            if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
                coerced[field.name] = pd.to_numeric(series, errors="coerce")
            elif pa.types.is_boolean(field.type):
                coerced[field.name] = series.astype("boolean")
            elif pa.types.is_timestamp(field.type) or pa.types.is_date(field.type):
                coerced[field.name] = pd.to_datetime(series, errors="coerce")
            elif pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
                coerced[field.name] = series.map(self._safe_value).astype("string")
            else:
                coerced[field.name] = series
        return coerced

    def _safe_frame(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        safe_frame = dataframe.copy(deep=True)
        for column_name in safe_frame.columns:
            series = safe_frame[column_name]
            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                safe_frame[column_name] = series.map(self._safe_value).astype("string")
        return safe_frame

    def _safe_value(self, value: Any) -> Any:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            return str(value)
        if isinstance(value, (list, tuple, set, dict)):
            return str(value)
        return str(value)

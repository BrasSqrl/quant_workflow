# Feature Subset Search Report Roadmap

This roadmap tracks the feature-subset-search report upgrade. The purpose of
`search_feature_subsets` is candidate selection before final model development,
so the report is intentionally comparison-first instead of final-validation
first.

## Scope

1. Add a candidate leaderboard that ranks every successful subset with
   discrimination, calibration, parsimony, and an overall selection score.
2. Add a champion-candidate summary with selected features, coefficients or
   feature importance, and a plain-language selection rationale.
3. Preserve AUC / ROC and KS evidence, and add richer comparison visuals across
   candidates.
4. Add calibration, Brier/log-loss, and feature-count tradeoff views so the
   selected subset is not judged by AUC alone.
5. Add top-candidate comparison tables for side-by-side review.
6. Add feature inclusion frequency, contribution consistency, excluded-feature,
   feature-family, and transformation-effectiveness diagnostics.
7. Add redundancy diagnostics for high-correlation feature pairs that appear
   together in top candidates.
8. Add candidate risk flags for complexity, metric gaps, calibration weakness,
   redundancy, and coefficient instability.
9. Add segment-level and time-split candidate performance where the scored
   candidate frames contain eligible segment or date fields.
10. Keep all outputs exported under the feature-subset-search artifact group so
    the mode remains separate from normal model-development validation.

## Implemented Outputs

- `subset_search_leaderboard`
- `subset_search_top_candidate_comparison`
- `subset_search_selection_rationale`
- `subset_search_candidate_risk_flags`
- `subset_search_contribution_consistency`
- `subset_search_redundancy_diagnostics`
- `subset_search_excluded_feature_insights`
- `subset_search_feature_family_view`
- `subset_search_transformation_effectiveness`
- `subset_search_segment_performance`
- `subset_search_time_performance`
- `subset_search_leaderboard_score_chart`
- `subset_search_metric_comparison_heatmap`
- `subset_search_calibration_comparison`
- `subset_search_risk_flag_summary`
- `subset_search_contribution_consistency_chart`
- `subset_search_redundancy_watchlist`
- `subset_search_segment_performance_chart`
- `subset_search_time_performance_chart`
- `subset_search_transformation_effectiveness_chart`
- `subset_search_feature_family_chart`

## Interpretation Standard

- The configured ranking metric still determines the selected subset.
- The overall selection score is a decision aid. It combines performance,
  calibration, and simplicity so reviewers can see tradeoffs.
- Risk flags are review prompts, not automatic rejection rules.
- Segment and time diagnostics are produced only when the scored candidate data
  contains enough eligible rows and fields.
- Final model validation should still be performed by running `fit_new_model`
  after choosing the feature subset.

# Statistical Test Catalog

This document describes the statistical tests and test-like validation statistics
available in Quant Studio today. It is intentionally tied to the implemented
code rather than to textbook descriptions alone.

Two scope notes matter:

1. The framework mixes formal hypothesis tests and industry-standard diagnostic
   statistics inside the same validation surface.
2. Some outputs are toggled directly through `DiagnosticConfig`, while others
   are always produced as part of a larger diagnostic suite such as evaluation
   or calibration.

Primary implementation files:

- `src/quant_pd_framework/steps/evaluation.py`
- `src/quant_pd_framework/steps/diagnostics.py`
- `src/quant_pd_framework/steps/backtesting.py`

## Quick Index

| Test or statistic | Primary output(s) | Typical use | Main code path |
| --- | --- | --- | --- |
| Augmented Dickey-Fuller (ADF) | `adf_tests` | Time-series or panel stationarity review | `DiagnosticsStep._add_adf_outputs`, `_run_adf_test` |
| Kolmogorov-Smirnov style separation statistic (KS) | split metric `ks_statistic` | Binary discrimination strength | `EvaluationStep._ks_statistic` |
| Hosmer-Lemeshow statistic | `calibration_summary` | Binary calibration goodness-of-fit by bins | `DiagnosticsStep._hosmer_lemeshow_statistic` |
| Calibration slope and intercept | `calibration_summary` | Over/under-confidence review | `DiagnosticsStep._calibration_slope_intercept` |
| Expected / Maximum Calibration Error (ECE / MCE) | `calibration_summary` | Bin-level calibration gap review | `DiagnosticsStep._calibration_error_metrics` |
| Variance Inflation Factor (VIF) | `vif` | Numeric multicollinearity screening | `DiagnosticsStep._add_vif_outputs` |
| Weight of Evidence / Information Value (WoE / IV) | `woe_iv_summary`, `woe_iv_detail` | Binary predictor strength and scorecard analysis | `DiagnosticsStep._add_woe_iv_outputs` |
| Population Stability Index (PSI) | `psi` | Development vs scored-population drift | `DiagnosticsStep._add_psi_outputs`, `_compute_population_stability_index` |

## 1. Augmented Dickey-Fuller (ADF)

### What it tests

ADF tests whether a time-ordered series appears to contain a unit root. In
practice, the framework uses it as a stationarity check on:

- average predicted score over time
- average target over time, when labels are available
- up to three top features aggregated over time

### When to use it

Use ADF when the workflow is `time_series` or `panel` and you need evidence
about whether the modeled series is behaving like a stationary process.

Typical cases:

- CECL lifetime PD development
- CCAR forecasting
- panel model validation over time

Do not treat it as a meaningful test for purely cross-sectional datasets. The
framework already enforces that by skipping ADF unless the split structure is
`time_series` or `panel`.

### How to interpret it

- Lower p-values suggest stronger evidence against a unit root.
- High p-values suggest the series may be non-stationary.
- ADF is sensitive to short samples, structural breaks, and aggregation choice.

### Implemented code

Source excerpt from `src/quant_pd_framework/steps/diagnostics.py`:

```python
def _run_adf_test(
    self,
    series: pd.Series,
    *,
    split_name: str,
    series_name: str,
) -> dict[str, Any] | None:
    if len(series) < 10:
        return None
    try:
        statistic, p_value, used_lag, nobs, _, _ = adfuller(series.to_numpy())
    except Exception:
        return None
    return {
        "split": split_name,
        "series_name": series_name,
        "adf_statistic": float(statistic),
        "p_value": float(p_value),
        "used_lag": int(used_lag),
        "observations": int(nobs),
    }
```

### Output location

- Table: `adf_tests`
- JSON export: `statistical_tests.json` under key `adf`

## 2. Kolmogorov-Smirnov Style Separation Statistic (KS)

### What it tests

The framework's KS statistic measures the maximum gap between the empirical CDF
of scores for defaults and the empirical CDF of scores for non-defaults.

This is a discrimination statistic, not a formal p-value-based two-sample test
in the exported output.

### When to use it

Use KS for binary PD-style models when you want a simple measure of class
separation that risk and validation teams commonly recognize.

### How to interpret it

- Higher values indicate better separation between positives and negatives.
- A low KS suggests poor ranking power.
- KS should be reviewed alongside ROC AUC, lift, and calibration, not alone.

### Implemented code

Source excerpt from `src/quant_pd_framework/steps/evaluation.py`:

```python
def _ks_statistic(self, y_true: pd.Series, y_score: np.ndarray) -> float:
    positives = y_score[y_true == 1]
    negatives = y_score[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("KS statistic requires both classes.")

    all_scores = np.sort(np.unique(y_score))
    positive_cdf = np.searchsorted(np.sort(positives), all_scores, side="right") / len(
        positives
    )
    negative_cdf = np.searchsorted(np.sort(negatives), all_scores, side="right") / len(
        negatives
    )
    return float(np.max(np.abs(positive_cdf - negative_cdf)))
```

### Output location

- Split metrics table: `split_metrics`
- Run metrics JSON: `metrics.json`

## 3. Hosmer-Lemeshow Statistic

### What it tests

Hosmer-Lemeshow compares observed and expected defaults across calibration bins.
The framework computes it inside the calibration workflow for every evaluated
calibration method:

- base model probabilities
- Platt scaling, when enabled and fit succeeds
- isotonic calibration, when enabled and fit succeeds

### When to use it

Use it for binary probability models when you want a bin-based calibration
goodness-of-fit statistic that is familiar to validators.

### How to interpret it

- Lower statistic values are generally better.
- Higher p-values suggest weaker evidence of poor calibration.
- The test can be unstable with very large samples, very small samples, or
  sparse bins, so it should be reviewed with calibration curves and ECE/MCE.

### Implemented code

Source excerpt from `src/quant_pd_framework/steps/diagnostics.py`:

```python
def _hosmer_lemeshow_statistic(
    self,
    calibration_table: pd.DataFrame,
) -> tuple[float, float]:
    observed_defaults = calibration_table["observed_default_count"].to_numpy(dtype=float)
    expected_defaults = calibration_table["expected_default_count"].to_numpy(dtype=float)
    observation_count = calibration_table["observation_count"].to_numpy(dtype=float)
    observed_non_defaults = observation_count - observed_defaults
    expected_non_defaults = observation_count - expected_defaults
    default_term = (observed_defaults - expected_defaults) ** 2 / np.maximum(
        expected_defaults,
        1e-9,
    )
    non_default_term = (
        (observed_non_defaults - expected_non_defaults) ** 2
        / np.maximum(expected_non_defaults, 1e-9)
    )
    statistic = float(np.sum(default_term + non_default_term))
    degrees_of_freedom = max(int(len(calibration_table) - 2), 1)
    p_value = float(1.0 - chi2.cdf(statistic, df=degrees_of_freedom))
    return statistic, p_value
```

### Output location

- Table: `calibration_summary`
- JSON export: `statistical_tests.json` under key `calibration_methods`

## 4. Calibration Slope and Intercept

### What it tests

The framework fits a logistic regression of the realized target on the logit of
predicted probability. That produces:

- a calibration intercept
- a calibration slope

These are standard calibration diagnostics rather than classical hypothesis
tests.

### When to use it

Use this whenever a binary model is meant to produce probabilities rather than
only ranks.

Typical use:

- PD development review
- challenger model comparison
- calibration challenger selection

### How to interpret it

- Intercept near `0` is desirable.
- Slope near `1` is desirable.
- Slope below `1` often suggests overly extreme predictions.
- Slope above `1` often suggests under-dispersed predictions.

### Implemented code

Source excerpt from `src/quant_pd_framework/steps/diagnostics.py`:

```python
def _calibration_slope_intercept(
    self,
    y_true: np.ndarray,
    probability: np.ndarray,
) -> tuple[float, float]:
    try:
        design = sm.add_constant(self._safe_logit(probability), has_constant="add")
        fit = sm.GLM(y_true, design, family=sm.families.Binomial()).fit()
        intercept = float(fit.params[0])
        slope = float(fit.params[1])
    except Exception:
        intercept = float("nan")
        slope = float("nan")
    return intercept, slope
```

### Output location

- Table: `calibration_summary`

## 5. Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)

### What it tests

These statistics summarize the absolute gap between observed and predicted event
rates across calibration bins.

- ECE is the weighted average absolute gap.
- MCE is the largest absolute gap.

### When to use it

Use ECE and MCE when you want a more intuitive calibration-gap summary than a
single hypothesis-test statistic.

They are particularly useful for:

- probability model validation
- comparing base, Platt, and isotonic calibration
- governance reporting

### How to interpret it

- Lower is better for both statistics.
- ECE gives the average miss.
- MCE highlights the worst bin miss.

### Implemented code

Source excerpt from `src/quant_pd_framework/steps/diagnostics.py`:

```python
def _calibration_error_metrics(self, calibration_table: pd.DataFrame) -> tuple[float, float]:
    total = float(calibration_table["observation_count"].sum()) or 1.0
    gaps = calibration_table["absolute_gap"].to_numpy(dtype=float)
    weights = calibration_table["observation_count"].to_numpy(dtype=float) / total
    expected_calibration_error = float(np.sum(weights * gaps))
    maximum_calibration_error = float(np.max(gaps))
    return expected_calibration_error, maximum_calibration_error
```

### Output location

- Table: `calibration_summary`

## 6. Variance Inflation Factor (VIF)

### What it tests

VIF estimates how strongly each numeric predictor is linearly explained by the
other numeric predictors in the selected train split.

This is a collinearity diagnostic, not a model-performance metric.

### When to use it

Use VIF when:

- building interpretable linear, logistic, probit, panel, or scorecard models
- checking whether coefficient instability may be caused by correlated inputs
- enforcing a feature policy such as `max_vif`

### How to interpret it

- Higher VIF implies more multicollinearity pressure.
- The framework does not hardcode a pass/fail threshold; that comes from
  `FeaturePolicyConfig.max_vif` if the user enables policy checks.

### Implemented code

Source excerpt from `src/quant_pd_framework/steps/diagnostics.py`:

```python
design = dataframe[numeric_columns].copy()
design = design.fillna(design.median(numeric_only=True))
matrix = design.to_numpy(dtype=float)
for index, column in enumerate(numeric_columns):
    try:
        vif_value = variance_inflation_factor(matrix, index)
    except Exception:
        vif_value = np.nan
    vif_rows.append({"feature_name": column, "vif": float(vif_value)})
```

### Output location

- Table: `vif`
- Figure: `vif_profile`

## 7. Weight of Evidence (WoE) and Information Value (IV)

### What it tests

For binary targets, the framework can bucket each top feature and compute:

- WoE by bucket
- IV contribution by bucket
- total IV by feature

This supports scorecard-style analysis and univariate predictor strength review.

### When to use it

Use WoE/IV when:

- building or reviewing PD scorecards
- screening binary predictors
- checking whether a variable carries meaningful univariate signal

### How to interpret it

- WoE shows the log ratio of good vs bad distribution by bucket.
- IV summarizes how much a feature separates good and bad populations.
- IV is sensitive to binning, rare groups, and sample instability, so it should
  be treated as a screen rather than as a sole selection rule.

### Implemented code

Source excerpt from `src/quant_pd_framework/steps/diagnostics.py`:

```python
summary["bad_pct"] = summary["bad"].clip(lower=0.5) / total_bad
summary["good_pct"] = summary["good"].clip(lower=0.5) / total_good
summary["woe"] = np.log(summary["good_pct"] / summary["bad_pct"])
summary["iv_component"] = (summary["good_pct"] - summary["bad_pct"]) * summary["woe"]
iv_value = float(summary["iv_component"].sum())
woe_rows.append({"feature_name": feature, "information_value": iv_value})
```

### Output location

- Summary table: `woe_iv_summary`
- Detail table: `woe_iv_detail`

## 8. Population Stability Index (PSI)

### What it tests

PSI measures how much the distribution of a feature or score changed between the
development population and the current scored population. In the framework, the
default comparison is:

- expected: train split
- actual: test split

It is calculated for top features and for the model score itself.

### When to use it

Use PSI when you want to review whether the held-out or newly scored sample
looks materially different from development.

### How to interpret it

- Higher PSI means more distribution shift.
- PSI is descriptive rather than hypothesis-test based.
- For numeric fields, results depend on the training-derived bucket edges.

### Implemented code

Source excerpt from `src/quant_pd_framework/steps/diagnostics.py`:

```python
all_buckets = expected_dist.index.union(actual_dist.index)
psi_value = 0.0
for bucket in all_buckets:
    expected_pct = max(float(expected_dist.get(bucket, 0.0)), 1e-6)
    actual_pct = max(float(actual_dist.get(bucket, 0.0)), 1e-6)
    psi_value += (actual_pct - expected_pct) * math.log(actual_pct / expected_pct)
return float(psi_value)
```

### Output location

- Table: `psi`
- Figure: `psi_profile`

## Related Non-Test Diagnostics

The following are often discussed like tests in practice, but they are better
thought of as analytical diagnostics:

- threshold sweep
- quantile backtest
- lift and gain
- ROC and precision-recall curves
- residual plots and QQ plots

Those are documented in [METRIC_CATALOG.md](./METRIC_CATALOG.md).

## Audit Notes

- Every test described here is generated from the exported run artifacts, not
  from an undocumented notebook.
- The main switch controlling test availability is `DiagnosticConfig` in
  `src/quant_pd_framework/config.py`.
- Tests that require labels are skipped automatically in score-only runs when
  the target column is absent.

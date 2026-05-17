# Security Policy

## Supported Use

Quant Studio is distributed for noncommercial use unless separate written commercial
permission is granted. Security fixes should target the current `main` branch unless
a maintained release branch is explicitly documented.

## Reporting Vulnerabilities

Report suspected vulnerabilities privately through the repository owner or a private
GitHub security advisory. Do not open a public issue with exploitable details, secrets,
or private model/data artifacts.

Include:

- A concise description of the issue and affected feature.
- Steps to reproduce using synthetic data where possible.
- Whether the issue can expose data, execute code, corrupt artifacts, or bypass audit
  controls.
- The commit hash, operating system, Python version, and package environment.

## Joblib Trust Model

Quant Studio treats `.joblib` files as executable-code-equivalent artifacts because
Python pickle/joblib loading can execute arbitrary code. The application writes a
SHA-256 sidecar beside model, checkpoint, and background snapshot artifacts as:

```text
<artifact>.joblib
<artifact>.joblib.sha256
```

Externally supplied model artifacts are blocked when the sidecar is missing or does
not match. Legacy internal checkpoints and background snapshots may be loaded only
from the current run folder with a warning so old Quant Studio runs remain recoverable.

Do not load `.joblib` files from untrusted parties. If a model must be exchanged,
exchange the full artifact folder, including the `.sha256` sidecar and manifest files.

## Secrets Policy

Do not commit API keys, passwords, AWS access keys, private bucket names, customer data,
or proprietary model output. S3 access should use IAM roles, AWS profiles, or environment
configuration outside the repository. Quant Studio does not collect or store cloud
access secrets.

The repository uses `detect-secrets` with `.secrets.baseline`. Any future baseline entry
must be reviewed as a false positive and documented in the pull request or commit
message. Active secrets must be removed and rotated.

## Artifact Handling

Run folders may contain model objects, predictions, source metadata, documentation
packages, and validation evidence. Treat artifact folders as sensitive. Before sharing,
review the output for row-level data, confidential feature names, source paths, or
organization-specific governance notes.

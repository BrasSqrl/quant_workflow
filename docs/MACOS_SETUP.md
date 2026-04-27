# macOS Setup

This guide explains how to run Quant Studio locally on a Mac. The Python
modeling framework and Streamlit GUI are the same as Windows; macOS only needs
shell scripts instead of the Windows `.bat` launchers.

## Recommended Workflow

1. Install Python 3.11 or newer.
2. Clone or pull the Quant Studio repository.
3. Open Terminal in the repo folder.
4. Run the macOS bootstrap script.
5. Run the macOS Streamlit launcher.
6. Open the local Streamlit URL in a browser.

## One-Time Setup

From the repo root:

```bash
bash scripts/bootstrap_macos.sh
```

The bootstrap script:

- verifies Python 3.11 or newer
- creates `.venv/` in the project folder when needed
- upgrades `pip`, `setuptools`, and `wheel`
- installs Quant Studio with GUI dependencies through `pip install -e ".[gui]"`
- verifies that Streamlit imports successfully

## Start The App

From the repo root:

```bash
bash scripts/run_macos_streamlit.sh
```

Then open:

```text
http://localhost:8501
```

Streamlit may open the browser automatically. If it does not, copy the local
URL shown in Terminal.

## Optional Executable Script Style

If you prefer running scripts directly instead of using `bash`, make them
executable once:

```bash
chmod +x scripts/bootstrap_macos.sh scripts/run_macos_streamlit.sh
```

Then run:

```bash
./scripts/bootstrap_macos.sh
./scripts/run_macos_streamlit.sh
```

## Python Selection

The scripts use `python3` by default. To force a specific Python executable:

```bash
PYTHON_BIN=/opt/homebrew/bin/python3 bash scripts/bootstrap_macos.sh
```

Common choices:

- Apple Silicon Homebrew: `/opt/homebrew/bin/python3`
- Intel Homebrew: `/usr/local/bin/python3`
- Python.org installer: usually available as `python3`
- pyenv: activate the pyenv version first, then run the scripts

Check your Python version with:

```bash
python3 --version
```

Quant Studio requires Python 3.11 or newer.

## Apple Silicon Notes

On Apple Silicon Macs, keep Terminal, Python, and installed packages on the
same architecture. Avoid mixing a Rosetta/x86 Terminal with an arm64 Homebrew
Python unless you intentionally manage both environments.

The simplest path is:

```bash
brew install python
PYTHON_BIN=/opt/homebrew/bin/python3 bash scripts/bootstrap_macos.sh
```

## Xcode Command Line Tools

Most dependencies install from wheels. If pip reports a native compile failure,
install Apple's command line build tools:

```bash
xcode-select --install
```

Then rerun:

```bash
bash scripts/bootstrap_macos.sh
```

## Data Loading

The same data-loading options are available on macOS:

- bundled sample data
- upload through the GUI
- place files in `Data_Load/` and select them from the dropdown

For multi-GB data, prefer `Data_Load/`, Parquet, and Large Data Mode instead of
browser upload.

## Useful Runtime Options

Use a different port:

```bash
PORT=8502 bash scripts/run_macos_streamlit.sh
```

Use a different host:

```bash
HOST=127.0.0.1 bash scripts/run_macos_streamlit.sh
```

Use a smaller Streamlit upload ceiling:

```bash
MAX_UPLOAD_MB=1024 bash scripts/run_macos_streamlit.sh
```

Use a different virtual environment path:

```bash
VENV_DIR="$HOME/.venvs/quant-studio" bash scripts/bootstrap_macos.sh
VENV_DIR="$HOME/.venvs/quant-studio" bash scripts/run_macos_streamlit.sh
```

## Troubleshooting

### Permission denied when running a script

Use the explicit bash form:

```bash
bash scripts/bootstrap_macos.sh
```

Or make scripts executable:

```bash
chmod +x scripts/bootstrap_macos.sh scripts/run_macos_streamlit.sh
```

### Python executable was not found

Install Python 3.11 or newer, then rerun setup. If Python is installed but not
on PATH, pass it explicitly:

```bash
PYTHON_BIN=/opt/homebrew/bin/python3 bash scripts/bootstrap_macos.sh
```

### pip install fails on a native build

Install Xcode Command Line Tools:

```bash
xcode-select --install
```

Then rerun setup.

### SSL certificate errors with python.org Python

Run the `Install Certificates.command` included with the Python.org installer,
then rerun setup.

### macOS blocks files after downloading a zip

If the repo was downloaded as a zip and macOS quarantine blocks script
execution, clear quarantine from the repo root:

```bash
xattr -dr com.apple.quarantine .
```

### Port 8501 is already in use

Run Streamlit on another port:

```bash
PORT=8502 bash scripts/run_macos_streamlit.sh
```

### Static PNG export does not work

The app works without static PNG export. Keep `Export individual figure HTML and
PNG files` off unless separate files are required. If PNG export is required,
Chrome may be needed by Plotly/Kaleido; install Chrome normally or follow the
error message from Plotly.

## Updating Later

After pulling new code:

```bash
git pull
bash scripts/bootstrap_macos.sh
bash scripts/run_macos_streamlit.sh
```

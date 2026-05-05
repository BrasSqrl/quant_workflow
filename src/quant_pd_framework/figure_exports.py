"""On-demand export helpers for standalone Plotly chart files."""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import plotly.io as pio
from plotly.offline import get_plotlyjs


@dataclass(frozen=True, slots=True)
class FigureExportAsset:
    """A rendered chart file ready to write to a zip archive or directory."""

    arcname: str
    data: bytes
    figure_name: str
    file_format: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class FigureExportResult:
    """Rendered chart assets and the manifest describing them."""

    assets: tuple[FigureExportAsset, ...]
    manifest: dict[str, Any]


def build_individual_figure_zip(
    visualizations: Mapping[str, Any],
    *,
    root_dir: str = "individual_images",
    include_html: bool = True,
    include_png: bool = True,
    max_figures: int | None = None,
    png_figure_limit: int | None = None,
) -> bytes:
    """Builds a portable zip of individual chart HTML and PNG files."""

    result = build_figure_export_assets(
        visualizations,
        root_dir=root_dir,
        include_html=include_html,
        include_png=include_png,
        max_figures=max_figures,
        png_figure_limit=png_figure_limit,
    )
    output = BytesIO()
    with ZipFile(output, mode="w", compression=ZIP_DEFLATED, compresslevel=3) as archive:
        write_figure_assets_to_archive(archive, result.assets)
    return output.getvalue()


def build_figure_export_assets(
    visualizations: Mapping[str, Any],
    *,
    root_dir: str,
    include_html: bool = True,
    include_png: bool = True,
    max_figures: int | None = None,
    png_figure_limit: int | None = None,
) -> FigureExportResult:
    """Renders figures once into reusable archive assets.

    HTML exports reference one shared Plotly JavaScript file instead of embedding
    Plotly in every chart. PNG exports use Plotly's batch image writer when
    available, which avoids repeatedly starting Kaleido for every chart.
    """

    normalized_root = _clean_root(root_dir)
    available_figures = _named_figures(visualizations)
    named_figures = _limit_named_figures(available_figures, max_figures)
    png_figures = (
        _limit_named_figures(named_figures, png_figure_limit)
        if png_figure_limit is not None
        else named_figures
    )
    generated_at_utc = datetime.now(UTC).isoformat()
    assets: list[FigureExportAsset] = []
    manifest: dict[str, Any] = {
        "generated_at_utc": generated_at_utc,
        "available_figure_count": len(available_figures),
        "figure_count": len(named_figures),
        "skipped_figure_count": max(len(available_figures) - len(named_figures), 0),
        "max_figures": max_figures,
        "png_figure_limit": png_figure_limit,
        "html_enabled": include_html,
        "png_enabled": include_png,
        "html_runtime_note": (
            "HTML charts share one Plotly JavaScript file at html/plotly.min.js."
            if include_html
            else "HTML chart export was not requested."
        ),
        "png_runtime_note": (
            "PNG charts are rendered with Plotly batch image export when available."
            if include_png
            else "PNG chart export was not requested."
        ),
        "figures": {},
        "selected_figures": [figure_name for figure_name, _safe_name, _figure in named_figures],
        "skipped_figures": [
            figure_name
            for figure_name, _safe_name, _figure in available_figures[len(named_figures) :]
        ],
        "support_files": [],
        "warnings": [],
    }

    if include_html and named_figures:
        plotly_js_arcname = f"{normalized_root}/html/plotly.min.js"
        plotly_js = get_plotlyjs().encode("utf-8")
        assets.append(
            FigureExportAsset(
                arcname=plotly_js_arcname,
                data=plotly_js,
                figure_name="plotly.min.js",
                file_format="javascript",
                size_bytes=len(plotly_js),
            )
        )
        manifest["support_files"].append(
            {
                "path": plotly_js_arcname,
                "file_format": "javascript",
                "size_bytes": len(plotly_js),
            }
        )
        for figure_name, safe_name, figure in named_figures:
            html_arcname = f"{normalized_root}/html/{safe_name}.html"
            html_bytes = _figure_html_document(figure_name, figure).encode("utf-8")
            assets.append(
                FigureExportAsset(
                    arcname=html_arcname,
                    data=html_bytes,
                    figure_name=figure_name,
                    file_format="html",
                    size_bytes=len(html_bytes),
                )
            )
            manifest["figures"].setdefault(figure_name, {})["html"] = html_arcname
            manifest["figures"][figure_name]["safe_name"] = safe_name

    if include_png and png_figures:
        png_records, warnings = _render_png_assets(png_figures, normalized_root)
        assets.extend(png_records)
        manifest["warnings"].extend(warnings)
        for asset in png_records:
            manifest["figures"].setdefault(asset.figure_name, {})["png"] = asset.arcname
            manifest["figures"][asset.figure_name]["safe_name"] = Path(asset.arcname).stem

    manifest_bytes = json.dumps(manifest, indent=2, default=str).encode("utf-8")
    assets.append(
        FigureExportAsset(
            arcname=f"{normalized_root}/figure_manifest.json",
            data=manifest_bytes,
            figure_name="figure_manifest",
            file_format="json",
            size_bytes=len(manifest_bytes),
        )
    )
    return FigureExportResult(assets=tuple(assets), manifest=manifest)


def write_figure_assets_to_archive(
    archive: ZipFile,
    assets: tuple[FigureExportAsset, ...],
) -> None:
    """Writes pre-rendered chart assets to an open zip archive."""

    for asset in assets:
        archive.writestr(asset.arcname, asset.data, compress_type=_compress_type_for_asset(asset))


def export_figure_files_to_directory(
    visualizations: Mapping[str, Any],
    *,
    html_dir: Path,
    png_dir: Path,
    include_html: bool,
    include_png: bool,
    warning_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Writes individual chart files to existing artifact folders."""

    result = build_figure_export_assets(
        visualizations,
        root_dir="figures",
        include_html=include_html,
        include_png=include_png,
    )
    for warning in result.manifest.get("warnings", []):
        if warning_callback is not None:
            warning_callback(str(warning))

    for asset in result.assets:
        parts = Path(asset.arcname).parts
        if len(parts) < 2:
            continue
        if parts[1] == "html":
            output_path = html_dir / Path(*parts[2:])
        elif parts[1] == "png":
            output_path = png_dir / Path(*parts[2:])
        else:
            output_path = html_dir.parent / Path(*parts[1:])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(asset.data)

    manifest = result.manifest.copy()
    manifest["figures"] = {
        name: {
            key: str(_directory_path_for_arcname(value, html_dir=html_dir, png_dir=png_dir))
            if key in {"html", "png"}
            else value
            for key, value in payload.items()
        }
        for name, payload in result.manifest.get("figures", {}).items()
    }
    manifest["support_files"] = [
        {
            **support_file,
            "path": str(
                _directory_path_for_arcname(
                    str(support_file.get("path", "")),
                    html_dir=html_dir,
                    png_dir=png_dir,
                )
            ),
        }
        for support_file in result.manifest.get("support_files", [])
        if isinstance(support_file, dict)
    ]
    return manifest


def _render_png_assets(
    named_figures: list[tuple[str, str, Any]],
    root_dir: str,
) -> tuple[list[FigureExportAsset], list[str]]:
    assets: list[FigureExportAsset] = []
    warnings: list[str] = []
    with tempfile.TemporaryDirectory(prefix="quant_studio_figures_") as temporary_directory:
        temporary_root = Path(temporary_directory)
        png_paths = [temporary_root / f"{safe_name}.png" for _, safe_name, _ in named_figures]
        figures = [figure for _, _, figure in named_figures]
        try:
            pio.write_images(figures, png_paths)
        except Exception as exc:
            warnings.append(f"Batch PNG export failed; using per-chart fallback: {exc}")
            for figure_name, _safe_name, figure in named_figures:
                png_path = temporary_root / f"{_safe_name}.png"
                try:
                    pio.write_image(figure, png_path)
                except Exception as fallback_exc:
                    warnings.append(f"Could not export PNG for {figure_name}: {fallback_exc}")
        for figure_name, safe_name, _figure in named_figures:
            png_path = temporary_root / f"{safe_name}.png"
            if not png_path.exists():
                continue
            data = png_path.read_bytes()
            assets.append(
                FigureExportAsset(
                    arcname=f"{root_dir}/png/{safe_name}.png",
                    data=data,
                    figure_name=figure_name,
                    file_format="png",
                    size_bytes=len(data),
                )
            )
    return assets, warnings


def _figure_html_document(figure_name: str, figure: Any) -> str:
    title = escape(str(figure_name).replace("_", " ").title())
    figure_div = pio.to_html(
        figure,
        include_plotlyjs=False,
        full_html=False,
        validate=False,
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <script src="plotly.min.js"></script>
  <style>
    body {{
      margin: 0;
      background: #f7f9fc;
      color: #172033;
      font-family: "Aptos", "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1280px;
      margin: 24px auto;
      padding: 24px;
      background: #ffffff;
      border: 1px solid #d9e2ef;
      border-radius: 18px;
      box-shadow: 0 18px 45px rgba(23, 32, 51, 0.08);
    }}
    h1 {{
      margin: 0 0 18px;
      font-size: 22px;
      letter-spacing: -0.02em;
    }}
  </style>
</head>
<body>
  <main>
    <h1>{title}</h1>
    {figure_div}
  </main>
</body>
</html>
"""


def _named_figures(visualizations: Mapping[str, Any]) -> list[tuple[str, str, Any]]:
    seen: dict[str, int] = {}
    named: list[tuple[str, str, Any]] = []
    for raw_name, figure in visualizations.items():
        if figure is None:
            continue
        figure_name = str(raw_name)
        safe_name = _sanitize_name(figure_name)
        count = seen.get(safe_name, 0)
        seen[safe_name] = count + 1
        if count:
            safe_name = f"{safe_name}_{count + 1}"
        named.append((figure_name, safe_name, figure))
    return named


def _limit_named_figures(
    named_figures: list[tuple[str, str, Any]],
    max_figures: int | None,
) -> list[tuple[str, str, Any]]:
    if max_figures is None:
        return named_figures
    return named_figures[: max(int(max_figures), 0)]


def _compress_type_for_asset(asset: FigureExportAsset) -> int:
    if asset.file_format == "png":
        return ZIP_STORED
    return ZIP_DEFLATED


def _sanitize_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip().lower()).strip("._-")
    return normalized or "figure"


def _clean_root(value: str) -> str:
    cleaned = str(value or "individual_images").replace("\\", "/").strip("/")
    return cleaned or "individual_images"


def _directory_path_for_arcname(arcname: str, *, html_dir: Path, png_dir: Path) -> Path:
    parts = Path(arcname).parts
    if len(parts) >= 2 and parts[1] == "html":
        return html_dir / Path(*parts[2:])
    if len(parts) >= 2 and parts[1] == "png":
        return png_dir / Path(*parts[2:])
    return html_dir.parent / Path(*parts[1:])

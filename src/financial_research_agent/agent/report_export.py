"""Export report files using a stable, generated filename."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from financial_research_agent.agent.report import ResearchReport, render_report_markdown


def export_research_report(
    report: ResearchReport, output_dir: Path, *, format: Literal["md", "json"], overwrite: bool = False,
) -> Path:
    if format not in {"md", "json"}:
        raise ValueError("unsupported report export format")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"alphascope-{report.report_id.hex}.{format}"
    text = render_report_markdown(report) if format == "md" else report.model_dump_json(indent=2) + "\n"
    if not overwrite:
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        return destination

    temporary = None
    try:
        with NamedTemporaryFile(mode="w", dir=output_dir, encoding="utf-8", newline="\n", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination

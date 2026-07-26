"""CLI tests for ``ghh inspect``."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from ghh.cli import main
from tests.test_inspect import make_synthetic_staff_image


def test_cli_inspect_parsing(tmp_path: Path):
    runner = CliRunner()
    img_path = tmp_path / "cli_sample.png"
    make_synthetic_staff_image(img_path)

    result = runner.invoke(main, [
        "inspect",
        str(img_path),
        "--ink-color", "255,0,0",
        "--alpha", "0.35",
        "--overlay-only",
        "--format", "jpg",
        "--no-line-text",
    ])
    assert result.exit_code == 0, result.output

    expected = img_path.with_name(img_path.stem + "_inspect.jpg")
    assert expected.exists(), f"Expected output file {expected} to exist"
    assert expected.with_suffix(".json").exists()


def test_cli_inspect_bad_color(tmp_path: Path):
    runner = CliRunner()
    img_path = tmp_path / "cli_sample.png"
    make_synthetic_staff_image(img_path)

    result = runner.invoke(main, [
        "inspect", str(img_path), "--ink-color", "nope",
    ])
    assert result.exit_code != 0
    assert "Invalid color" in result.output


def test_cli_inspect_stage_stem(tmp_path: Path):
    runner = CliRunner()
    out = tmp_path / "book_output"
    ckpt = out / "08_deskewed"
    ckpt.mkdir(parents=True)
    make_synthetic_staff_image(ckpt / "IMG_0099.png")

    result = runner.invoke(main, [
        "inspect", "IMG_0099",
        "--stage", "dewarp",
        "--output-dir", str(out),
        "--overlay-only",
        "--output", str(tmp_path / "staged.png"),
    ])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "staged.png").exists()

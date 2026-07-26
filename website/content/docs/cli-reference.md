---
title: "CLI Reference"
weight: 20
description: "Complete reference for all ghh commands and options"
---

Guido's Helping Hand provides a set of commands for processing, inspecting,
comparing, and publishing book page images. The main entry point is `ghh`.

```bash
ghh --version        # show version
ghh --help           # list all commands
ghh COMMAND --help   # help for a specific command
```

## ghh run

Process book page photos through the pipeline.

```bash
ghh run INPUT_DIR [OPTIONS]
```

This is the primary command. It runs all implemented stages by default,
producing a clean PDF from raw photographs.

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-o`, `--output-dir` | path | `INPUT_DIR_output` | Output directory |
| `--config` | path | auto-detected | Path to `book.toml` configuration file |
| `--stages` | string | all | Stages to run (e.g. `0,1,2` or `0-2,5,7`) |
| `--profile` | choice | `full` | Processing profile: `full`, `geometry`, `clean`, `quick` |
| `--preview` | integer | 0 | Process only N images (0 = all) |
| `--book-only` | flag | | Run only the Book branch (skip Score branch) |
| `--scores-only` | flag | | Run only the Score branch (skip Book branch) |
| `--skip-content-area` | flag | | Skip content area detection (Stage 6) |
| `--skip-deskew` | flag | | Skip deskewing (Stage 8) |
| `--skip-dewarp` | flag | | Skip dewarping (Stage 10) |
| `--skip-enhance` | flag | | Skip image enhancement (Stage 11) |
| `--skip-normalize` | flag | | Skip cross-page normalization (Stage 12) |
| `--skip-ocr` | flag | | Skip OCR |
| `--skip-omr` | flag | | Skip OMR (Stage 13) |
| `--model-dir` | path | from config | Path to chant-omr OpenVINO model directory |
| `--ai-dewarp` | flag | | Use AI-based dewarping |
| `--binarize` | flag | | Binarize output images |
| `--cleanup` | flag | | Delete intermediate checkpoints after success |
| `--on-error` | choice | `skip` | Error handling: `skip`, `stop`, `force` |
| `-j`, `--jobs` | integer | half CPU cores | Parallel workers per stage (1 = sequential) |
| `-v`, `--verbose` | flag | | Verbose logging (DEBUG level) |
| `-q`, `--quiet` | flag | | Quiet logging (WARNING level only) |

### Examples

```bash
# Process all photos with default settings
ghh run ~/photos/LPA-1

# Process first 10 images with geometry-only profile
ghh run ~/photos/LPA-1 --preview 10 --profile geometry

# Run specific stages
ghh run ~/photos/LPA-1 --stages 0-5,8-9

# Use a custom config file
ghh run ~/photos/LPA-1 --config ~/photos/LPA-1/book.toml

# Skip OCR and clean up intermediates
ghh run ~/photos/LPA-1 --skip-ocr --cleanup

# Run only the book branch (skip OMR/score processing)
ghh run ~/photos/LPA-1 --book-only

# Run with 4 parallel workers per stage
ghh run ~/photos/LPA-1 -j 4

# Run with OMR (requires chant-omr and exported model)
ghh run ~/photos/LPA-1 --model-dir ~/models/chant-omr
```

## ghh analyze

Analyze book photos and generate a `book.toml` configuration file.

```bash
ghh analyze INPUT_DIR [OPTIONS]
```

Samples a subset of images to auto-detect ink colors, page layout,
photography conditions, and physical condition. Writes the results
to `book.toml` in the input directory.

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-o`, `--output-dir` | path | input dir | Where to write `book.toml` |
| `--samples` | integer | 15 | Number of sample images to analyze |

### Examples

```bash
# Analyze and generate config
ghh analyze ~/photos/LPA-1

# Analyze with more samples for better accuracy
ghh analyze ~/photos/LPA-1 --samples 30
```

## ghh compare

Open an interactive HTML comparison viewer in your browser.

```bash
ghh compare OUTPUT_DIR [IMAGE_STEM] [OPTIONS]
```

Displays all pipeline stages side-by-side for visual inspection.
Keyboard navigation lets you quickly flip between images and stages.

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `IMAGE_STEM` | argument | first image | Open at this image (e.g. `IMG_0012`) |
| `--input-dir` | path | auto-detected | Original input directory |
| `--no-open` | flag | | Don't open the browser automatically |

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| **Up** / **Down** | Previous / next image (reset to stage 0) |
| **PgUp** / **PgDn** | Previous / next image (keep current stage) |
| **Left** / **Right** | Previous / next stage |
| **S** | Toggle side-by-side mode |
| **M** | Toggle metadata panel |
| **Z** | Toggle zoom |
| **1**--**9** | Jump to stage by number |

In side-by-side mode, each pane can show a different image:

| Key | Action |
|-----|--------|
| **Up** / **Down** | Left pane: previous / next image (reset to stage 0) |
| **Shift+Up** / **Shift+Down** | Right pane: previous / next image (reset to stage 0) |
| **PgUp** / **PgDn** | Left pane: previous / next image (keep stage) |
| **Shift+PgUp** / **Shift+PgDn** | Right pane: previous / next image (keep stage) |
| **Left** / **Right** | Change left pane stage |
| **Shift+Left** / **Shift+Right** | Change right pane stage |
| **1**--**9** | Set left pane stage |
| **Shift+1**--**Shift+9** | Set right pane stage |

### Examples

```bash
# Compare all stages
ghh compare ~/photos/LPA-1_output

# Open at a specific image
ghh compare ~/photos/LPA-1_output IMG_0042
```

## ghh publish

Generate a self-contained, web-publishable comparison site.

```bash
ghh publish OUTPUT_DIR PUBLISH_DIR [OPTIONS]
```

Creates a directory with downscaled JPEG images and an HTML viewer
suitable for uploading to any static web host.

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input-dir` | path | auto-detected | Original input directory |
| `--max-dim` | integer | 1500 | Max pixel dimension for JPEGs |
| `--quality` | integer | 85 | JPEG compression quality (1--100) |
| `--stages` | string | all | Comma-separated stage numbers to include |
| `--with-flipbook` | flag | | Include the flipbook viewer |
| `--with-pdf` | flag | | Include PDF download (implies `--with-flipbook`) |
| `--no-open` | flag | | Don't open the browser automatically |

### Examples

```bash
# Publish with flipbook and PDF
ghh publish ~/photos/LPA-1_output ~/publish/LPA-1 \
    --with-flipbook --with-pdf

# Publish only specific stages, smaller images
ghh publish ~/photos/LPA-1_output ~/publish/LPA-1 \
    --stages 0,5,7 --max-dim 1000 --quality 75
```

## ghh flipbook

Generate a static HTML flipbook with page-turning animation.

```bash
ghh flipbook OUTPUT_DIR [FLIPBOOK_DIR] [OPTIONS]
```

Creates a standalone HTML page using the StPageFlip library.
By default includes a PDF download link.

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `FLIPBOOK_DIR` | argument | `OUTPUT_DIR/flipbook/` | Output directory |
| `--max-width` | integer | 1600 | Max page width in pixels |
| `--quality` | integer | 85 | JPEG compression quality |
| `--title` | string | directory name | Title displayed in the viewer |
| `--no-pdf` | flag | | Omit the PDF download link |
| `--cover` | flag | | Treat first page as a standalone cover |
| `--no-open` | flag | | Don't open the browser automatically |

### Examples

```bash
# Generate flipbook with defaults
ghh flipbook ~/photos/LPA-1_output

# Generate with custom title and smaller images
ghh flipbook ~/photos/LPA-1_output --title "LPA-1 Antiphonary" \
    --max-width 1200 --quality 80

# Generate without PDF link, treating first page as cover
ghh flipbook ~/photos/LPA-1_output --no-pdf --cover
```

## ghh inspect

Inspect a single image with diagnostic overlays (ink mask, staff lines,
polynomial fits, page quad, foxing spots) and suggested `book.toml`
ink parameters.

```bash
ghh inspect IMAGE_PATH [OPTIONS]
```

`IMAGE_PATH` may be an existing file, or an image stem such as
`IMG_0045` resolved from a pipeline checkpoint via `--stage` and
`--output-dir`.

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | path | none | Path to `book.toml` |
| `--stage` | string | none | Stage number, name, or checkpoint dir (`deskew`, `8`, `08_deskewed`). Alias: `dewarp` → `deskew` |
| `--output-dir` | path | auto | Pipeline output directory for stem resolution |
| `--open` | flag | off | Open the result in the system viewer |
| `--max-width` | int | 1600 | Downscale width for faster inspection |
| `--alpha` | float | 0.45 | Overlay opacity (0–1) |
| `--ink-color` | color | `#00ff00` | Ink mask color (`#RRGGBB` or `R,G,B`) |
| `--geom-color` | color | `#ffff00` | Geometric ink highlight color |
| `--foxing-color` | color | `#0000ff` | Foxing spot color |
| `--staff-color` | color | `#ff0000` | Staff segment color |
| `--poly-color` | color | `#00ffff` | Polynomial fit color |
| `--quad-color` | color | `#ff00ff` | Page quad color |
| `--format` | choice | `png` | Output format (`png`, `jpg`) |
| `--overlay-only` | flag | off | Omit the right-hand info panel |
| `--hide-coeffs` | flag | off | Hide polynomial coefficient lists |
| `--rms-only` | flag | off | Show only RMS and degree per line |
| `--small-font` | flag | off | Smaller per-line text |
| `--font-scale` | float | 0.45 | Explicit font scale for per-line text |
| `--no-line-text` | flag | off | Suppress all per-line metadata text |
| `--output` | path | next to source | Output path for the annotated image |

### Examples

```bash
# Inspect a file directly
ghh inspect /path/to/IMG_0045.jpg --config /path/to/book.toml

# Load a checkpoint by stem + stage from a pipeline output dir
ghh inspect IMG_0045 --stage deskew --output-dir ~/photos/LPA-1_output

# Legacy alias from PLAN.md (dewarp → deskew)
ghh inspect IMG_0045.jpg --stage dewarp --output-dir ~/photos/LPA-1_output

# Customize colors / opacity; JPEG overlay only
ghh inspect /path/to/IMG_0045.jpg --ink-color "255,0,0" --alpha 0.4 \
    --overlay-only --format jpg --no-line-text

# Reduce per-line text clutter
ghh inspect /path/to/IMG_0045.jpg --rms-only --small-font
```

### Visual examples

With per-line polynomial metadata (coefficients and RMS):

{{< img src="images/inspect_with_text.jpg" alt="Inspect overlay with per-line polynomial metadata" >}}

Clean overlay without per-line text:

{{< img src="images/inspect_no_text.jpg" alt="Inspect overlay without per-line text" >}}

A JSON sidecar is written next to the image with current vs suggested
`staff_color_hue` / `staff_color_range` values for tuning `book.toml`.

## ghh review

Review processed pages and generate a contact sheet.

```bash
ghh review OUTPUT_DIR [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--stage` | string | none | Specific stage directory to review |

## ghh cleanup

Delete intermediate checkpoint directories to save disk space.

```bash
ghh cleanup OUTPUT_DIR [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--keep` | string | none | Comma-separated stage numbers to keep |

### Examples

```bash
# Remove all intermediates, keep only the final PDF
ghh cleanup ~/photos/LPA-1_output

# Keep stages 0 (original preprocessing) and 8 (deskewed)
ghh cleanup ~/photos/LPA-1_output --keep 0,8
```

## ghh stages

List all implemented pipeline stages.

```bash
ghh stages
```

Displays a table with stage number, name, and checkpoint directory name.
No options.

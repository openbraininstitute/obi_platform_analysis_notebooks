"""Generate compact validation reports for registration."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from morph_spines_visualizer.core import data_loading


FONT_DIRECTORY = Path('lib') / 'fonts'


ISSUE_LABELS = {
    "false_positive": "False Positive",
    "incomplete_spine": "Incomplete Spine",
    "false_positive_quality": "Falsely Extended",
    "merged_spine": "Merged Spine",
    "split_spine": "Split Spine",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _identifier_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value.lower())


def _parse_screenshot(path: Path):
    parts = path.stem.split("_")
    if len(parts) == 3 and parts[0] == "section" and parts[2].isdigit():
        return "section", parts[1], None, int(parts[2])
    if (
        len(parts) == 5
        and parts[0] == "section"
        and parts[2] == "spine"
        and parts[4].isdigit()
    ):
        return "spine", parts[1], parts[3], int(parts[4])
    return None


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = FONT_DIRECTORY / "font_regular.otf"
    if not font_path.is_file():
        raise FileNotFoundError(f"Bundled report font not found: {font_path}")
    try:
        return ImageFont.truetype(str(font_path), size)
    except OSError as exc:
        raise OSError(f"Could not load bundled report font: {font_path}") from exc


def create_validation_report_pdf(
    image_directory,
    validation_csv_path,
    pdf_path,
    max_dimension=1600,
    jpeg_quality=75,
    morphology_path=None,
):
    """Create the three-part validation PDF used in registration.

    The report contains the section-validity figure, only sections marked
    ``missing`` in the CSV, and only spine screenshots with a positive issue
    record. ``morphology_path`` is supplied separately because the CSV may be
    stored in the mesh's proofreading directory.
    """
    image_directory = Path(image_directory)
    validation_csv_path = Path(validation_csv_path)
    pdf_path = Path(pdf_path)

    if not image_directory.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {image_directory}")
    if not validation_csv_path.is_file():
        raise FileNotFoundError(f"Validation CSV does not exist: {validation_csv_path}")
    if max_dimension is not None and max_dimension <= 0:
        raise ValueError("max_dimension must be positive or None")
    if not 0 < jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")

    morphology_path = (
        Path(morphology_path)
        if morphology_path is not None
        else validation_csv_path.with_name(
            validation_csv_path.name.removesuffix("_validation.csv") + ".h5"
        )
    )
    if not morphology_path.is_file():
        raise FileNotFoundError(
            f"Could not infer the morphology file from {validation_csv_path}"
        )

    analysis_image_paths = [
        validation_csv_path.with_name(
            f"{validation_csv_path.stem}_section_validity.png"
        ),
        validation_csv_path.with_name(
            f"{validation_csv_path.stem}_issue_counts.png"
        ),
    ]
    for analysis_image_path in analysis_image_paths:
        if not analysis_image_path.is_file():
            raise FileNotFoundError(
                f"Analysis figure does not exist: {analysis_image_path}"
            )

    missing_section_ids = set()
    issue_labels = {}
    with validation_csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(handle) if row]

    versioned_csv = (
        rows
        and len(rows[0]) == 2
        and rows[0][0] == 'validation_format'
        and rows[0][1] in {'2', '3'}
    )
    if versioned_csv:
        validation_format_version = rows[0][1]
        section_marker = ["table", "section"]
        spine_marker = ["table", "spine"]
        section_fields = [
            "Section", "Number Spines", "Validated (Yes or No)",
            "Remaining Spines to Validate", "False Positives",
            "Incomplete Spines", "Falsely Extended Spines",
            "Merged Spines", "Split Spines", "Missing Segmented Spines",
        ]
        spine_fields = [
            "Section ID", "Local Spine ID", "Global Spine ID", "Validity",
            "False Positive", "Incomplete Spine", "Falsely Extended",
            "Merged Spine", "Split Spine",
        ]
        legacy_spine_fields = [
            "Section ID", "Spine ID", "Validity", "False Positive",
            "Incomplete Spine", "Falsely Extended", "Merged Spine",
            "Split Spine",
        ]
        if rows[1] != section_marker or rows[2] != section_fields:
            raise ValueError(f"Invalid section table in {validation_csv_path}")
        spine_marker_index = rows.index(spine_marker)
        spine_header = rows[spine_marker_index + 1]
        if spine_header not in (spine_fields, legacy_spine_fields):
            raise ValueError(f"Invalid spine table in {validation_csv_path}")
        explicit_identity = spine_header == spine_fields
        spine_field_indices = {
            field_name: spine_header.index(field_name)
            for field_name in spine_header
        }
        error_columns = {
            "False Positive": "False Positive",
            "Incomplete Spine": "Incomplete Spine",
            "Falsely Extended": "Falsely Extended",
            "Merged Spine": "Merged Spine",
            "Split Spine": "Split Spine",
        }
        for row in rows[3:spine_marker_index]:
            if row[9].strip().lower() == "missing":
                missing_section_ids.add(int(row[0]))
        spine_table_end = len(rows)
        if validation_format_version == '3':
            try:
                spine_table_end = rows.index(
                    ['table', 'subsection'],
                    spine_marker_index + 2,
                )
            except ValueError as exc:
                raise ValueError(f'Missing subsection table in {validation_csv_path}') from exc
        for row in rows[spine_marker_index + 2:spine_table_end]:
            section_id = int(row[spine_field_indices["Section ID"]])
            spine_id = int(row[spine_field_indices[
                "Global Spine ID" if explicit_identity else "Spine ID"
            ]])
            labels = [
                label
                for column, label in error_columns.items()
                if row[spine_field_indices[column]].strip().lower() == "yes"
            ]
            if labels:
                issue_labels[(section_id, spine_id)] = "; ".join(labels)
    else:
        issue_records = {}
        with validation_csv_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                record_type = (row.get("record_type") or "").strip().lower()
                status = (row.get("status") or "").strip().lower()
                if record_type == "section" and status == "missing":
                    missing_section_ids.add(int(row["section_id"]))
                elif record_type in ISSUE_LABELS and status == "yes":
                    key = (int(row["section_id"]), int(row["spine_index"]))
                    issue_records.setdefault(key, []).append(ISSUE_LABELS[record_type])

        morphology = data_loading.load_spiny_morphology(morphology_path)
        spine_ids_by_section = {}
        for (section_id, spine_index), labels in issue_records.items():
            spine_ids = spine_ids_by_section.setdefault(
                section_id,
                list(morphology.spines.spine_indices_for_section(section_id + 1)),
            )
            if 0 <= spine_index < len(spine_ids):
                issue_labels[(section_id, int(spine_ids[spine_index]))] = "; ".join(labels)

    section_images = []
    spine_images = []
    for path in image_directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parsed = _parse_screenshot(path)
        if parsed is None:
            continue
        if parsed[0] == "section":
            if int(parsed[1]) in missing_section_ids:
                section_images.append((path, parsed[1], parsed[3]))
        elif issue_labels.get((int(parsed[1]), int(parsed[2]))):
            spine_images.append((path, parsed[1], parsed[2], parsed[3]))

    section_images.sort(
        key=lambda item: (
            _identifier_sort_key(item[1]),
            item[2],
            item[0].name.lower(),
        )
    )
    spine_images.sort(
        key=lambda item: (
            _identifier_sort_key(item[1]),
            _identifier_sort_key(item[2]),
            item[3],
            item[0].name.lower(),
        )
    )

    report_dimension = max_dimension or 1600
    title_font = _load_font(max(28, int(report_dimension / 24)))
    caption_font = _load_font(max(22, int(report_dimension / 52)))

    def make_title_page(title, subtitle):
        width = report_dimension
        page = Image.new("RGB", (width, int(width * 0.62)), "white")
        draw = ImageDraw.Draw(page)
        draw.multiline_text(
            (width // 2, page.height // 2 - 20),
            title,
            font=title_font,
            fill="black",
            anchor="mm",
            align="center",
        )
        if subtitle:
            draw.text(
                (width // 2, page.height // 2 + 55),
                subtitle,
                font=caption_font,
                fill="#555555",
                anchor="mm",
            )
        return page

    def make_image_page(image_path, caption):
        with Image.open(image_path) as source_image:
            image = ImageOps.exif_transpose(source_image).convert("RGB")
            if max_dimension is not None:
                image.thumbnail(
                    (max_dimension, max_dimension),
                    Image.Resampling.LANCZOS,
                )
            header_height = max(
                90,
                int(report_dimension / 12) * (caption.count("\n") + 1),
            )
            page = Image.new(
                "RGB",
                (image.width, image.height + header_height),
                "white",
            )
            draw = ImageDraw.Draw(page)
            draw.multiline_text(
                (24, header_height // 2),
                caption,
                font=caption_font,
                fill="black",
                anchor="lm",
                spacing=8,
            )
            page.paste(image, (0, header_height))
            image.close()
            return page

    pages = [
        make_title_page(
            "Part 1 - Analysis results",
            f"{len(analysis_image_paths)} analysis plot(s)",
        ),
    ]
    pages.extend(
        make_image_page(path, path.stem.replace('_', ' ').title())
        for path in analysis_image_paths
    )
    pages.append(
        make_title_page(
            "Part 2 - Sections with missing spines",
            f"{len(section_images)} section image(s)",
        )
    )
    pages.extend(
        make_image_page(path, f"Section {section_id}")
        for path, section_id, _sequence in section_images
    )
    pages.append(
        make_title_page(
            "Part 3 - Spine issues",
            f"{len(spine_images)} spine image(s)",
        )
    )

    spine_counts = {}
    for _path, section_id, spine_id, _sequence in spine_images:
        key = (section_id, spine_id)
        spine_counts[key] = spine_counts.get(key, 0) + 1
    spine_occurrences = {}
    for path, section_id, spine_id, _sequence in spine_images:
        key = (section_id, spine_id)
        spine_occurrences[key] = spine_occurrences.get(key, 0) + 1
        caption = f"Section {section_id} - Spine {spine_id}"
        if spine_counts[key] > 1:
            caption += f" ({spine_occurrences[key]})"
        caption += f"\n{issue_labels[(int(section_id), int(spine_id))]}"
        pages.append(make_image_page(path, caption))

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pages[0].save(
            pdf_path,
            "PDF",
            save_all=True,
            append_images=pages[1:],
            resolution=150.0,
            quality=jpeg_quality,
            optimize=True,
        )
    finally:
        for page in pages:
            page.close()

    print(
        f"Created validation report with {len(pages)} pages "
        f"({len(section_images)} missing-section images, "
        f"{len(spine_images)} issue-spine images): {pdf_path}"
    )
    return pdf_path

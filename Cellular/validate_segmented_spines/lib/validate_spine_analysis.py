from pathlib import Path
from io import BytesIO
import csv
import re

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from PIL import Image as PILImage


FONT_DIRECTORY = Path('lib') / 'fonts'


def _load_validation_records(csv_path):
    """Load legacy or current validation CSV data in analysis-record form."""
    csv_path = Path(csv_path)
    with csv_path.open('r', newline='', encoding='utf-8') as handle:
        rows = [row for row in csv.reader(handle) if row]

    versioned_csv = (
        rows
        and len(rows[0]) == 2
        and rows[0][0] == 'validation_format'
        and rows[0][1] in {'2', '3', '4', '5'}
    )
    if not versioned_csv:
        return pd.read_csv(csv_path)
    validation_format_version = rows[0][1]

    try:
        section_marker_index = rows.index(['table', 'section'])
        section_header = rows[section_marker_index + 1]
        spine_marker_index = rows.index(['table', 'spine'])
        spine_header = rows[spine_marker_index + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f'Invalid validation table in {csv_path}') from exc

    section_field_indices = {
        field_name: section_header.index(field_name)
        for field_name in section_header
    }
    missing_section_field = 'Missing Segmented Spines'
    if missing_section_field not in section_field_indices:
        raise ValueError(f'Unexpected section table headers in {csv_path}')

    expected_spine_header = [
        'Section ID', 'Local Spine ID', 'Global Spine ID', 'Validity',
        'Correct Type', 'False Positive', 'Incomplete Spine', 'Falsely Extended',
        'Merged Spine', 'Split Spine',
    ]
    current_spine_header = [
        'Section ID', 'Local Spine ID', 'Global Spine ID', 'Validity',
        'Correct Type', 'Valid Structure', 'False Positive',
        'Incomplete Spine', 'Falsely Extended', 'Merged Spine', 'Split Spine',
    ]
    previous_spine_header = [
        'Section ID', 'Local Spine ID', 'Global Spine ID', 'Validity',
        'False Positive', 'Incomplete Spine', 'Falsely Extended',
        'Merged Spine', 'Split Spine',
    ]
    legacy_spine_header = [
        'Section ID', 'Spine ID', 'Validity', 'Correct Type',
        'False Positive', 'Incomplete Spine', 'Falsely Extended',
        'Merged Spine', 'Split Spine',
    ]
    previous_legacy_spine_header = [
        'Section ID', 'Spine ID', 'Validity', 'False Positive',
        'Incomplete Spine', 'Falsely Extended', 'Merged Spine',
        'Split Spine',
    ]
    if spine_header not in (
        expected_spine_header,
        current_spine_header,
        previous_spine_header,
        legacy_spine_header,
        previous_legacy_spine_header,
    ):
        raise ValueError(f'Unexpected spine table headers in {csv_path}')
    explicit_identity = spine_header in (
        expected_spine_header,
        current_spine_header,
        previous_spine_header,
    )
    has_correct_type = 'Correct Type' in spine_header
    spine_field_indices = {
        field_name: spine_header.index(field_name)
        for field_name in spine_header
    }

    records = []
    for row in rows[section_marker_index + 2:spine_marker_index]:
        if len(row) != len(section_header):
            raise ValueError(f'Invalid section row in {csv_path}')
        records.append({
            'record_type': 'section',
            'section_id': row[section_field_indices['Section']],
            'spine_index': '',
            'spine_id': '',
            'status': row[section_field_indices[missing_section_field]].strip().lower(),
        })

    issue_columns = {
        'False Positive': 'false_positive',
        'Incomplete Spine': 'incomplete_spine',
        'Falsely Extended': 'false_positive_quality',
        'Merged Spine': 'merged_spine',
        'Split Spine': 'split_spine',
    }
    spine_indices = {}
    spine_table_end = len(rows)
    if validation_format_version in {'3', '4', '5'}:
        try:
            spine_table_end = rows.index(
                ['table', 'subsection'],
                spine_marker_index + 2,
            )
        except ValueError as exc:
            raise ValueError(f'Missing subsection table in {csv_path}') from exc
    for row in rows[spine_marker_index + 2:spine_table_end]:
        if len(row) != len(spine_header):
            raise ValueError(f'Invalid spine row in {csv_path}')
        section_id = row[spine_field_indices['Section ID']]
        global_spine_id = row[spine_field_indices[
            'Global Spine ID' if explicit_identity else 'Spine ID'
        ]]
        if explicit_identity:
            spine_index = int(row[spine_field_indices['Local Spine ID']])
        else:
            spine_index = spine_indices.get(section_id, 0)
        spine_indices[section_id] = spine_index + 1
        records.append({
            'record_type': 'spine',
            'section_id': section_id,
            'spine_index': spine_index,
            'spine_id': global_spine_id,
            'status': row[spine_field_indices['Validity']].strip().lower(),
        })
        if has_correct_type:
            records.append({
                'record_type': 'correct_type',
                'section_id': section_id,
                'spine_index': spine_index,
                'spine_id': global_spine_id,
                'status': row[spine_field_indices['Correct Type']].strip().lower(),
            })
        for field_name, record_type in issue_columns.items():
            records.append({
                'record_type': record_type,
                'section_id': section_id,
                'spine_index': spine_index,
                'spine_id': global_spine_id,
                'status': row[spine_field_indices[field_name]].strip().lower(),
            })

    return pd.DataFrame(
        records,
        columns=['record_type', 'section_id', 'spine_index', 'spine_id', 'status'],
    )


def generate_report(
    images_dir,
    csv_path,
    fonts_dir,
    output_dir,
    render_dpi=150,
    export_dpi=150,
    snapshot_jpeg_quality=75,
):
    """Generate the combined analysis PDF report.

    Parameters
    ----------
    images_dir : path-like
        Directory containing the ``section_*.png`` snapshot images.
    csv_path : path-like
        Path to the validation CSV file.
    fonts_dir : path-like
        Directory containing ``font_regular.otf``.
    output_dir : path-like
        Directory the PNG/PDF outputs are written to.
    render_dpi, export_dpi : int, optional
        DPI used when rendering and exporting the matplotlib figures.
    snapshot_jpeg_quality : int, optional
        JPEG quality used when embedding snapshot images in the PDF.
    """
    images_dir = Path(images_dir)
    csv_path = Path(csv_path)
    fonts_dir = FONT_DIRECTORY if fonts_dir is None else Path(fonts_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f'{csv_path.stem}_stacked_status.png'
    combined_pdf_path = output_dir / f'{csv_path.stem}_analysis.pdf'
    error_png_path = output_dir / f'{csv_path.stem}_stacked_error_types.png'

    font_path = fonts_dir / 'font_regular.otf'
    if not font_path.is_file():
        raise FileNotFoundError(f'font_regular.otf was not found in {fonts_dir}.')
    font_manager.fontManager.addfont(str(font_path))
    font_family = font_manager.FontProperties(fname=str(font_path)).get_name()

    plt.rcParams.update({
        'font.family': font_family,
        'font.weight': 'normal',
        'axes.titleweight': 'normal',
        'axes.labelweight': 'normal',
        'axes.titlesize': 21,
        'axes.labelsize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 11,
    })

    validation = _load_validation_records(csv_path)
    required_columns = {'record_type', 'section_id', 'status'}
    missing_columns = required_columns.difference(validation.columns)
    if missing_columns:
        raise ValueError(f'CSV is missing required columns: {sorted(missing_columns)}')

    spine_rows = validation.loc[validation['record_type'].eq('spine')].copy()
    if spine_rows.empty:
        raise ValueError(f'No record_type=spine rows found in {csv_path}')
    spine_rows['section_id'] = pd.to_numeric(
        spine_rows['section_id'], errors='raise'
    ).astype(int)
    spine_rows['spine_index'] = pd.to_numeric(
        spine_rows['spine_index'], errors='raise'
    ).astype(int)
    spine_rows['spine_id'] = pd.to_numeric(
        spine_rows['spine_id'], errors='raise'
    ).astype(int)
    spine_identity = {
        (int(row.section_id), int(row.spine_id)): int(row.spine_index)
        for row in spine_rows.itertuples()
    }
    raw_status = spine_rows['status'].fillna('').astype(str).str.strip().str.lower()
    status_aliases = {
        'valid': 'valid',
        'invalid': 'invalid',
        'unset': 'unset',
        'not set': 'unset',
        '': 'unset',
    }
    unknown_statuses = sorted(set(raw_status).difference(status_aliases))
    if unknown_statuses:
        raise ValueError(f'Unexpected spine statuses in CSV: {unknown_statuses}')
    spine_rows['status'] = raw_status.map(status_aliases)

    # Every section present in the CSV is retained, including sections with
    # only invalid or only unset rows.
    section_ids = sorted(spine_rows['section_id'].unique())
    counts = (
        spine_rows.groupby(['section_id', 'status']).size()
        .unstack(fill_value=0)
        .reindex(index=section_ids, columns=['valid', 'invalid', 'unset'], fill_value=0)
    )
    counts = counts.astype(int)

    # Clean, image-first stacked rendering.
    background = '#F7F9FC'
    ink = '#000000'
    muted_ink = '#000000'
    grid = '#D9E2EC'
    colors = {
        'valid': '#168A55',
        'invalid': '#D1495B',
        'unset': '#AAB4C3',
    }
    labels = {
        'valid': 'Valid',
        'invalid': 'Invalid',
        'unset': 'Not Set',
    }

    section_count = len(section_ids)
    total_spines = int(counts.to_numpy().sum())
    validated_spines = int(counts[['valid', 'invalid']].to_numpy().sum())
    unset_spines = int(counts['unset'].sum())
    fig_width = max(7.5, min(24, 0.34 * section_count + 2.4))
    fig, ax = plt.subplots(figsize=(fig_width, 8.2), dpi=render_dpi)
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)
    x = list(range(section_count))
    bottom = pd.Series(0, index=counts.index, dtype='int64')

    for status in ('valid', 'invalid', 'unset'):
        values = counts[status].to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom.to_numpy(),
            width=0.78,
            color=colors[status],
            label=labels[status],
            edgecolor=background,
            linewidth=0.9,
        )
        bottom += counts[status]

    # Add segment labels only when there is enough vertical space.
    for section_position, section_id in enumerate(section_ids):
        cumulative = 0
        for status in ('valid', 'invalid', 'unset'):
            value = int(counts.loc[section_id, status])
            if value >= max(3, int(bottom[section_id] * 0.08)):
                ax.text(
                    section_position,
                    cumulative + value / 2,
                    str(value),
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='black',
                )
            cumulative += value

    fig.text(0.025, 0.985, 'Section Analysis - Spine Validity', fontsize=21, color=ink, ha='left', va='top')
    fig.text(
        0.025,
        0.94,
        f'{section_count} Sections  •  {validated_spines:,}/{total_spines:,} spines proofread  •  {unset_spines:,} not set',
        color=muted_ink,
        fontsize=11,
        ha='left',
        va='top',
    )
    ax.set_xlabel('Section', labelpad=12, color=muted_ink)
    ax.set_ylabel('Number of spines', labelpad=12, color=muted_ink)
    ax.set_xticks(x, [str(section_id) for section_id in section_ids])
    ax.tick_params(axis='x', rotation=45, colors=muted_ink, length=0, pad=7)
    ax.tick_params(axis='y', colors=muted_ink, length=0)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color=grid, linewidth=0.8, alpha=0.85)
    ax.margins(x=0.01)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#BCCCDC')
    ax.legend(
        ncol=3,
        loc='upper right',
        bbox_to_anchor=(1, 1.09),
        frameon=False,
        labelcolor=muted_ink,
    )
    fig.tight_layout(rect=(0.005, 0, 0.995, 0.89), pad=0.4)

    # Save the rendered image first, then make the PDF from that exact image.
    fig.savefig(png_path, dpi=export_dpi, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    # Detailed error findings by section. Only confirmed 'yes' records are
    # counted, so the chart shows actual error findings rather than 'no' reviews.
    error_type_labels = {
        'false_positive': 'False Positive',
        'incomplete_spine': 'Incomplete Spine',
        'false_positive_quality': 'Falsely Extended',
        'merged_spine': 'Merged Spine',
        'split_spine': 'Split Spine',
    }
    error_type_keys = list(error_type_labels)
    error_rows = validation.loc[validation['record_type'].isin(error_type_keys)].copy()
    error_rows['section_id'] = pd.to_numeric(
        error_rows['section_id'], errors='raise'
    ).astype(int)
    error_rows['status'] = error_rows['status'].fillna('').astype(str).str.strip().str.lower()
    if 'spine_id' not in error_rows:
        error_rows['spine_id'] = error_rows['spine_index']
    error_rows['spine_id'] = pd.to_numeric(error_rows['spine_id'], errors='coerce')
    confirmed_error_rows = error_rows.loc[
        error_rows['status'].eq('yes')
    ].dropna(subset=['spine_id'])
    error_labels_by_spine = {}
    for (section_id, spine_id), grouped_rows in confirmed_error_rows.groupby(
        ['section_id', 'spine_id'], sort=False
    ):
        error_labels_by_spine[(int(section_id), int(spine_id))] = '; '.join(
            error_type_labels[record_type]
            for record_type in error_type_keys
            if record_type in set(grouped_rows['record_type'])
        )
    error_counts = pd.DataFrame(
        0, index=section_ids, columns=error_type_keys, dtype='int64'
    )
    for error_type in error_type_keys:
        confirmed = error_rows.loc[
            error_rows['record_type'].eq(error_type)
            & error_rows['status'].eq('yes')
        ]
        if not confirmed.empty:
            grouped = confirmed.groupby('section_id').size()
            error_counts.loc[grouped.index, error_type] = grouped.astype('int64')

    error_colors = {
        'false_positive': '#E76F51',
        'incomplete_spine': '#F4A261',
        'false_positive_quality': '#E9C46A',
        'merged_spine': '#2A9D8F',
        'split_spine': '#457B9D',
    }
    error_total = int(error_counts.to_numpy().sum())
    error_fig, error_ax = plt.subplots(
        figsize=(fig_width, 8.2),
        dpi=render_dpi,
    )
    error_fig.patch.set_facecolor(background)
    error_ax.set_facecolor(background)
    error_bottom = pd.Series(0, index=error_counts.index, dtype='int64')
    for error_type in error_type_keys:
        values = error_counts[error_type].to_numpy()
        error_ax.bar(
            x,
            values,
            bottom=error_bottom.to_numpy(),
            width=0.78,
            color=error_colors[error_type],
            label=error_type_labels[error_type],
            edgecolor=background,
            linewidth=0.9,
        )
        error_bottom += error_counts[error_type]

    for section_position, section_id in enumerate(section_ids):
        cumulative = 0
        for error_type in error_type_keys:
            value = int(error_counts.loc[section_id, error_type])
            if value >= max(2, int(error_bottom[section_id] * 0.10)):
                error_ax.text(
                    section_position,
                    cumulative + value / 2,
                    str(value),
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='black',
                )
            cumulative += value

    fig_x = 0.025
    error_fig.text(
        fig_x, 0.985, 'Section Analysis - Detailed Error Types',
        fontsize=21, color=ink, ha='left', va='top'
    )
    error_fig.text(
        fig_x, 0.94,
        f'{section_count} Sections  •  {error_total:,} confirmed error findings',
        color=muted_ink, fontsize=11, ha='left', va='top'
    )
    error_ax.set_xlabel('Section', labelpad=12, color=muted_ink)
    error_ax.set_ylabel('Number of error findings', labelpad=12, color=muted_ink)
    error_ax.set_xticks(x, [str(section_id) for section_id in section_ids])
    error_ax.tick_params(axis='x', rotation=45, colors=muted_ink, length=0, pad=7)
    error_ax.tick_params(axis='y', colors=muted_ink, length=0)
    error_ax.set_axisbelow(True)
    error_ax.grid(axis='y', color=grid, linewidth=0.8, alpha=0.85)
    error_ax.margins(x=0.01)
    error_ax.spines[['top', 'right', 'left']].set_visible(False)
    error_ax.spines['bottom'].set_color('#BCCCDC')
    error_ax.legend(
        ncol=5, loc='upper right', bbox_to_anchor=(1, 1.09),
        frameon=False, labelcolor=muted_ink,
    )
    error_fig.tight_layout(rect=(0.005, 0, 0.995, 0.89), pad=0.4)
    error_fig.savefig(
        error_png_path, dpi=export_dpi, facecolor=error_fig.get_facecolor(),
        bbox_inches='tight', pad_inches=0.04
    )
    plt.close(error_fig)

    # Build one complete report PDF in streaming page order.
    section_record_status = validation.loc[
        validation['record_type'].eq('section')
    ].copy()
    section_record_status['section_id'] = pd.to_numeric(
        section_record_status['section_id'], errors='raise'
    ).astype(int)
    missing_section_ids = set(
        section_record_status.loc[
            section_record_status['status'].fillna('').astype(str).str.lower().eq('missing'),
            'section_id',
        ]
    )
    section_snapshot_pattern = re.compile(
        r'^section_(\d+)(?:_(?:error|structure))?_(\d+)\.png$'
    )
    spine_snapshot_pattern = re.compile(
        r'^section_(\d+)_spine_(\d+)(?:_(?:error|structure|structure_snapshot))?_(\d+)\.png$'
    )
    section_snapshots = []
    spine_snapshots = []
    for image_path in images_dir.glob('section_*.png'):
        section_match = section_snapshot_pattern.match(image_path.name)
        if section_match is not None:
            section_id, snapshot_id = map(int, section_match.groups())
            if section_id in missing_section_ids:
                section_snapshots.append((section_id, snapshot_id, image_path))
            continue
        spine_match = spine_snapshot_pattern.match(image_path.name)
        if spine_match is not None:
            section_id, spine_id, snapshot_id = map(int, spine_match.groups())
            spine_snapshots.append((section_id, spine_id, snapshot_id, image_path))
    section_snapshots.sort(key=lambda item: (item[0], item[1], item[2].name))
    spine_snapshots.sort(key=lambda item: (item[0], item[1], item[2], item[3].name))

    def add_full_image_page(report, image_path):
        """Append an analysis plot image as a full PDF page."""
        with PILImage.open(image_path) as image:
            image = image.convert('RGB')
            image_ratio = image.width / image.height
        page_height = 20 / image_ratio
        page = plt.figure(figsize=(20, page_height), facecolor='white')
        axis = page.add_axes([0, 0, 1, 1])
        axis.imshow(image)
        axis.axis('off')
        report.savefig(page, dpi=export_dpi, facecolor='white')
        plt.close(page)

    def add_divider_page(report, title, subtitle):
        """Append a titled divider page to the report."""
        page = plt.figure(figsize=(11.69, 8.27), facecolor='white')
        page.text(0.08, 0.60, title, fontsize=30, color='black', ha='left')
        page.text(0.08, 0.52, subtitle, fontsize=14, color='black', ha='left')
        report.savefig(page, dpi=export_dpi, facecolor='white')
        plt.close(page)

    def add_snapshot_page(report, image_path, title, page_text=None):
        """Append one titled JPEG-compressed snapshot image to the report."""
        jpeg_buffer = BytesIO()
        with PILImage.open(image_path) as source_image:
            image = source_image.convert('RGB')
            image.save(
                jpeg_buffer,
                format='JPEG',
                quality=snapshot_jpeg_quality,
                optimize=False,
            )
        jpeg_buffer.seek(0)
        with PILImage.open(jpeg_buffer) as compressed_image:
            image = compressed_image.copy()
        page = plt.figure(figsize=(11.69, 8.27), facecolor='white')
        title_fontsize = 16 if len(title) <= 75 else 12
        page.text(
            0.04, 0.965, title, fontsize=title_fontsize, color='black',
            ha='left', va='top'
        )
        if page_text:
            page.text(
                0.04, 0.925, page_text, fontsize=11, color='black',
                ha='left', va='top'
            )
        axis = page.add_axes([0.04, 0.04, 0.92, 0.86])
        axis.imshow(image)
        axis.axis('off')
        report.savefig(page, dpi=export_dpi, facecolor='white')
        plt.close(page)

    with PdfPages(combined_pdf_path) as report:
        add_full_image_page(report, png_path)
        add_full_image_page(report, error_png_path)
        add_divider_page(
            report,
            'Missing Spines Analysis',
            f'{len(section_snapshots)} snapshots from sections marked as missing segmented spines',
        )
        for section_id, snapshot_id, image_path in section_snapshots:
            add_snapshot_page(
                report, image_path,
                f'Section {section_id} : Snapshot {snapshot_id}',
            )
        add_divider_page(
            report,
            'Spine Issues Analysis',
            f'{len(spine_snapshots)} snapshots of detailed spine issues',
        )
        for section_id, spine_id, snapshot_id, image_path in spine_snapshots:
            local_id = spine_identity[(section_id, spine_id)]
            title = (
                f'Section {section_id}, Spine {local_id} ({spine_id}), '
                f'Snapshot {snapshot_id}'
            )
            error_text = error_labels_by_spine.get(
                (section_id, spine_id),
                'None recorded',
            )
            add_snapshot_page(
                report,
                image_path,
                title,
                page_text=f'Errors: {error_text}',
            )

    return combined_pdf_path


if __name__ == '__main__':
    csv_path = Path('864691134886335738_validation.csv')
    generate_report(
        images_dir=csv_path.parent,
        csv_path=csv_path,
        fonts_dir=Path('lib') / 'fonts',
        output_dir=csv_path.parent,
    )

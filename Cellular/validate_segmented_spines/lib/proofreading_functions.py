"""Functional and state logic for the spine proofreading interface.

This module contains the schema, morphology loading, subsection construction,
and state-derived values used by the presentation layer. It intentionally does
not import ipywidgets or k3d so the data model can be reused independently of
the notebook UI.
"""

import csv
import os
import random
import tempfile
from pathlib import Path

import numpy as np

from morph_spines_visualizer.core import data_loading, geometry
from morph_spines_visualizer.core import spines as spines_lib


ANALYSIS_FIELDS = [
    ('correct_type', 'Incorrect Type'),
    ('false_positive', 'False positive'),
    ('incomplete_spine', 'Incomplete spine'),
    ('falsely_extended', 'Falsely extended'),
    ('merged_spine', 'Merged spine'),
    ('split_spine', 'Split spine'),
]
ISSUE_FIELDS = [key for key, _ in ANALYSIS_FIELDS if key != 'correct_type']
VALID_STRUCTURE_FIELD = ('valid_structure', 'Invalid Structure')
ALL_ANSWER_KEYS = [key for key, _ in ANALYSIS_FIELDS] + [VALID_STRUCTURE_FIELD[0]]
# Invalid Structure is informational and must not determine spine validity.
VALIDITY_FIELDS = [
    key for key in ALL_ANSWER_KEYS if key != VALID_STRUCTURE_FIELD[0]
]

SPINE_TYPES = ['thin', 'stubby', 'mushroom', 'branched', 'filopodia']

TOTAL_SECTIONS = 63
TOTAL_SPINES = 1047
NEURON_ID = '864691134886335738'


def get_neuron_id_from_mesh_path(mesh_path):
    """Return the neuron root encoded by a downloaded mesh filename."""
    if mesh_path is None:
        return None
    stem = Path(mesh_path).stem.strip()
    if not stem:
        return None
    return stem.split('_', 1)[0] or stem


VALIDATION_SCHEMA_VERSION = '5'
SUPPORTED_VALIDATION_SCHEMA_VERSIONS = {'2', '3', '4', '5'}
VALIDITY_NOT_SET = 'Not Set'
SECTION_CSV_FIELDS = [
    'Section', 'Number Spines', 'Validated (Yes or No)',
    'Remaining Spines to Validate', 'Incorrect Type', 'False Positives',
    'Incomplete Spines', 'Falsely Extended Spines',
    'Merged Spines', 'Split Spines', 'Invalid Structure',
    'Missing Segmented Spines',
]
SPINE_CSV_FIELDS = [
    'Section ID', 'Local Spine ID', 'Global Spine ID', 'Validity',
    'Incorrect Type', 'Invalid Structure', 'False Positive',
    'Incomplete Spine', 'Falsely Extended', 'Merged Spine', 'Split Spine',
]
PREVIOUS_CURRENT_SPINE_CSV_FIELDS = [
    'Section ID', 'Local Spine ID', 'Global Spine ID', 'Validity',
    'Correct Type', 'Valid Structure', 'False Positive',
    'Incomplete Spine', 'Falsely Extended', 'Merged Spine', 'Split Spine',
]
SUBSECTION_CSV_FIELDS = [
    'Section ID', 'Subsection Index', 'Missing Spine Count', 'Status',
]
PREVIOUS_SECTION_CSV_FIELDS = [
    'Section', 'Number Spines', 'Validated (Yes or No)',
    'Remaining Spines to Validate', 'False Positives',
    'Incomplete Spines', 'Falsely Extended Spines',
    'Merged Spines', 'Split Spines', 'Missing Segmented Spines',
]
PREVIOUS_SPINE_CSV_FIELDS = [
    'Section ID', 'Local Spine ID', 'Global Spine ID', 'Validity',
    'False Positive', 'Incomplete Spine', 'Falsely Extended',
    'Merged Spine', 'Split Spine',
]
LEGACY_SPINE_CSV_FIELDS = [
    'Section ID', 'Spine ID', 'Validity', 'Correct Type',
    'False Positive', 'Incomplete Spine', 'Falsely Extended',
    'Merged Spine', 'Split Spine',
]
PREVIOUS_LEGACY_SPINE_CSV_FIELDS = [
    'Section ID', 'Spine ID', 'Validity', 'False Positive',
    'Incomplete Spine', 'Falsely Extended', 'Merged Spine', 'Split Spine',
]
LEGACY_VALIDATION_CSV_FIELDS = [
    'record_type', 'schema_version', 'source_id',
    'section_id', 'spine_index', 'status',
]


# ============================================================
# Validation persistence helpers
# ============================================================

def normalize_missing_spines_status(status):
    """Normalize legacy missing-spine labels to the canonical three states."""
    if status is None:
        return 'Not Set'
    normalized = str(status).strip().casefold().replace('_', ' ')
    if normalized.startswith('missing'):
        return 'Missing'
    if normalized in {'no missing', 'no missing spines'}:
        return 'No Missing'
    return 'Not Set'


def get_validation_csv_path(mesh_path=None, morphology_path=None, validation_csv_path=None):
    """Return the legacy validation path, or ``None`` when persistence is disabled."""
    if validation_csv_path is not None:
        return Path(validation_csv_path)
    if mesh_path is None or morphology_path is None:
        return None
    mesh_path = Path(mesh_path)
    morphology_path = Path(morphology_path)
    output_dir = mesh_path.with_name(f'{mesh_path.stem}_proofreading')
    return output_dir / f'{morphology_path.stem}_validation.csv'


def _normalize_answer(value):
    """Return an internal yes/no answer, or ``None`` for an unset value."""
    if value is None:
        return None
    normalized = str(value).strip().casefold()
    return normalized if normalized in {'yes', 'no'} else None


def _normalize_validity(value):
    """Return an internal valid/invalid value, or ``None`` for an unset value."""
    if value is None:
        return None
    normalized = str(value).strip().casefold()
    return normalized if normalized in {'valid', 'invalid'} else None


def _serialize_answer(value):
    """Serialize a binary answer; unset answers are persisted as No."""
    normalized = _normalize_answer(value)
    return normalized.title() if normalized is not None else 'No'


def _serialize_validity(value):
    normalized = _normalize_validity(value)
    return normalized.title() if normalized is not None else VALIDITY_NOT_SET


def derive_spine_validity(spine):
    """Derive validity from error answers, excluding Invalid Structure.

    Any explicit Yes in a validity-driving field makes the spine invalid.
    A spine is valid only after every validity-driving field has an explicit
    No; Invalid Structure is intentionally excluded from both checks.
    """
    answers = spine.get('answers', {})
    if any(answers.get(key) == 'yes' for key in VALIDITY_FIELDS):
        return 'invalid'
    if any(answers.get(key) not in {'yes', 'no'} for key in VALIDITY_FIELDS):
        return None
    return 'valid'


def effective_spine_validity(spine):
    """Return the persisted validity of an explicitly completed spine."""
    if not spine.get('checked', False):
        return None
    return _normalize_validity(spine.get('validity'))


# ============================================================
# Mock and morphology data
# ============================================================

def _make_section(section_index, rng, global_id_start):
    """Build one mock section with a full spine list and subsections."""
    if section_index == 0:
        spine_count = 11
    else:
        spine_count = rng.randint(6, 28)

    spines = []
    for i in range(spine_count):
        spine_type = 'branched' if (section_index == 0 and i == 0) else rng.choice(SPINE_TYPES)
        spines.append({
            'index': i,
            'global_id': global_id_start + i,
            'type': spine_type,
            'answers': {key: None for key in ALL_ANSWER_KEYS},
            'checked': False,
            'validity': None,
        })

    if section_index == 0:
        spines[0]['answers'].update({
            'correct_type': 'yes',
            'false_positive': 'no',
            'incomplete_spine': 'no',
            'falsely_extended': 'yes',
            'merged_spine': 'no',
            'split_spine': 'no',
            'valid_structure': 'no',
        })
        for spine in spines[:5]:
            spine['checked'] = True

    if section_index == 0:
        subsections = [{'index': j, 'missing_count': 0, 'done': True} for j in range(4)]
        subsections[3]['missing_count'] = 2
    else:
        subsection_count = max(1, -(-spine_count // 3))
        subsections = [{'index': j, 'missing_count': 0, 'done': False} for j in range(subsection_count)]

    return {
        'index': section_index,
        'spines': spines,
        'subsections': subsections,
        'checked': False,
    }, spine_count


def _build_mock_sections(seed=7):
    rng = random.Random(seed)
    sections = []
    global_id = 1
    for i in range(TOTAL_SECTIONS):
        section, spine_count = _make_section(i, rng, global_id)
        sections.append(section)
        global_id += spine_count
    return sections


def _format_spine_type(morphology, spine_id):
    """Return the morphology spine type in the format used by the UI."""
    try:
        spine_type = morphology.spines.spine_type(int(spine_id))
        spine_type = getattr(spine_type, 'value', spine_type)
        spine_type = str(spine_type).strip()
        if spine_type.startswith('SpineType.'):
            spine_type = spine_type.rsplit('.', 1)[-1]
        return spine_type.replace('_', ' ').lower() or 'unknown'
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return 'unknown'


def _build_section_subsections(points, subsection_length_um=10.0):
    """Split an ordered section centerline into path-length subsections."""
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return []
    if len(points) == 1:
        point = points[0].copy()
        return [{
            'index': 0,
            'start_um': 0.0,
            'end_um': 0.0,
            'points': points.copy(),
            'center': point,
            'radius': 1e-6,
            'missing_count': 0,
            'done': False,
        }]

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative_lengths = np.concatenate([
        np.array([0.0], dtype=np.float64),
        np.cumsum(segment_lengths, dtype=np.float64),
    ])
    total_length = float(cumulative_lengths[-1])
    if total_length <= np.finfo(np.float64).eps:
        point = points[0].copy()
        return [{
            'index': 0,
            'start_um': 0.0,
            'end_um': 0.0,
            'points': points[:1].copy(),
            'center': point,
            'radius': 1e-6,
            'missing_count': 0,
            'done': False,
        }]

    boundaries = list(np.arange(0.0, total_length, subsection_length_um))
    if not boundaries or boundaries[-1] != total_length:
        boundaries.append(total_length)

    def point_at_distance(distance):
        return np.asarray([
            np.interp(distance, cumulative_lengths, points[:, axis])
            for axis in range(3)
        ], dtype=np.float32)

    subsections = []
    for subsection_index, start_um in enumerate(boundaries[:-1]):
        end_um = boundaries[subsection_index + 1]
        interior_mask = (
            (cumulative_lengths > start_um)
            & (cumulative_lengths < end_um)
        )
        subsection_points = np.vstack([
            point_at_distance(start_um),
            points[interior_mask],
            point_at_distance(end_um),
        ]).astype(np.float32)
        subsection_min = subsection_points.min(axis=0)
        subsection_max = subsection_points.max(axis=0)
        subsection_center = (subsection_min + subsection_max) * 0.5
        subsection_radius = max(
            float(np.linalg.norm(subsection_max - subsection_min)) * 0.6,
            1e-6,
        )
        subsections.append({
            'index': subsection_index,
            'start_um': float(start_um),
            'end_um': float(end_um),
            'points': subsection_points,
            'center': subsection_center.astype(np.float32),
            'radius': subsection_radius,
            'missing_count': 0,
            'done': False,
        })
    return subsections


def _empty_subsection(index=0):
    """Return a safe placeholder for a section without centerline geometry."""
    point = np.zeros(3, dtype=np.float32)
    return {
        'index': index,
        'start_um': 0.0,
        'end_um': 0.0,
        'points': np.empty((0, 3), dtype=np.float32),
        'center': point,
        'radius': 1e-6,
        'missing_count': 0,
        'done': False,
    }


def _load_real_sections(morphology_path):
    """Load the same spiny-section navigation domain used by the core UI."""
    morphology = data_loading.load_spiny_morphology(morphology_path)
    section_points = geometry.get_sections_points(morphology)
    section_ids_with_counts = (
        spines_lib.get_section_ids_with_spine_counts_for_sections_with_spines(
            morphology=morphology
        )
    )
    sections = []
    for section_index, (section_id, _spine_count) in enumerate(section_ids_with_counts):
        spine_ids = list(morphology.spines.spine_indices_for_section(section_id + 1))
        spines = [
            {
                'index': local_index,
                'global_id': int(spine_id),
                'type': _format_spine_type(morphology, spine_id),
                'answers': {key: None for key in ALL_ANSWER_KEYS},
                'checked': False,
                'validity': None,
            }
            for local_index, spine_id in enumerate(spine_ids)
        ]
        subsections = _build_section_subsections(
            section_points.get(int(section_id), np.empty((0, 3), dtype=np.float32))
        )
        if not subsections:
            subsections = [_empty_subsection()]
        sections.append({
            'index': section_index,
            'section_id': int(section_id),
            'spines': spines,
            'subsections': subsections,
            'checked': False,
            'missing_status': 'Not Set',
        })

    return {
        'morphology': morphology,
        'sections': sections,
        'total_sections': len(section_ids_with_counts),
        'total_spines': int(morphology.spines.spine_count),
        'neuron_id': Path(morphology_path).stem,
    }


# ============================================================
# State and derived values
# ============================================================

class DesignState:
    """State driving the redesigned UI, with optional real morphology data."""

    def __init__(self, mesh_path=None, morphology_path=None, validation_csv_path=None):
        self.mesh_path = None if mesh_path is None else Path(mesh_path)
        self.morphology_path = None if morphology_path is None else Path(morphology_path)
        self.validation_csv_path = get_validation_csv_path(
            mesh_path=self.mesh_path,
            morphology_path=self.morphology_path,
            validation_csv_path=validation_csv_path,
        )
        if self.validation_csv_path is not None:
            self.validation_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.morphology = None
        mesh_neuron_id = get_neuron_id_from_mesh_path(self.mesh_path)
        if self.morphology_path is None:
            self.neuron_id = mesh_neuron_id or NEURON_ID
            self.sections = _build_mock_sections()
            self.total_sections = TOTAL_SECTIONS
            self.total_spines = TOTAL_SPINES
        else:
            loaded = _load_real_sections(self.morphology_path)
            self.morphology = loaded['morphology']
            self.neuron_id = mesh_neuron_id or loaded['neuron_id']
            self.sections = loaded['sections']
            self.total_sections = loaded['total_sections']
            self.total_spines = loaded['total_spines']
        self.section_index = 0
        self.subsection_index = 0
        self.spine_index = 0
        self.show_structure = False
        self.white_background = True
        self.current_view = 'xy'
        self.status_message = 'Ready.'
        if self.validation_csv_path is not None and self.validation_csv_path.is_file():
            try:
                restore_validation_state(self, self.validation_csv_path)
            except Exception as exc:
                self.status_message = f'Could not restore saved validation state: {exc}'

    @property
    def section(self):
        return self.sections[self.section_index]

    @property
    def subsection(self):
        return self.section['subsections'][self.subsection_index]

    @property
    def spine(self):
        return self.section['spines'][self.spine_index]

    def spine_validity(self, spine):
        return effective_spine_validity(spine) or 'unset'

    def spine_flag_count(self, spine):
        return sum(1 for key in ISSUE_FIELDS if spine['answers'][key] == 'yes')

    def section_checked_count(self, section):
        return sum(1 for spine in section['spines'] if spine['checked'])

    def section_is_checked(self, section):
        """Return and persist whether a non-empty section is complete."""
        spines = section.get('spines', [])
        section['checked'] = bool(spines) and all(spine['checked'] for spine in spines)
        return section['checked']

    def sections_checked_count(self):
        return sum(1 for section in self.sections if self.section_is_checked(section))

    def total_spines_checked(self):
        return sum(self.section_checked_count(section) for section in self.sections)

    def clamp_indices(self):
        self.section_index = max(0, min(self.section_index, len(self.sections) - 1))
        subsections = self.section['subsections']
        self.subsection_index = max(0, min(self.subsection_index, len(subsections) - 1))
        spines = self.section['spines']
        self.spine_index = max(0, min(self.spine_index, len(spines) - 1))



def _section_id(section):
    return int(section.get('section_id', section['index']))


def section_summary_row(state, section):
    """Build one legacy section-summary row from the current in-memory state."""
    spines = section.get('spines', [])
    validated_count = sum(
        effective_spine_validity(spine) is not None
        for spine in spines
    )
    finding_fields = (
        ('Incorrect Type', 'correct_type'),
        ('False Positives', 'false_positive'),
        ('Incomplete Spines', 'incomplete_spine'),
        ('Falsely Extended Spines', 'falsely_extended'),
        ('Merged Spines', 'merged_spine'),
        ('Split Spines', 'split_spine'),
        ('Invalid Structure', 'valid_structure'),
    )
    row = {
        'Section': _section_id(section),
        'Number Spines': len(spines),
        'Validated (Yes or No)': (
            'Yes' if spines and validated_count == len(spines) else 'No'
        ),
        'Remaining Spines to Validate': max(len(spines) - validated_count, 0),
        'Missing Segmented Spines': normalize_missing_spines_status(
            section.get('missing_status')
        ),
    }
    for column, answer_key in finding_fields:
        row[column] = sum(
            spine.get('answers', {}).get(answer_key) == 'yes'
            for spine in spines
        )
    return row


def section_summary_rows(state):
    """Return the full mesh-analysis summary in section order."""
    return [section_summary_row(state, section) for section in state.sections]


def snapshot_validation_state(state):
    """Create an immutable three-table snapshot of all validation state."""
    section_rows = section_summary_rows(state)
    spine_rows = []
    subsection_rows = []
    for section in state.sections:
        section_id = _section_id(section)
        for local_index, spine in enumerate(section.get('spines', [])):
            spine_rows.append({
                'Section ID': section_id,
                'Local Spine ID': int(spine.get('index', local_index)),
                'Global Spine ID': int(spine['global_id']),
                'Validity': _serialize_validity(
                    effective_spine_validity(spine)
                ),
                'Incorrect Type': _serialize_answer(
                    spine.get('answers', {}).get('correct_type')
                ),
                'Invalid Structure': _serialize_answer(
                    spine.get('answers', {}).get('valid_structure')
                ),
                'False Positive': _serialize_answer(
                    spine.get('answers', {}).get('false_positive')
                ),
                'Incomplete Spine': _serialize_answer(
                    spine.get('answers', {}).get('incomplete_spine')
                ),
                'Falsely Extended': _serialize_answer(
                    spine.get('answers', {}).get('falsely_extended')
                ),
                'Merged Spine': _serialize_answer(
                    spine.get('answers', {}).get('merged_spine')
                ),
                'Split Spine': _serialize_answer(
                    spine.get('answers', {}).get('split_spine')
                ),
            })
        for subsection in section.get('subsections', []):
            subsection_rows.append({
                'Section ID': section_id,
                'Subsection Index': int(subsection['index']),
                'Missing Spine Count': int(subsection.get('missing_count', 0)),
                'Status': 'Done' if subsection.get('done') else 'Not Set',
            })
    return {
        'section_rows': section_rows,
        'spine_rows': spine_rows,
        'subsection_rows': subsection_rows,
    }


def write_validation_csv(path, snapshot):
    """Atomically and durably write all section, spine, and subsection tables."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', newline='', encoding='utf-8', delete=False,
            dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp'
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.writer(handle)
            writer.writerow(['validation_format', VALIDATION_SCHEMA_VERSION])
            writer.writerow(['table', 'section'])
            writer.writerow(SECTION_CSV_FIELDS)
            writer.writerows(
                [[row[field] for field in SECTION_CSV_FIELDS] for row in snapshot['section_rows']]
            )
            writer.writerow(['table', 'spine'])
            writer.writerow(SPINE_CSV_FIELDS)
            writer.writerows(
                [[row[field] for field in SPINE_CSV_FIELDS] for row in snapshot['spine_rows']]
            )
            writer.writerow(['table', 'subsection'])
            writer.writerow(SUBSECTION_CSV_FIELDS)
            writer.writerows(
                [[row[field] for field in SUBSECTION_CSV_FIELDS]
                 for row in snapshot['subsection_rows']]
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)
        directory_fd = os.open(str(path.parent), directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def save_validation_state(state):
    """Persist the current state, returning the path or ``None`` when disabled."""
    if state.validation_csv_path is None:
        return None
    write_validation_csv(
        state.validation_csv_path,
        snapshot_validation_state(state),
    )
    return state.validation_csv_path


def _empty_loaded_validation_state():
    return {
        'section_missing_status': {},
        'spines': {},
        'subsections': {},
    }


def _parse_int(value, description, line_number):
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid {description} at line {line_number}') from exc


def _parse_csv_answer(value, field_name, line_number):
    normalized = _normalize_answer(value)
    text = '' if value is None else str(value).strip().casefold()
    if text in {'', 'not set'}:
        return None
    if normalized is None:
        raise ValueError(f'Invalid {field_name} status at line {line_number}')
    return normalized


def _parse_csv_validity(value, line_number):
    normalized = _normalize_validity(value)
    text = '' if value is None else str(value).strip().casefold()
    if text in {'', 'not set'}:
        return None
    if normalized is None:
        raise ValueError(f'Invalid validity at line {line_number}')
    return normalized


def read_validation_csv(path, state):
    """Read a legacy/versioned CSV into state-independent validation records."""
    path = Path(path)
    if not path.is_file():
        return _empty_loaded_validation_state()

    sections_by_id = {_section_id(section): section for section in state.sections}
    spine_ids_by_section = {
        section_id: [int(spine['global_id']) for spine in section.get('spines', [])]
        for section_id, section in sections_by_id.items()
    }
    loaded = _empty_loaded_validation_state()

    with path.open('r', newline='', encoding='utf-8') as handle:
        rows = [row for row in csv.reader(handle) if row]
    if not rows:
        return loaded

    first_row = rows[0]
    is_versioned = (
        len(first_row) == 2
        and first_row[0] == 'validation_format'
        and first_row[1] in SUPPORTED_VALIDATION_SCHEMA_VERSIONS
    )
    if not is_versioned:
        if first_row != LEGACY_VALIDATION_CSV_FIELDS:
            raise ValueError(f'Unexpected validation CSV headers in {path}')
        source_id = state.morphology_path.name if state.morphology_path is not None else None
        for line_number, row in enumerate(rows[1:], start=2):
            if len(row) != len(LEGACY_VALIDATION_CSV_FIELDS):
                raise ValueError(f'Invalid legacy validation row at line {line_number}')
            record = dict(zip(LEGACY_VALIDATION_CSV_FIELDS, row))
            if record['schema_version'] != '1':
                raise ValueError(f'Unsupported legacy validation CSV version at line {line_number}')
            if source_id is not None and record['source_id'] != source_id:
                raise ValueError(f'Validation CSV source does not match {source_id}')
            section_id = _parse_int(record['section_id'], 'section ID', line_number)
            if section_id not in sections_by_id:
                raise ValueError(f'Unknown section ID {section_id} at line {line_number}')
            record_type = record['record_type']
            if record_type == 'section':
                if record['spine_index']:
                    raise ValueError(f'Invalid section record at line {line_number}')
                loaded['section_missing_status'][section_id] = normalize_missing_spines_status(
                    record['status']
                )
                continue
            spine_index = _parse_int(record['spine_index'], 'spine index', line_number)
            if not 0 <= spine_index < len(spine_ids_by_section[section_id]):
                raise ValueError(f'Out-of-range spine index at line {line_number}')
            key = (section_id, spine_index)
            spine_record = loaded['spines'].setdefault(key, {'answers': {}, 'validity': None})
            if record_type == 'spine':
                status = _normalize_validity(record['status'])
                if status is None:
                    raise ValueError(f'Invalid spine status at line {line_number}')
                spine_record['validity'] = status
            else:
                answer_key = {
                    'correct_type': 'correct_type',
                    'valid_structure': 'valid_structure',
                    'false_positive': 'false_positive',
                    'incomplete_spine': 'incomplete_spine',
                    'false_positive_quality': 'falsely_extended',
                    'merged_spine': 'merged_spine',
                    'split_spine': 'split_spine',
                }.get(record_type)
                if answer_key is None:
                    raise ValueError(f'Unknown record type at line {line_number}')
                status = _normalize_answer(record['status'])
                if status is None:
                    raise ValueError(f'Invalid {record_type} status at line {line_number}')
                spine_record['answers'][answer_key] = status
        return loaded

    version = first_row[1]
    if len(rows) < 3 or rows[1] != ['table', 'section']:
        raise ValueError(f'Invalid validation table markers in {path}')
    section_header = rows[2]
    required_section_fields = {
        'Section', 'Number Spines', 'Missing Segmented Spines',
    }
    if not required_section_fields.issubset(section_header):
        raise ValueError(f'Unexpected section table headers in {path}')
    section_field_indices = {field: section_header.index(field) for field in section_header}
    row_index = 3
    while row_index < len(rows) and rows[row_index] != ['table', 'spine']:
        row = rows[row_index]
        line_number = row_index + 1
        if len(row) != len(section_header):
            raise ValueError(f'Invalid section row at line {line_number}')
        section_id = _parse_int(row[section_field_indices['Section']], 'section ID', line_number)
        number_spines = _parse_int(
            row[section_field_indices['Number Spines']], 'section spine count', line_number
        )
        if section_id not in sections_by_id:
            raise ValueError(f'Unknown section ID {section_id} at line {line_number}')
        if number_spines != len(sections_by_id[section_id].get('spines', [])):
            raise ValueError(f'Section summary does not match morphology at line {line_number}')
        if section_id in loaded['section_missing_status']:
            raise ValueError(f'Duplicate section summary at line {line_number}')
        loaded['section_missing_status'][section_id] = normalize_missing_spines_status(
            row[section_field_indices['Missing Segmented Spines']]
        )
        row_index += 1

    if row_index >= len(rows) or rows[row_index] != ['table', 'spine']:
        raise ValueError(f'Missing spine table marker in {path}')
    row_index += 1
    if row_index >= len(rows):
        raise ValueError(f'Missing spine table headers in {path}')
    spine_header = rows[row_index]
    valid_spine_headers = {
        tuple(SPINE_CSV_FIELDS), tuple(PREVIOUS_CURRENT_SPINE_CSV_FIELDS),
        tuple(PREVIOUS_SPINE_CSV_FIELDS), tuple(LEGACY_SPINE_CSV_FIELDS),
        tuple(PREVIOUS_LEGACY_SPINE_CSV_FIELDS),
    }
    if tuple(spine_header) not in valid_spine_headers:
        raise ValueError(f'Unexpected spine table headers in {path}')
    spine_field_indices = {field: spine_header.index(field) for field in spine_header}
    has_explicit_local_id = 'Local Spine ID' in spine_header
    global_id_field = 'Global Spine ID' if has_explicit_local_id else 'Spine ID'
    row_index += 1
    try:
        subsection_marker_index = rows.index(['table', 'subsection'], row_index)
    except ValueError:
        subsection_marker_index = len(rows)
    seen_spines = set()
    answer_columns = {
        'Incorrect Type': 'correct_type',
        'Invalid Structure': 'valid_structure',
        'Correct Type': 'correct_type',
        'Valid Structure': 'valid_structure',
        'False Positive': 'false_positive',
        'Incomplete Spine': 'incomplete_spine',
        'Falsely Extended': 'falsely_extended',
        'Merged Spine': 'merged_spine',
        'Split Spine': 'split_spine',
    }
    while row_index < subsection_marker_index:
        row = rows[row_index]
        line_number = row_index + 1
        if len(row) != len(spine_header):
            raise ValueError(f'Invalid spine row at line {line_number}')
        section_id = _parse_int(row[spine_field_indices['Section ID']], 'section ID', line_number)
        if section_id not in sections_by_id:
            raise ValueError(f'Unknown section ID at line {line_number}')
        global_spine_id = _parse_int(
            row[spine_field_indices[global_id_field]], 'global spine ID', line_number
        )
        spine_ids = spine_ids_by_section[section_id]
        if has_explicit_local_id:
            local_index = _parse_int(
                row[spine_field_indices['Local Spine ID']], 'local spine ID', line_number
            )
            if not 0 <= local_index < len(spine_ids) or spine_ids[local_index] != global_spine_id:
                raise ValueError(f'Local/global spine IDs do not match at line {line_number}')
        else:
            if global_spine_id not in spine_ids:
                raise ValueError(f'Unknown spine ID at line {line_number}')
            local_index = spine_ids.index(global_spine_id)
        key = (section_id, local_index)
        if key in seen_spines:
            raise ValueError(f'Duplicate spine row at line {line_number}')
        seen_spines.add(key)
        record = {'answers': {}, 'validity': _parse_csv_validity(
            row[spine_field_indices['Validity']], line_number
        )}
        for field_name, answer_key in answer_columns.items():
            if field_name in spine_field_indices:
                record['answers'][answer_key] = _parse_csv_answer(
                    row[spine_field_indices[field_name]], field_name, line_number
                )
        loaded['spines'][key] = record
        row_index += 1

    if subsection_marker_index == len(rows):
        return loaded
    row_index = subsection_marker_index + 1
    if row_index >= len(rows) or rows[row_index] != SUBSECTION_CSV_FIELDS:
        raise ValueError(f'Unexpected subsection table headers in {path}')
    row_index += 1
    subsection_field_indices = {field: SUBSECTION_CSV_FIELDS.index(field) for field in SUBSECTION_CSV_FIELDS}
    seen_subsections = set()
    while row_index < len(rows):
        row = rows[row_index]
        line_number = row_index + 1
        if len(row) != len(SUBSECTION_CSV_FIELDS):
            raise ValueError(f'Invalid subsection row at line {line_number}')
        section_id = _parse_int(row[subsection_field_indices['Section ID']], 'section ID', line_number)
        subsection_index = _parse_int(
            row[subsection_field_indices['Subsection Index']], 'subsection index', line_number
        )
        missing_count = _parse_int(
            row[subsection_field_indices['Missing Spine Count']], 'missing-spine count', line_number
        )
        if section_id not in sections_by_id:
            raise ValueError(f'Unknown section ID at line {line_number}')
        subsections = sections_by_id[section_id].get('subsections', [])
        if not 0 <= subsection_index < len(subsections) or missing_count < 0:
            raise ValueError(f'Invalid subsection summary at line {line_number}')
        key = (section_id, subsection_index)
        if key in seen_subsections:
            raise ValueError(f'Duplicate subsection row at line {line_number}')
        status = row[subsection_field_indices['Status']].strip().casefold()
        if status not in {'done', 'not set', ''}:
            raise ValueError(f'Invalid subsection status at line {line_number}')
        seen_subsections.add(key)
        loaded['subsections'][key] = {
            'missing_count': missing_count,
            'done': status == 'done',
        }
        row_index += 1
    return loaded


def restore_validation_state(state, path):
    """Apply saved validation records to a :class:`DesignState`."""
    loaded = read_validation_csv(path, state)
    sections_by_id = {_section_id(section): section for section in state.sections}
    for section_id, status in loaded['section_missing_status'].items():
        sections_by_id[section_id]['missing_status'] = status

    for section_id, section in sections_by_id.items():
        for local_index, spine in enumerate(section.get('spines', [])):
            record = loaded['spines'].get((section_id, local_index))
            if record is None:
                continue
            spine['answers'].update(record['answers'])
            # Answer columns are always binary, but Validity is only assigned
            # by the explicit Spine done action. Do not infer completion from
            # a row whose persisted validity is Not Set.
            spine['validity'] = record['validity']
            spine['checked'] = spine['validity'] is not None
        for subsection in section.get('subsections', []):
            record = loaded['subsections'].get((section_id, int(subsection['index'])))
            if record is None:
                continue
            subsection['missing_count'] = record['missing_count']
            subsection['done'] = record['done']
        state.section_is_checked(section)
    return loaded

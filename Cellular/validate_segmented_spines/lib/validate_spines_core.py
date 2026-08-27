import asyncio
import base64
import colorsys
import csv
import html
import importlib
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import ipywidgets as widgets
import k3d
import numpy as np
from IPython.display import HTML, display

from morph_spines_visualizer.core import data_loading, geometry, k3d_core
from morph_spines_visualizer.core import spines as spines_lib

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - exercised only in minimal installations
    cKDTree = None


MISSING_SPINES_MISSING = 'Missing'
MISSING_SPINES_NO_MISSING = 'No Missing'
MISSING_SPINES_NOT_SET = 'Not Set'


def normalize_missing_spines_status(status):
    """Return the canonical section missing-spines status label."""
    normalized = '' if status is None else str(status).strip().casefold()
    status_labels = {
        '': MISSING_SPINES_NOT_SET,
        'not set': MISSING_SPINES_NOT_SET,
        'missing': MISSING_SPINES_MISSING,
        'no missing': MISSING_SPINES_NO_MISSING,
    }
    try:
        return status_labels[normalized]
    except KeyError as exc:
        raise ValueError(
            f'Invalid missing-segmented-spines status: {status!r}'
        ) from exc


def _migrate_legacy_validation_csv(validation_csv_path, morphology):
    """Convert the legacy long-form CSV to the version-2 two-table format."""
    validation_csv_path = Path(validation_csv_path)
    if not validation_csv_path.is_file():
        return
    with validation_csv_path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != [
            'record_type', 'schema_version', 'source_id',
            'section_id', 'spine_index', 'status',
        ]:
            return
        rows = list(reader)

    section_ids = [
        section.id for section in morphology.morphology.sections
        if len(spines_lib.get_spine_ids_by_section_id(morphology, section.id)) > 0
    ]
    spine_ids_by_section = {
        section_id: list(
            spines_lib.get_spine_ids_by_section_id(morphology, section_id)
        )
        for section_id in section_ids
    }
    section_counts = {
        section_id: len(spine_ids)
        for section_id, spine_ids in spine_ids_by_section.items()
    }
    validity = {}
    issue_maps = {
        'false_positive': {},
        'incomplete_spine': {},
        'false_positive_quality': {},
        'merged_spine': {},
        'split_spine': {},
    }
    section_missing = {}
    for row in rows:
        section_id = int(row['section_id'])
        record_type = row['record_type']
        status = row['status'].strip().lower()
        if record_type == 'section':
            section_missing[section_id] = normalize_missing_spines_status(status)
            continue
        spine_index = int(row['spine_index'])
        key = (section_id, spine_index)
        if record_type == 'spine':
            validity[key] = status
        elif record_type in issue_maps:
            issue_maps[record_type][key] = status

    section_fields = [
        'Section', 'Number Spines', 'Validated (Yes or No)',
        'Remaining Spines to Validate', 'False Positives',
        'Incomplete Spines', 'Falsely Extended',
        'Merged Spines', 'Split Spines', 'Missing Segmented Spines',
    ]
    spine_fields = [
        'Section ID', 'Local Spine ID', 'Global Spine ID', 'Validity',
        'Correct Type', 'False Positive', 'Incomplete Spine', 'Falsely Extended',
        'Merged Spine', 'Split Spine',
    ]
    issue_columns = (
        ('False Positive', 'false_positive'),
        ('Incomplete Spine', 'incomplete_spine'),
        ('Falsely Extended', 'false_positive_quality'),
        ('Merged Spine', 'merged_spine'),
        ('Split Spine', 'split_spine'),
    )
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', newline='', encoding='utf-8', delete=False,
            dir=validation_csv_path.parent,
            prefix=f'.{validation_csv_path.name}.', suffix='.tmp'
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.writer(handle)
            writer.writerow(['validation_format', '2'])
            writer.writerow(['table', 'section'])
            writer.writerow(section_fields)
            for section_id in section_ids:
                count = section_counts[section_id]
                validated_count = sum(
                    saved_section_id == section_id
                    for saved_section_id, _ in validity
                )
                finding_counts = [
                    sum(
                        status == 'yes'
                        for (saved_section_id, _), status in results.items()
                        if saved_section_id == section_id
                    )
                    for results in issue_maps.values()
                ]
                writer.writerow([
                    section_id,
                    count,
                    'Yes' if validated_count == count else 'No',
                    max(count - validated_count, 0),
                    *finding_counts,
                    normalize_missing_spines_status(section_missing.get(section_id)),
                ])
            writer.writerow(['table', 'spine'])
            writer.writerow(spine_fields)
            for section_id in section_ids:
                for spine_index, spine_id in enumerate(spine_ids_by_section[section_id]):
                    key = (section_id, spine_index)
                    writer.writerow([
                        section_id,
                        spine_index,
                        int(spine_id),
                        validity.get(key, 'Not Set').title(),
                        'Not Set',
                        *[
                            issue_maps[record_type].get(key, 'Not Set').title()
                            for _column, record_type in issue_columns
                        ],
                    ])
        os.replace(temporary_path, validation_csv_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


class _MeshSpatialIndex:
    """Query full mesh vertices inside axis-aligned boxes."""

    def __init__(self, vertices):
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.tree = cKDTree(self.vertices) if cKDTree is not None and len(self.vertices) else None
        self.cell_size = None
        self.bins = None

        if self.tree is None and len(self.vertices):
            # Keep a dependency-free fallback for installations without SciPy. The grid
            # only narrows the candidate set; the final box mask remains exact.
            extent = np.ptp(self.vertices, axis=0)
            self.cell_size = max(float(np.max(extent)) / 128.0, np.finfo(np.float32).eps)
            cell_coordinates = np.floor(self.vertices / self.cell_size).astype(np.int64)
            bins = {}
            for vertex_index, cell in enumerate(map(tuple, cell_coordinates)):
                bins.setdefault(cell, []).append(vertex_index)
            self.bins = bins

    def query_indices(self, box_min, box_max):
        """Return indices of indexed points inside the inclusive axis-aligned box."""
        if not len(self.vertices):
            return np.empty(0, dtype=np.intp)

        box_min = np.asarray(box_min, dtype=np.float32)
        box_max = np.asarray(box_max, dtype=np.float32)
        center = (box_min + box_max) * 0.5
        half_extent = (box_max - box_min) * 0.5

        if self.tree is not None:
            # cKDTree supports a Chebyshev-radius query, which returns a cube.
            # Filtering below makes rectangular section bounds exact.
            radius = max(float(np.max(half_extent)), np.finfo(np.float32).eps)
            candidate_indices = self.tree.query_ball_point(center, radius, p=np.inf)
        else:
            cell_min = np.floor(box_min / self.cell_size).astype(np.int64)
            cell_max = np.floor(box_max / self.cell_size).astype(np.int64)
            candidate_indices = []
            for x in range(int(cell_min[0]), int(cell_max[0]) + 1):
                for y in range(int(cell_min[1]), int(cell_max[1]) + 1):
                    for z in range(int(cell_min[2]), int(cell_max[2]) + 1):
                        candidate_indices.extend(self.bins.get((x, y, z), ()))

        if not candidate_indices:
            return np.empty(0, dtype=np.intp)
        candidate_indices = np.asarray(candidate_indices, dtype=np.intp)
        candidates = self.vertices[candidate_indices]
        inside = np.all((candidates >= box_min) & (candidates <= box_max), axis=1)
        return candidate_indices[inside]

    def query_box(self, box_min, box_max):
        """Return all indexed vertices inside the inclusive axis-aligned box."""
        return self.vertices[self.query_indices(box_min, box_max)]


class _MeshSurfaceSampler:
    """Draw extra, randomly distributed points from a mesh surface inside a box.

    The exact full-resolution vertices selected by `_MeshSpatialIndex` follow
    the mesh's original tessellation, which can look sparse or unevenly
    patterned for validation. This sampler adds area-weighted, barycentric
    random points on the triangles near a query box, producing a denser and
    visually uniform point cloud without discarding the real vertex data.
    """

    def __init__(self, vertices, faces):
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int64)
        self.face_v0 = vertices[faces[:, 0]]
        self.face_v1 = vertices[faces[:, 1]]
        self.face_v2 = vertices[faces[:, 2]]
        cross = np.cross(self.face_v1 - self.face_v0, self.face_v2 - self.face_v0)
        self.face_areas = 0.5 * np.linalg.norm(cross, axis=1)
        centroids = (self.face_v0 + self.face_v1 + self.face_v2) / 3.0
        self.centroid_index = _MeshSpatialIndex(centroids)

    def sample_within_box(self, box_min, box_max, target_count, rng, margin=0.0):
        """Draw up to `target_count` random surface points inside the box."""
        if target_count <= 0 or not len(self.face_areas):
            return np.empty((0, 3), dtype=np.float32)

        box_min = np.asarray(box_min, dtype=np.float32)
        box_max = np.asarray(box_max, dtype=np.float32)
        candidate_indices = self.centroid_index.query_indices(box_min - margin, box_max + margin)
        if not len(candidate_indices):
            return np.empty((0, 3), dtype=np.float32)

        areas = self.face_areas[candidate_indices]
        total_area = float(areas.sum())
        if total_area <= 0.0:
            return np.empty((0, 3), dtype=np.float32)

        weights = areas / total_area
        chosen = rng.choice(candidate_indices, size=target_count, p=weights)

        # Uniform-in-area triangle sampling via a square-root barycentric transform.
        r1 = rng.random(target_count).astype(np.float32)
        r2 = rng.random(target_count).astype(np.float32)
        sqrt_r1 = np.sqrt(r1)
        v0, v1, v2 = self.face_v0[chosen], self.face_v1[chosen], self.face_v2[chosen]
        points = (
            (1.0 - sqrt_r1)[:, None] * v0
            + (sqrt_r1 * (1.0 - r2))[:, None] * v1
            + (sqrt_r1 * r2)[:, None] * v2
        )
        inside = np.all((points >= box_min) & (points <= box_max), axis=1)
        return points[inside].astype(np.float32)


def validate_spines(morphology_path, mesh_path):

    # Keep all proofreading artifacts beside the input mesh, rather than beside
    # the morphology file or the notebook's working directory.
    mesh_path = Path(mesh_path)
    mesh_path_for_loader = str(mesh_path)
    proofreading_dir = mesh_path.with_name(f'{mesh_path.stem}_proofreading')
    proofreading_dir.mkdir(parents=True, exist_ok=True)

    # Load the spiny morphology and capture the assessment identity once so all
    # displayed and registered assessment metadata uses the same user value.
    morphology = data_loading.load_spiny_morphology(morphology_path)
    validation_csv_path = proofreading_dir / (
        f'{Path(morphology_path).stem}_validation.csv'
    )
    _migrate_legacy_validation_csv(validation_csv_path, morphology)
    morphology_section_count = len(list(morphology.morphology.sections))
    total_spines = int(morphology.spines.spine_count)
    user_email = os.environ.get('OBI_USERNAME', 'unknown-user')

    # Extra surface sampling remains available as an opt-in mode. Keep the
    # default at zero to preserve the faster exact-vertex point-cloud density.
    SECTION_DENSE_SAMPLE_MULTIPLIER = 0

    # Retain the full-resolution vertices for section-local queries and use a
    # deterministic 15% sample for the mesh context shown at all times.
    if SECTION_DENSE_SAMPLE_MULTIPLIER > 0:
        full_mesh_vertices, full_mesh_faces = data_loading.load_mesh_vertices_and_faces_pylmesh(
            mesh_path_for_loader, scale_factor=1e-3
        )
    else:
        full_mesh_vertices = data_loading.load_mesh_vertices_pylmesh(
            mesh_path_for_loader, scale_factor=1e-3
        )
        full_mesh_faces = None

    GLOBAL_MESH_SAMPLE_FRACTION = 0.15
    global_sample_count = max(
        1, int(round(len(full_mesh_vertices) * GLOBAL_MESH_SAMPLE_FRACTION))
    ) if len(full_mesh_vertices) else 0
    global_sample_indices = np.linspace(
        0, len(full_mesh_vertices) - 1, global_sample_count, dtype=np.intp
    ) if global_sample_count else np.empty(0, dtype=np.intp)
    mesh_vertices = full_mesh_vertices[global_sample_indices]
    mesh_spatial_index = _MeshSpatialIndex(full_mesh_vertices)
    mesh_surface_sampler = (
        _MeshSurfaceSampler(full_mesh_vertices, full_mesh_faces)
        if SECTION_DENSE_SAMPLE_MULTIPLIER > 0 else None
    )
    # Get sections that have spines
    section_ids_with_counts = spines_lib.get_section_ids_with_spine_counts_for_sections_with_spines(
        morphology=morphology
    )
    spiny_section_count = len(section_ids_with_counts)
    spines_in_spiny_sections = sum(
        spine_count for _, spine_count in section_ids_with_counts
    )
    sections_with_spines_percent = (
        spiny_section_count / morphology_section_count
        if morphology_section_count else 0.0
    )
    average_spines_per_spiny_section = (
        spines_in_spiny_sections / spiny_section_count
        if spiny_section_count else 0.0
    )

    print()
    print('Validation Dataset')
    print('------------------')
    print(f'Morphology: {Path(morphology_path).name}')
    print(f'Mesh: {Path(mesh_path).name}')
    print(
        f'Sections: {morphology_section_count:,} total | '
        f'{spiny_section_count:,} with spines '
        f'({sections_with_spines_percent:.1%})'
    )
    print(f'Total spines: {total_spines:,}')
    print(
        f'Spines represented in spiny sections: '
        f'{spines_in_spiny_sections:,}'
    )
    print(
        f'Average spines per spiny section: '
        f'{average_spines_per_spiny_section:.1f}'
    )
    print()
    print('Mesh Data')
    print('---------')
    print(
        f'Indexed vertices: {full_mesh_vertices.shape[0]:,}'
    )
    print(
        f'Displayed vertices: {mesh_vertices.shape[0]:,} '
        f'({GLOBAL_MESH_SAMPLE_FRACTION:.0%} sample)'
    )
    print()

    # ============================================================
    # Create K3D plot
    # ============================================================

    plot = k3d.plot(
        grid_visible=False,
        menu_visibility=False,
        camera_mode='trackball',
        camera_up_axis='y',
        background_color=0xffffff,
        height=800,
    )

    # Add the sampled mesh context and a separate, initially hidden, local cloud.
    global_point_cloud = k3d.points(
        mesh_vertices,
        point_size=0.06,
        color=0x87CEEB,
        opacity=0.45,
        shader='flat'
    )
    section_point_cloud = k3d.points(
        np.empty((0, 3), dtype=np.float32),
        point_size=0.025,
        color=0x2F80ED,
        opacity=0.8,
        shader='flat'
    )
    section_point_cloud.visible = False
    plot += global_point_cloud
    plot += section_point_cloud

    # Add finite gray morphology centerlines. NaN-separated K3D lines can blank
    # the WebGL scene in K3D 2.18.1, so keep each section finite and independent.
    sections_points = geometry.get_sections_points(morphology)

    SECTION_SUBSECTION_LENGTH_UM = 10.0

    def build_section_subsections(points):
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
            }]

        boundaries = list(np.arange(
            0.0,
            total_length,
            SECTION_SUBSECTION_LENGTH_UM,
        ))
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
            })
        return subsections

    section_subsections = {
        sec_id: build_section_subsections(points)
        for sec_id, points in sections_points.items()
    }
    section_lines = []
    for sec_id, points in sections_points.items():
        section_line = k3d.line(
            np.asarray(points, dtype=np.float32),
            width=1.5,
            color=0x888888,
            shader='simple',
            name=f'Morphology section {sec_id}',
        )
        plot += section_line
        section_lines.append(section_line)

    # Reuse one focused overlay for every selected section.
    section_centerline_overlay = k3d.line(
        np.zeros((2, 3), dtype=np.float32),
        width=9.0,
        color=0xFF0000,
        shader='simple',
        name='Selected morphology section',
    )
    section_centerline_overlay.visible = False
    plot += section_centerline_overlay

    # Reuse an orange mesh-style line for the currently reviewed 10 µm subsection.
    SUBSECTION_CENTERLINE_WIDTH = 0.025
    SUBSECTION_CENTERLINE_RADIAL_SEGMENTS = 4
    subsection_centerline_overlay = k3d.line(
        np.zeros((2, 3), dtype=np.float32),
        width=SUBSECTION_CENTERLINE_WIDTH,
        color=0xFF8C00,
        shader='mesh',
        radial_segments=SUBSECTION_CENTERLINE_RADIAL_SEGMENTS,
        name='Current morphology subsection',
    )
    subsection_centerline_overlay.visible = False
    plot += subsection_centerline_overlay

    # Compute morphology bounds for initial camera
    _, _, morph_center, morph_extent, morph_radius = geometry.compute_morphology_bounds(morphology)


    # ============================================================
    # State variables
    # ============================================================

    # List of (section_id, spine_count) for sections with spines
    spiny_sections = section_ids_with_counts
    current_section_idx = [0]       # Index into spiny_sections list
    current_subsection_idx = [0]    # Index into the current section's subsections
    updating_subsection_dropdown = [False]
    subsection_missing_counts = {}  # (section_id, subsection_index) -> count
    subsection_done_results = set()  # (section_id, subsection_index) values marked Done
    current_spine_idx = [0]         # Index into current section's spine list
    spine_selected = [False]        # True only after an explicit spine selection
    updating_section_dropdown = [False]
    updating_spine_dropdown = [False]
    current_spine_meshes_k3d = []   # K3D objects for current section's spines
    current_spine_data = []         # List of spine trimesh objects for current section
    current_spine_global_ids = []   # Global H5 spine IDs aligned with current_spine_data
    current_spine_colors = []       # Palette colors aligned with displayed spine meshes
    spine_structure_meshes_k3d = [] # Head/neck K3D objects for the selected spine
    spine_structure_visible = [False]
    spine_structure_base_mesh = [None]
    previous_colored_spine_idx = [None]
    spine_colors_initialized = [False]
    section_centerline_k3d = [section_centerline_overlay]
    loading_task = [None]
    load_generation = [0]

    # A golden-angle hue sequence stays deterministic and well distributed even
    # when a section contains hundreds of spines. Blue/cyan hues are reserved
    # for the point-cloud context, so spine colors use the remaining hue ranges.
    section_spine_colors = {}
    PALETTE_HUE_STEP = 0.618033988749895
    PALETTE_BLUE_HUE_START = 0.50
    PALETTE_BLUE_HUE_END = 0.68
    PALETTE_SATURATION = 0.82
    PALETTE_VALUE = 0.90
    UNSELECTED_SPINE_COLOR = 0xA0A0A0
    SPINE_LOAD_DELAY = 0.05
    SECTION_CENTERLINE_WIDTH = 9.0
    SECTION_CENTERLINE_COLOR = 0xFF0000
    SECTION_CENTERLINE_HOT_COLOR = 0xFF1744
    SUBSECTION_CENTERLINE_COLOR = 0xFF8C00
    SPINE_NECK_STRUCTURE_COLOR = 0x64C864
    SPINE_HEAD_STRUCTURE_COLOR = 0xFF6464

    # Bounding box lines for selected spine
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    bbox_lines = []
    dummy = np.zeros((2, 3), dtype=np.float32)
    for _ in edges:
        line = k3d.line(dummy, color=0xFF0000, shader='simple', width=0.3)
        line.visible = False
        line.opacity = 0.5
        bbox_lines.append(line)
        plot += line

    # Validation results: (section_id, spine_index_in_section) -> 'valid' | 'invalid'
    validation_results = {}
    VALIDITY_NOT_SET = 'Not Set'
    correct_type_results = {}
    valid_structure_results = {}
    false_positive_results = {}
    incomplete_spine_results = {}
    false_positive_quality_results = {}
    merged_spine_results = {}
    split_spine_results = {}
    # Section-level segmented-spine review: section_id -> canonical status label
    section_missing_results = {}
    # Validation state persistence
    VALIDATION_SCHEMA_VERSION = '5'
    SUPPORTED_VALIDATION_SCHEMA_VERSIONS = {'2', '3', '4', '5'}
    LEGACY_VALIDATION_CSV_FIELDS = [
        'record_type', 'schema_version', 'source_id',
        'section_id', 'spine_index', 'status',
    ]
    SECTION_CSV_FIELDS = [
        'Section', 'Number Spines', 'Validated (Yes or No)',
        'Remaining Spines to Validate', 'False Positives',
        'Incomplete Spines', 'Falsely Extended Spines',
        'Merged Spines', 'Split Spines', 'Valid Structure',
        'Missing Segmented Spines',
    ]
    PREVIOUS_SECTION_CSV_FIELDS = [
        'Section', 'Number Spines', 'Validated (Yes or No)',
        'Remaining Spines to Validate', 'False Positives',
        'Incomplete Spines', 'Falsely Extended Spines',
        'Merged Spines', 'Split Spines', 'Missing Segmented Spines',
    ]
    SPINE_CSV_FIELDS = [
        'Section ID', 'Local Spine ID', 'Global Spine ID', 'Validity',
        'Correct Type', 'Valid Structure', 'False Positive', 'Incomplete Spine', 'Falsely Extended',
        'Merged Spine', 'Split Spine',
    ]
    PREVIOUS_SPINE_CSV_FIELDS = [
        'Section ID', 'Local Spine ID', 'Global Spine ID', 'Validity',
        'False Positive', 'Incomplete Spine', 'Falsely Extended',
        'Merged Spine', 'Split Spine',
    ]
    LEGACY_SPINE_CSV_FIELDS = [
        'Section ID', 'Spine ID', 'Validity', 'Correct Type', 'False Positive',
        'Incomplete Spine', 'Falsely Extended', 'Merged Spine', 'Split Spine',
    ]
    PREVIOUS_LEGACY_SPINE_CSV_FIELDS = [
        'Section ID', 'Spine ID', 'Validity', 'False Positive',
        'Incomplete Spine', 'Falsely Extended', 'Merged Spine',
        'Split Spine',
    ]
    SUBSECTION_CSV_FIELDS = [
        'Section ID', 'Subsection Index', 'Missing Spine Count', 'Status',
    ]
    validation_source_id = Path(morphology_path).name
    validation_csv_path = proofreading_dir / (
        f'{Path(morphology_path).stem}_validation.csv'
    )
    validation_save_task = [None]
    pending_validation_snapshot = [None]
    screenshot_request_path = [None]
    screenshot_request_token = [0]
    screenshot_save_task = [None]
    registration_task = [None]
    report_task = [None]


    def snapshot_validation_state():
        """Create an immutable three-table snapshot of all validation state."""
        section_rows = []
        for sec_id, spine_count in spiny_sections:
            validated_count = sum(
                1 for (saved_sec_id, _), _status in validation_results.items()
                if saved_sec_id == sec_id
            )
            finding_maps = (
                false_positive_results,
                incomplete_spine_results,
                false_positive_quality_results,
                merged_spine_results,
                split_spine_results,
                valid_structure_results,
            )
            finding_counts = [
                sum(
                    status == 'yes'
                    for (saved_sec_id, _), status in results.items()
                    if saved_sec_id == sec_id
                )
                for results in finding_maps
            ]
            section_rows.append({
                'Section': sec_id,
                'Number Spines': spine_count,
                'Validated (Yes or No)': 'Yes' if validated_count == spine_count else 'No',
                'Remaining Spines to Validate': max(spine_count - validated_count, 0),
                'False Positives': finding_counts[0],
                'Incomplete Spines': finding_counts[1],
                'Falsely Extended Spines': finding_counts[2],
                'Merged Spines': finding_counts[3],
                'Split Spines': finding_counts[4],
                'Valid Structure': finding_counts[5],
                'Missing Segmented Spines': normalize_missing_spines_status(
                    section_missing_results.get(sec_id)
                ),
            })

        spine_rows = []
        for sec_id, _spine_count in spiny_sections:
            spine_ids = list(morphology.spines.spine_indices_for_section(sec_id + 1))
            for spine_idx, spine_id in enumerate(spine_ids):
                key = (sec_id, spine_idx)
                spine_rows.append({
                    'Section ID': sec_id,
                    'Local Spine ID': spine_idx,
                    'Global Spine ID': int(spine_id),
                    'Validity': validation_results.get(key, 'Not Set').title(),
                    'Correct Type': correct_type_results.get(key, 'Not Set').title(),
                    'Valid Structure': valid_structure_results.get(key, 'Not Set').title(),
                    'False Positive': false_positive_results.get(key, 'Not Set').title(),
                    'Incomplete Spine': incomplete_spine_results.get(key, 'Not Set').title(),
                    'Falsely Extended': false_positive_quality_results.get(key, 'Not Set').title(),
                    'Merged Spine': merged_spine_results.get(key, 'Not Set').title(),
                    'Split Spine': split_spine_results.get(key, 'Not Set').title(),
                })
        subsection_rows = []
        for sec_id, _spine_count in spiny_sections:
            for subsection in section_subsections.get(sec_id, []):
                subsection_idx = subsection['index']
                key = (sec_id, subsection_idx)
                subsection_rows.append({
                    'Section ID': sec_id,
                    'Subsection Index': subsection_idx,
                    'Missing Spine Count': subsection_missing_counts.get(key, 0),
                    'Status': 'Done' if key in subsection_done_results else 'Not Set',
                })
        return {
            'section_rows': section_rows,
            'spine_rows': spine_rows,
            'subsection_rows': subsection_rows,
        }


    def write_validation_csv(path, snapshot):
        """Atomically write section, spine, and subsection validation tables."""
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
                    [
                        [
                            normalize_missing_spines_status(row[field])
                            if field == 'Missing Segmented Spines'
                            else row[field]
                            for field in SECTION_CSV_FIELDS
                        ]
                        for row in snapshot['section_rows']
                    ]
                )
                writer.writerow(['table', 'spine'])
                writer.writerow(SPINE_CSV_FIELDS)
                writer.writerows(
                    [
                        [row[field] for field in SPINE_CSV_FIELDS]
                        for row in snapshot['spine_rows']
                    ]
                )
                writer.writerow(['table', 'subsection'])
                writer.writerow(SUBSECTION_CSV_FIELDS)
                writer.writerows(
                    [
                        [row[field] for field in SUBSECTION_CSV_FIELDS]
                        for row in snapshot['subsection_rows']
                    ]
                )
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()


    def read_validation_csv(path):
        """Read legacy or versioned validation CSV data."""
        path = Path(path)
        if not path.exists():
            return {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, set()

        section_counts = dict(spiny_sections)
        targets = {
            'spine': {},
            'correct_type': {},
            'valid_structure': {},
            'false_positive': {},
            'incomplete_spine': {},
            'false_positive_quality': {},
            'merged_spine': {},
            'split_spine': {},
        }
        loaded_sections = {}
        loaded_subsection_counts = {}
        loaded_subsection_done = set()

        def parse_legacy():
            with path.open('r', newline='', encoding='utf-8') as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != LEGACY_VALIDATION_CSV_FIELDS:
                    raise ValueError(f'Unexpected validation CSV headers in {path}')
                for line_number, row in enumerate(reader, start=2):
                    if row.get('schema_version') != '1':
                        raise ValueError(f'Unsupported legacy validation CSV version at line {line_number}')
                    if row.get('source_id') != validation_source_id:
                        raise ValueError(f'Validation CSV source does not match {validation_source_id}')
                    try:
                        sec_id = int(row['section_id'])
                    except (TypeError, ValueError, KeyError) as exc:
                        raise ValueError(f'Invalid section ID at line {line_number}') from exc
                    if sec_id not in section_counts:
                        raise ValueError(f'Unknown section ID {sec_id} at line {line_number}')
                    record_type = row.get('record_type')
                    status = row.get('status')
                    if record_type == 'section':
                        section_status = normalize_missing_spines_status(status)
                        if row.get('spine_index', ''):
                            raise ValueError(f'Invalid section record at line {line_number}')
                        if sec_id in loaded_sections:
                            raise ValueError(f'Duplicate section record at line {line_number}')
                        loaded_sections[sec_id] = section_status
                    elif record_type in targets:
                        try:
                            spine_idx = int(row['spine_index'])
                        except (TypeError, ValueError, KeyError) as exc:
                            raise ValueError(f'Invalid spine index at line {line_number}') from exc
                        if not 0 <= spine_idx < section_counts[sec_id]:
                            raise ValueError(f'Out-of-range spine index at line {line_number}')
                        allowed_statuses = {'valid', 'invalid'} if record_type == 'spine' else {'yes', 'no'}
                        if status not in allowed_statuses:
                            raise ValueError(f'Invalid {record_type} status at line {line_number}')
                        key = (sec_id, spine_idx)
                        if key in targets[record_type]:
                            raise ValueError(f'Duplicate {record_type} record at line {line_number}')
                        targets[record_type][key] = status
                    else:
                        raise ValueError(f'Unknown record type at line {line_number}')

        with path.open('r', newline='', encoding='utf-8') as handle:
            first_row = next(csv.reader(handle), [])
        is_versioned = (
            len(first_row) == 2
            and first_row[0] == 'validation_format'
            and first_row[1] in SUPPORTED_VALIDATION_SCHEMA_VERSIONS
        )
        if not is_versioned:
            parse_legacy()
        else:
            validation_format_version = first_row[1]
            with path.open('r', newline='', encoding='utf-8') as handle:
                rows = [row for row in csv.reader(handle) if row]
            if rows[:2] != [
                ['validation_format', validation_format_version],
                ['table', 'section'],
            ]:
                raise ValueError(f'Invalid validation table markers in {path}')
            if len(rows) < 4 or rows[2] not in (
                SECTION_CSV_FIELDS,
                PREVIOUS_SECTION_CSV_FIELDS,
            ):
                raise ValueError(f'Unexpected section table headers in {path}')
            section_header = rows[2]
            section_field_indices = {
                field_name: section_header.index(field_name)
                for field_name in section_header
            }
            row_index = 3
            while row_index < len(rows) and rows[row_index] != ['table', 'spine']:
                row = rows[row_index]
                if len(row) != len(section_header):
                    raise ValueError(f'Invalid section row at line {row_index + 1}')
                try:
                    sec_id = int(row[0])
                    number_spines = int(row[1])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f'Invalid section summary at line {row_index + 1}') from exc
                if sec_id not in section_counts or number_spines != section_counts[sec_id]:
                    raise ValueError(f'Section summary does not match morphology at line {row_index + 1}')
                if sec_id in loaded_sections:
                    raise ValueError(f'Duplicate section summary at line {row_index + 1}')
                missing_status = normalize_missing_spines_status(
                    row[section_field_indices['Missing Segmented Spines']]
                )
                loaded_sections[sec_id] = missing_status
                row_index += 1
            if row_index >= len(rows) or rows[row_index] != ['table', 'spine']:
                raise ValueError(f'Missing spine table marker in {path}')
            row_index += 1
            spine_header = rows[row_index] if row_index < len(rows) else None
            if (
                spine_header != SPINE_CSV_FIELDS
                and spine_header != PREVIOUS_SPINE_CSV_FIELDS
                and spine_header != LEGACY_SPINE_CSV_FIELDS
                and spine_header != PREVIOUS_LEGACY_SPINE_CSV_FIELDS
            ):
                raise ValueError(f'Unexpected spine table headers in {path}')
            row_index += 1
            spine_ids_by_section = {
                sec_id: list(morphology.spines.spine_indices_for_section(sec_id + 1))
                for sec_id in section_counts
            }
            spine_field_indices = {
                field_name: spine_header.index(field_name)
                for field_name in spine_header
            }
            has_explicit_local_id = spine_header in (
                SPINE_CSV_FIELDS,
                PREVIOUS_SPINE_CSV_FIELDS,
            )
            has_correct_type = 'Correct Type' in spine_header
            seen_spines = set()
            field_targets = {
                'False Positive': 'false_positive',
                'Incomplete Spine': 'incomplete_spine',
                'Falsely Extended': 'false_positive_quality',
                'Merged Spine': 'merged_spine',
                'Split Spine': 'split_spine',
            }
            if has_correct_type:
                field_targets = {
                    'Correct Type': 'correct_type',
                    **field_targets,
                }
            if 'Valid Structure' in spine_header:
                field_targets = {
                    'Valid Structure': 'valid_structure',
                    **field_targets,
                }
            spine_table_end = len(rows)
            if validation_format_version in {'3', '4', '5'}:
                try:
                    subsection_marker_index = rows.index(
                        ['table', 'subsection'],
                        row_index,
                    )
                except ValueError as exc:
                    raise ValueError(f'Missing subsection table marker in {path}') from exc
                spine_table_end = subsection_marker_index
            while row_index < spine_table_end:
                row = rows[row_index]
                if len(row) != len(spine_header):
                    raise ValueError(f'Invalid spine row at line {row_index + 1}')
                try:
                    sec_id = int(row[spine_field_indices['Section ID']])
                    global_spine_id = int(
                        row[spine_field_indices[
                            'Global Spine ID' if has_explicit_local_id else 'Spine ID'
                        ]]
                    )
                    if has_explicit_local_id:
                        local_spine_id = int(row[spine_field_indices['Local Spine ID']])
                except (TypeError, ValueError, KeyError) as exc:
                    raise ValueError(f'Invalid spine identity at line {row_index + 1}') from exc
                if sec_id not in spine_ids_by_section:
                    raise ValueError(f'Unknown section ID at line {row_index + 1}')
                spine_ids = spine_ids_by_section[sec_id]
                if has_explicit_local_id:
                    if not 0 <= local_spine_id < len(spine_ids):
                        raise ValueError(f'Invalid local spine ID at line {row_index + 1}')
                    if spine_ids[local_spine_id] != global_spine_id:
                        raise ValueError(f'Local/global spine IDs do not match at line {row_index + 1}')
                else:
                    if global_spine_id not in spine_ids:
                        raise ValueError(f'Unknown spine ID at line {row_index + 1}')
                    local_spine_id = spine_ids.index(global_spine_id)
                key = (sec_id, local_spine_id)
                if key in seen_spines:
                    raise ValueError(f'Duplicate spine row at line {row_index + 1}')
                seen_spines.add(key)
                validity = row[spine_field_indices['Validity']].strip().lower()
                if validity not in {'valid', 'invalid', 'not set', ''}:
                    raise ValueError(f'Invalid validity at line {row_index + 1}')
                if validity in {'valid', 'invalid'}:
                    targets['spine'][key] = validity
                for field_name, record_type in field_targets.items():
                    status = row[spine_field_indices[field_name]].strip().lower()
                    if status not in {'yes', 'no', 'not set', ''}:
                        raise ValueError(f'Invalid {field_name} status at line {row_index + 1}')
                    if status in {'yes', 'no'}:
                        targets[record_type][key] = status
                row_index += 1

            if validation_format_version in {'3', '4', '5'}:
                row_index = subsection_marker_index + 1
                subsection_header = rows[row_index] if row_index < len(rows) else None
                if subsection_header != SUBSECTION_CSV_FIELDS:
                    raise ValueError(f'Unexpected subsection table headers in {path}')
                row_index += 1
                subsection_counts_by_section = {
                    sec_id: len(section_subsections.get(sec_id, []))
                    for sec_id in section_counts
                }
                seen_subsections = set()
                while row_index < len(rows):
                    row = rows[row_index]
                    if len(row) != len(subsection_header):
                        raise ValueError(f'Invalid subsection row at line {row_index + 1}')
                    try:
                        sec_id = int(row[0])
                        subsection_idx = int(row[1])
                        missing_count = int(row[2])
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f'Invalid subsection summary at line {row_index + 1}') from exc
                    if sec_id not in subsection_counts_by_section:
                        raise ValueError(f'Unknown section ID at line {row_index + 1}')
                    if not 0 <= subsection_idx < subsection_counts_by_section[sec_id]:
                        raise ValueError(f'Invalid subsection index at line {row_index + 1}')
                    if missing_count < 0:
                        raise ValueError(f'Invalid missing-spine count at line {row_index + 1}')
                    key = (sec_id, subsection_idx)
                    if key in seen_subsections:
                        raise ValueError(f'Duplicate subsection row at line {row_index + 1}')
                    seen_subsections.add(key)
                    status = row[3].strip().casefold()
                    if status not in {'done', 'not set', ''}:
                        raise ValueError(f'Invalid subsection status at line {row_index + 1}')
                    loaded_subsection_counts[key] = missing_count
                    if status == 'done':
                        loaded_subsection_done.add(key)
                    row_index += 1

        return (
            targets['spine'],
            targets['correct_type'],
            targets['valid_structure'],
            targets['false_positive'],
            targets['incomplete_spine'],
            targets['false_positive_quality'],
            targets['merged_spine'],
            targets['split_spine'],
            loaded_sections,
            loaded_subsection_counts,
            loaded_subsection_done,
        )


    def stats_figure_paths():
        """Return deterministic paths for the registration analysis figures."""
        stem = validation_csv_path.stem
        return [
            validation_csv_path.with_name(f'{stem}_section_validity.png'),
            validation_csv_path.with_name(f'{stem}_issue_counts.png'),
        ]


    def build_stats_figures_from_csv():
        """Generate the section-validity figure from the persisted validation CSV."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.font_manager import FontProperties
        except ImportError as exc:
            raise RuntimeError(
                'Generating assessment statistics requires matplotlib.'
            ) from exc

        font_path = Path('lib') / 'fonts' / 'font_regular.otf'
        if not font_path.is_file():
            raise RuntimeError(f'Bundled assessment font not found: {font_path}')
        axis_label_font = FontProperties(fname=str(font_path))

        if not validation_csv_path.exists():
            write_validation_csv(validation_csv_path, snapshot_validation_state())

        (
            loaded_spines,
            loaded_correct_type,
            loaded_valid_structure,
            loaded_false_positives,
            loaded_incomplete_spines,
            loaded_false_positive_quality,
            loaded_merged_spines,
            loaded_split_spines,
            loaded_sections,
            _,
            _,
        ) = read_validation_csv(validation_csv_path)

        section_ids = [section_id for section_id, _ in spiny_sections]
        section_counts = dict(spiny_sections)
        valid_by_section = {
            section_id: sum(
                status == 'valid'
                for (saved_section_id, _), status in loaded_spines.items()
                if saved_section_id == section_id
            )
            for section_id in section_ids
        }
        invalid_by_section = {
            section_id: sum(
                status == 'invalid'
                for (saved_section_id, _), status in loaded_spines.items()
                if saved_section_id == section_id
            )
            for section_id in section_ids
        }
        unset_by_section = {
            section_id: max(
                section_counts[section_id]
                - valid_by_section[section_id]
                - invalid_by_section[section_id],
                0,
            )
            for section_id in section_ids
        }

        figure_paths = stats_figure_paths()
        obsolete_figure_paths = [
            validation_csv_path.with_name(f'{validation_csv_path.stem}_overview.png'),
            validation_csv_path.with_name(f'{validation_csv_path.stem}_findings.png'),
            validation_csv_path.with_name(f'{validation_csv_path.stem}_finding_heatmap.png'),
            validation_csv_path.with_name(f'{validation_csv_path.stem}_section_review.png'),
        ]
        for figure_path in [*figure_paths, *obsolete_figure_paths]:
            if figure_path.exists():
                figure_path.unlink()


        # Per-section validity composition.
        figure, axis = plt.subplots(
            figsize=(max(11, len(section_ids) * 0.24), 6)
        )
        bottoms = np.zeros(len(section_ids))
        for label, values, color in (
            ('Valid Spines', [valid_by_section[s] for s in section_ids], '#188038'),
            ('Invalid Spines', [invalid_by_section[s] for s in section_ids], '#c5221f'),
            ('Not Set Spines', [unset_by_section[s] for s in section_ids], '#f29900'),
        ):
            axis.bar(section_ids, values, bottom=bottoms, label=label, color=color)
            bottoms += np.asarray(values)
        axis.set_title('Spine Validity by Section')
        axis.set_xlabel('Section ID', fontproperties=axis_label_font)
        axis.set_ylabel('Spine Count', fontproperties=axis_label_font)
        axis.legend()
        axis.grid(axis='y', alpha=0.25)
        if len(section_ids) > 30:
            axis.set_xticks(section_ids[::max(1, len(section_ids) // 20)])
        figure.tight_layout()
        figure.savefig(figure_paths[0], dpi=150, bbox_inches='tight')
        plt.close(figure)

        # Per-section positive issue counts.
        issue_series = (
            ('False Positive', loaded_false_positives, '#c5221f'),
            ('Incomplete Spine', loaded_incomplete_spines, '#e37400'),
            ('False Positive Quality', loaded_false_positive_quality, '#f9ab00'),
            ('Merged Spine', loaded_merged_spines, '#9334e6'),
            ('Split Spine', loaded_split_spines, '#1a73e8'),
        )
        figure, axis = plt.subplots(
            figsize=(max(11, len(section_ids) * 0.24), 6)
        )
        bottoms = np.zeros(len(section_ids))
        for label, results, color in issue_series:
            values = [
                sum(
                    status == 'yes'
                    for (saved_section_id, _), status in results.items()
                    if saved_section_id == section_id
                )
                for section_id in section_ids
            ]
            axis.bar(section_ids, values, bottom=bottoms, label=label, color=color)
            bottoms += np.asarray(values)
        axis.set_title('Spine Issues by Section')
        axis.set_xlabel('Section ID', fontproperties=axis_label_font)
        axis.set_ylabel('Issue Count', fontproperties=axis_label_font)
        axis.legend()
        axis.grid(axis='y', alpha=0.25)
        if len(section_ids) > 30:
            axis.set_xticks(section_ids[::max(1, len(section_ids) // 20)])
        figure.tight_layout()
        figure.savefig(figure_paths[1], dpi=150, bbox_inches='tight')
        plt.close(figure)

        return figure_paths


    async def wait_for_validation_save():
        """Wait for pending CSV persistence and guarantee a current CSV exists."""
        task = validation_save_task[0]
        if task is not None and not task.done():
            await task
        if pending_validation_snapshot[0] is not None:
            rows = pending_validation_snapshot[0]
            pending_validation_snapshot[0] = None
            await asyncio.to_thread(write_validation_csv, validation_csv_path, rows)
        elif not validation_csv_path.exists():
            await asyncio.to_thread(
                write_validation_csv,
                validation_csv_path,
                snapshot_validation_state(),
            )


    async def validation_save_worker():
        """Write the newest pending snapshot without blocking the notebook loop."""
        while pending_validation_snapshot[0] is not None:
            rows = pending_validation_snapshot[0]
            pending_validation_snapshot[0] = None
            await asyncio.to_thread(write_validation_csv, validation_csv_path, rows)


    def report_validation_save_error(task):
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as exc:
            print(f'Failed to save validation state: {exc}')


    def schedule_validation_save():
        """Queue the newest validation state for asynchronous persistence."""
        pending_validation_snapshot[0] = snapshot_validation_state()
        task = validation_save_task[0]
        if task is None or task.done():
            task = asyncio.create_task(validation_save_worker())
            task.add_done_callback(report_validation_save_error)
            validation_save_task[0] = task


    async def restore_validation_state():
        """Load a saved validation snapshot without blocking the notebook UI."""
        try:
            (
                loaded_spines,
                loaded_correct_type,
                loaded_valid_structure,
                loaded_false_positives,
                loaded_incomplete_spines,
                loaded_false_positive_quality,
                loaded_merged_spines,
                loaded_split_spines,
                loaded_sections,
                loaded_subsection_counts,
                loaded_subsection_done,
            ) = read_validation_csv(validation_csv_path)
        except Exception as exc:
            print(f'Could not restore validation state: {exc}')
            return

        validation_results.clear()
        validation_results.update(loaded_spines)
        correct_type_results.clear()
        correct_type_results.update(loaded_correct_type)
        valid_structure_results.clear()
        valid_structure_results.update(loaded_valid_structure)
        false_positive_results.clear()
        false_positive_results.update(loaded_false_positives)
        incomplete_spine_results.clear()
        incomplete_spine_results.update(loaded_incomplete_spines)
        false_positive_quality_results.clear()
        false_positive_quality_results.update(loaded_false_positive_quality)
        merged_spine_results.clear()
        merged_spine_results.update(loaded_merged_spines)
        split_spine_results.clear()
        split_spine_results.update(loaded_split_spines)
        section_missing_results.clear()
        section_missing_results.update(loaded_sections)
        subsection_missing_counts.clear()
        subsection_missing_counts.update(loaded_subsection_counts)
        subsection_done_results.clear()
        subsection_done_results.update(loaded_subsection_done)
        schedule_validation_save()
        refresh_section_dropdown()
        update_info()
        if (
            loaded_spines
            or loaded_correct_type
            or loaded_valid_structure
            or loaded_false_positives
            or loaded_incomplete_spines
            or loaded_false_positive_quality
            or loaded_merged_spines
            or loaded_split_spines
            or loaded_sections
            or loaded_subsection_counts
            or loaded_subsection_done
        ):
            print(f'Restored validation state from {validation_csv_path}')


    def report_validation_restore_error(task):
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as exc:
            print(f'Failed to restore validation state: {exc}')


    async def restore_and_start():
        await restore_validation_state()
        await wait_for_validation_save()
        start_load_section(0)


    # Spine colors
    HIGHLIGHT_COLOR = 0xFFFF00  # Retained for compatibility; selection is shown by the bbox


    def make_distinct_palette(count):
        """Generate a deterministic, saturated RGB palette for spine meshes."""
        if count < 0:
            raise ValueError('Palette size cannot be negative')

        colors = []
        non_blue_hue_span = (
            PALETTE_BLUE_HUE_START
            + (1.0 - PALETTE_BLUE_HUE_END)
        )
        for index in range(count):
            hue_position = (
                index * PALETTE_HUE_STEP
            ) % 1.0 * non_blue_hue_span
            if hue_position < PALETTE_BLUE_HUE_START:
                hue = hue_position
            else:
                hue = (
                    PALETTE_BLUE_HUE_END
                    + hue_position
                    - PALETTE_BLUE_HUE_START
                )
            red, green, blue = colorsys.hsv_to_rgb(
                hue,
                PALETTE_SATURATION,
                PALETTE_VALUE,
            )
            rgb = (
                round(red * 255),
                round(green * 255),
                round(blue * 255),
            )
            colors.append(
                (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]
            )
        return colors


    # ============================================================
    # Helper functions
    # ============================================================

    def add_to_plot(obj):
        """Add a k3d drawable through K3D's trait-aware registration path."""
        nonlocal plot
        plot += obj

    def remove_from_plot(obj):
        """Remove a k3d drawable through K3D's registration path."""
        nonlocal plot
        plot -= obj


    def update_bbox(vertices):
        """Update bounding box lines around given vertices."""
        if vertices is None or len(vertices) == 0:
            for line in bbox_lines:
                line.visible = False
            return
        bmin = vertices.min(axis=0)
        bmax = vertices.max(axis=0)
        x0, y0, z0 = bmin
        x1, y1, z1 = bmax
        corners = np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ], dtype=np.float32)
        for k, (i, j) in enumerate(edges):
            bbox_lines[k].vertices = np.array([corners[i], corners[j]], dtype=np.float32)
            bbox_lines[k].visible = True


    def hide_bbox():
        for line in bbox_lines:
            line.visible = False


    def set_camera_on(center, radius):
        """Set camera looking at center from +Z."""
        plot.camera_auto_fit = False
        d = max(float(radius), 1e-6) * 1.5
        plot.camera = [
            float(center[0]), float(center[1]), float(center[2] + d),
            float(center[0]), float(center[1]), float(center[2]),
            0.0, 1.0, 0.0,
        ]


    def set_camera_on_oriented(center, radius, view_direction):
        """Set the camera on a direction with a stable perpendicular up vector."""
        center = np.asarray(center, dtype=np.float32)
        view = np.asarray(view_direction, dtype=np.float32)
        view_norm = float(np.linalg.norm(view))
        if (
            center.shape != (3,)
            or not np.all(np.isfinite(center))
            or view.shape != (3,)
            or not np.all(np.isfinite(view))
            or view_norm <= 1e-6
        ):
            set_camera_on(center, radius)
            return

        view /= view_norm
        basis = np.eye(3, dtype=np.float32)
        for candidate in basis[np.argsort(np.abs(basis @ view))]:
            up = candidate - np.dot(candidate, view) * view
            up_norm = float(np.linalg.norm(up))
            if up_norm > 1e-6:
                up /= up_norm
                break
        else:
            set_camera_on(center, radius)
            return

        plot.camera_auto_fit = False
        distance = max(float(radius), 1e-6) * 1.5
        position = center + view * distance
        plot.camera = position.tolist() + center.tolist() + up.tolist()


    def set_projection(view_direction, up_direction):
        """Set an orthographic axis view while preserving the current target."""
        camera = np.asarray(plot.camera, dtype=np.float32)
        target = camera[3:6]
        distance = float(np.linalg.norm(camera[:3] - target))
        distance = max(distance, 1.0)
        direction = np.asarray(view_direction, dtype=np.float32)
        up = np.asarray(up_direction, dtype=np.float32)
        plot.camera_auto_fit = False
        position = target + direction * distance
        plot.camera = position.tolist() + target.tolist() + up.tolist()


    def current_spine_id():
        """Return the morphology spine ID for the currently selected spine."""
        sec_id = spiny_sections[current_section_idx[0]][0]
        spine_ids = list(morphology.spines.spine_indices_for_section(sec_id + 1))
        spine_idx = current_spine_idx[0]
        return spine_ids[spine_idx] if spine_idx < len(spine_ids) else spine_idx + 1


    def screenshot_path():
        """Build the next available screenshot path for the current view.

        Section-only captures use ``section_<section_id>_error_<sequence>.png``.
        Normal selected-spine captures use
        ``section_<section_id>_spine_<spine_id>_error_<sequence>.png``.
        Structure captures use
        ``section_<section_id>_spine_<spine_id>_structure_<sequence>.png``.
        The sequence is scoped to the active filename prefix and is the snapshot ID.
        """
        sec_id = spiny_sections[current_section_idx[0]][0]
        output_dir = proofreading_dir

        if spine_selected[0] and current_spine_data:
            spine_id = current_spine_id()
            view_marker = (
                'structure'
                if spine_structure_visible[0]
                else 'error'
            )
            prefix = f'section_{sec_id}_spine_{spine_id}_{view_marker}_'
        else:
            prefix = f'section_{sec_id}_error_'

        pattern = re.compile(rf'^{re.escape(prefix)}(\d+)\.png$')
        sequence_numbers = []
        for candidate in output_dir.glob(f'{prefix}*.png'):
            match = pattern.match(candidate.name)
            if match is not None:
                sequence_numbers.append(int(match.group(1)))

        next_sequence = max(sequence_numbers, default=0) + 1
        return output_dir / f'{prefix}{next_sequence}.png'


    SCALE_BAR_UNIT_LABEL = 'µm'
    SCALE_BAR_WORLD_UNITS_PER_DISPLAY_UNIT = 1.0

    def scale_bar_js():
        """Build a screen-fixed, camera-aware scale-bar overlay."""
        unit_label = SCALE_BAR_UNIT_LABEL.replace('\\', '\\\\').replace("'", "\\'")
        world_units_per_display_unit = SCALE_BAR_WORLD_UNITS_PER_DISPLAY_UNIT
        return f"""
(() => {{
    const widgetView = this;
    const scaleBarId = 'obi-scale-bar-overlay';
    const unitLabel = '{unit_label}';
    const worldUnitsPerDisplayUnit = {world_units_per_display_unit};
    const canvas = K3DInstance.getWorld().renderer.domElement;
    if (!canvas) {{
        return;
    }}

    const host = canvas.parentElement || canvas;
    if (host !== canvas && getComputedStyle(host).position === 'static') {{
        host.style.position = 'relative';
    }}

    let overlay = host.querySelector('#' + scaleBarId);
    if (!overlay) {{
        overlay = document.createElement('div');
        overlay.id = scaleBarId;
        overlay.setAttribute('aria-label', '3D viewer scale bar');
        overlay.style.cssText = [
            'position:absolute',
            'left:14px',
            'bottom:14px',
            'z-index:20',
            'display:flex',
            'flex-direction:column',
            'align-items:flex-start',
            'gap:4px',
            'padding:5px 7px 6px',
            'border-radius:3px',
            'background:rgba(0,0,0,0.58)',
            'color:#ffffff',
            'font-family:sans-serif',
            'font-size:12px',
            'font-weight:600',
            'line-height:1.1',
            'pointer-events:none',
            'user-select:none',
            'box-sizing:border-box',
            'white-space:nowrap'
        ].join(';');

        const label = document.createElement('div');
        label.id = scaleBarId + '-label';
        const bar = document.createElement('div');
        bar.id = scaleBarId + '-bar';
        bar.style.cssText = [
            'height:4px',
            'min-width:1px',
            'background:#ffffff',
            'border:1px solid #111111',
            'box-sizing:border-box'
        ].join(';');
        overlay.appendChild(label);
        overlay.appendChild(bar);
        host.appendChild(overlay);
    }}

    const label = overlay.querySelector('#' + scaleBarId + '-label');
    const bar = overlay.querySelector('#' + scaleBarId + '-bar');
    const world = K3DInstance.getWorld();
    let framePending = false;

    const update = () => {{
        const rect = canvas.getBoundingClientRect();
        const canvasWidth = rect.width;
        const canvasHeight = rect.height;
        if (!(canvasWidth > 0 && canvasHeight > 0)) {{
            return;
        }}

        const cameraValues = widgetView.model && widgetView.model.get('camera');
        if (!cameraValues || cameraValues.length < 6) {{
            return;
        }}
        const cameraPosition = [
            Number(cameraValues[0]),
            Number(cameraValues[1]),
            Number(cameraValues[2])
        ];
        const cameraTarget = [
            Number(cameraValues[3]),
            Number(cameraValues[4]),
            Number(cameraValues[5])
        ];
        const cameraDistance = Math.hypot(
            cameraPosition[0] - cameraTarget[0],
            cameraPosition[1] - cameraTarget[1],
            cameraPosition[2] - cameraTarget[2]
        );
        if (!(cameraDistance > 0)) {{
            return;
        }}

        const camera = world.camera;
        const fieldOfView = camera && Number.isFinite(Number(camera.fov))
            ? Number(camera.fov)
            : 45;
        const worldUnitsPerPixel = (
            2 * cameraDistance * Math.tan(fieldOfView * Math.PI / 360)
        ) / canvasHeight;
        if (!(worldUnitsPerPixel > 0)) {{
            return;
        }}

        const targetPixels = Math.min(180, Math.max(90, canvasWidth * 0.22));
        const rawWorldLength = (
            worldUnitsPerPixel * targetPixels * worldUnitsPerDisplayUnit
        );
        const exponent = Math.floor(Math.log10(rawWorldLength));
        const magnitude = Math.pow(10, exponent);
        const candidates = [1, 2, 5, 10];
        const normalized = rawWorldLength / magnitude;
        const niceFactor = candidates.reduce((closest, candidate) =>
            Math.abs(candidate - normalized) < Math.abs(closest - normalized)
                ? candidate
                : closest
        );
        const worldLength = niceFactor * magnitude;
        const displayLength = worldLength / worldUnitsPerDisplayUnit;
        const barPixels = Math.max(1, Math.round(worldLength / worldUnitsPerPixel));
        const labelValue = displayLength >= 10
            ? String(Math.round(displayLength))
            : displayLength.toFixed(displayLength < 1 ? 2 : 1).replace(/\\.0+$/, '');

        bar.style.width = barPixels + 'px';
        label.textContent = labelValue + ' ' + unitLabel;
        overlay.setAttribute('aria-label', 'Scale bar: ' + label.textContent);
    }};

    const refresh = () => {{
        if (framePending) {{
            return;
        }}
        framePending = true;
        window.requestAnimationFrame(() => {{
            framePending = false;
            update();
        }});
    }};

    update();
    window.addEventListener('resize', refresh, {{passive:true}});
    if (typeof ResizeObserver !== 'undefined') {{
        new ResizeObserver(refresh).observe(host);
    }}
    if (widgetView.model && typeof widgetView.model.on === 'function') {{
        widgetView.model.on('change:camera', refresh);
        widgetView.model.on('change:background_color', refresh);
    }}
    window.setInterval(update, 200);
}})();
"""


    def live_canvas_capture_js(request_token):
        """Build frontend code that copies the already-rendered WebGL canvas."""
        return f"""
(() => {{
    const widgetView = this;
    const requestToken = {request_token};
    const reportError = (error) => {{
        const message = error && error.message ? error.message : String(error);
        widgetView.model.save(
            'screenshot',
            `__K3D_CAPTURE_ERROR__${{requestToken}}:${{message}}`,
            {{patch: true}}
        );
    }};

    Promise.resolve(K3DInstance.render(true)).then(() => {{
        const canvas = K3DInstance.getWorld().renderer.domElement;
        if (!canvas) {{
            throw new Error('The K3D WebGL canvas is unavailable.');
        }}
        canvas.toBlob((blob) => {{
            if (!blob) {{
                reportError(new Error('The browser could not encode the K3D canvas.'));
                return;
            }}
            const reader = new FileReader();
            reader.onloadend = () => {{
                const result = typeof reader.result === 'string' ? reader.result : '';
                const separator = result.indexOf(',');
                const encoded = separator >= 0 ? result.slice(separator + 1) : '';
                if (!encoded) {{
                    reportError(new Error('The encoded K3D canvas is empty.'));
                    return;
                }}
                widgetView.model.save(
                    'screenshot',
                    `__K3D_CAPTURE__${{requestToken}}:${{encoded}}`,
                    {{patch: true}}
                );
            }};
            reader.onerror = () => reportError(reader.error || new Error('PNG read failed.'));
            reader.readAsDataURL(blob);
        }}, 'image/png');
    }}).catch(reportError);
}})();
"""


    def write_screenshot_file(encoded_image, destination):
        """Decode and write a live-canvas PNG outside the notebook event loop."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(base64.b64decode(encoded_image, validate=True))


    async def save_screenshot_file(encoded_image, destination, request_token):
        """Persist an exact WebGL canvas capture without blocking widget callbacks."""
        try:
            await asyncio.to_thread(write_screenshot_file, encoded_image, destination)
            print(f'Saved screenshot to {destination}')
        except Exception as exc:
            print(f'Failed to save screenshot: {exc}')
        finally:
            if screenshot_request_token[0] == request_token:
                screenshot_request_path[0] = None
                btn_screenshot.disabled = False


    def on_screenshot_ready(change):
        """Receive the exact live-canvas PNG returned by the K3D frontend."""
        payload = change.get('new')
        if not isinstance(payload, str) or not payload:
            return

        if payload.startswith('__K3D_CAPTURE_ERROR__'):
            token_text, _, message = payload.removeprefix('__K3D_CAPTURE_ERROR__').partition(':')
            try:
                request_token = int(token_text)
            except ValueError:
                return
            if request_token != screenshot_request_token[0]:
                return
            screenshot_request_path[0] = None
            btn_screenshot.disabled = False
            print(f'Failed to capture screenshot: {message}')
            return

        if not payload.startswith('__K3D_CAPTURE__'):
            return
        token_text, separator, encoded_image = payload.removeprefix('__K3D_CAPTURE__').partition(':')
        if not separator or not encoded_image:
            return
        try:
            request_token = int(token_text)
        except ValueError:
            return
        if request_token != screenshot_request_token[0]:
            return
        destination = screenshot_request_path[0]
        if destination is None:
            return
        screenshot_save_task[0] = asyncio.create_task(
            save_screenshot_file(encoded_image, destination, request_token)
        )


    def take_screenshot(_):
        """Capture the displayed K3D canvas with its exact WebGL shading."""
        if screenshot_request_path[0] is not None:
            return

        screenshot_request_token[0] += 1
        request_token = screenshot_request_token[0]
        screenshot_request_path[0] = screenshot_path()
        btn_screenshot.disabled = True
        try:
            # Updating this trait executes the capture in the existing K3D PlotView.
            # It uses the normal live render, never K3D's expensive renderOffScreen path.
            plot.additional_js_code = live_canvas_capture_js(request_token)
        except Exception as exc:
            screenshot_request_path[0] = None
            btn_screenshot.disabled = False
            print(f'Failed to request screenshot: {exc}')


    def registration_upload_succeeded(result):
        """Recognize successful responses from the Drive upload helper."""
        if not isinstance(result, dict):
            return False
        if result.get('error'):
            return False

        success = result.get('success')
        if isinstance(success, str):
            return success.strip().lower() in {'true', 'success', 'ok', 'uploaded'}
        if success is True:
            return True

        status = result.get('status')
        if isinstance(status, str) and status.strip().lower() in {
            'success', 'ok', 'uploaded', 'complete', 'completed'
        }:
            return True

        # Some Apps Script responses omit a success flag but return the uploaded
        # filename or URL. upload_file has already raised for explicit errors.
        return bool(result.get('filename') or result.get('url'))


    def registration_identifier():
        """Return a mesh-ID/date-time directory for this assessment."""
        mesh_id = Path(mesh_path).stem
        date_time = datetime.now(timezone.utc).strftime('%y.%m.%d_%H.%M')
        user_token = re.sub(r'[^A-Za-z0-9_.-]+', '_', user_email).strip('._-')
        user_token = user_token or 'unknown-user'
        return mesh_path.parent / f'{mesh_id}_{user_token}_{date_time}'


    def stage_registration_directory(destination, report_path):
        """Stage exactly the validation CSV and analysis PDF for registration."""
        destination = Path(destination)
        report_path = Path(report_path)
        destination.mkdir(parents=True, exist_ok=False)

        if not validation_csv_path.exists():
            write_validation_csv(validation_csv_path, snapshot_validation_state())
        if not report_path.is_file():
            raise FileNotFoundError(f'Registration analysis PDF does not exist: {report_path}')
        if report_path.suffix.lower() != '.pdf' or not report_path.stem.lower().endswith('_analysis'):
            raise ValueError(
                f'Registration requires an analysis PDF named *_analysis.pdf: {report_path}'
            )

        csv_destination = destination / validation_csv_path.name
        pdf_destination = destination / report_path.name
        shutil.copy2(validation_csv_path, csv_destination)
        shutil.copy2(report_path, pdf_destination)

        staged_files = {path.name for path in destination.iterdir() if path.is_file()}
        expected_files = {csv_destination.name, pdf_destination.name}
        if staged_files != expected_files:
            raise RuntimeError(
                f'Registration staging must contain only CSV and analysis PDF; found {sorted(staged_files)}'
            )
        return destination


    def load_registration_uploader():
        """Load the current registration helper code for this notebook kernel."""
        try:
            registration_module = importlib.import_module(
                'validate_spines_registration'
            )
        except ModuleNotFoundError as exc:
            if exc.name != 'validate_spines_registration':
                raise
            registration_module = importlib.import_module(
                'examples.validate_spines_registration'
            )

        registration_module = importlib.reload(registration_module)
        upload_zip = getattr(registration_module, 'upload_zip', None)
        if not callable(upload_zip):
            raise TypeError(
                f'Registration helper {registration_module.__file__} '
                'does not define callable upload_zip'
            )
        return upload_zip, registration_module.__file__


    async def register_assessment_worker():
        """Generate the analysis PDF, stage CSV+PDF, ZIP them, and upload."""
        try:
            registration_status.value = '<i>1/4 Saving validation data...</i>'
            print('[Registration] 1/4 Saving validation CSV...')
            await wait_for_validation_save()

            screenshot_task = screenshot_save_task[0]
            if screenshot_task is not None and not screenshot_task.done():
                registration_status.value = '<i>1/4 Waiting for the latest screenshot...</i>'
                await screenshot_task

            registration_status.value = '<i>2/4 Generating analysis artifacts...</i>'
            print('[Registration] 2/4 Generating analysis data...')
            analysis_module = importlib.import_module('validate_spine_analysis')
            analysis_module = importlib.reload(analysis_module)
            analysis_pdf_path = await asyncio.to_thread(
                analysis_module.generate_report,
                images_dir=validation_csv_path.parent,
                csv_path=validation_csv_path,
                fonts_dir=Path('lib') / 'fonts',
                output_dir=validation_csv_path.parent,
            )
            analysis_pdf_path = Path(analysis_pdf_path)
            if not analysis_pdf_path.is_file():
                raise FileNotFoundError(
                    f'Analysis PDF generation did not produce: {analysis_pdf_path}'
                )
            print(f'[Registration] Analysis PDF: {analysis_pdf_path.name}')

            destination = registration_identifier()
            registration_status.value = '<i>3/4 Staging analysis data...</i>'
            print('[Registration] 3/4 Staging analysis data...')
            await asyncio.to_thread(
                stage_registration_directory,
                destination,
                analysis_pdf_path,
            )

            upload_zip, helper_path = load_registration_uploader()
            print(f'[Registration] Using helper: {helper_path}')
            registration_status.value = (
                '<i>4/4 Registering file...</i>'
            )
            print('[Registration] 4/4 Registering results...')
            message = 'Validation CSV and analysis PDF registration'
            result = await asyncio.to_thread(
                upload_zip,
                str(destination),
                'archives',
                message=message,
            )
            if registration_upload_succeeded(result):
                registration_status.value = (
                    f'<span style="color:#188038">'
                    f'Registered: {destination.name}</span>'
                )
            else:
                error = result.get('error') if isinstance(result, dict) else result
                safe_error = html.escape(str(error))
                registration_status.value = (
                    f'<span style="color:#c5221f">'
                    f'Registration upload failed: {safe_error}</span>'
                )
                print(f'Registration upload failed: {error}')
        except Exception as exc:
            error_detail = html.escape(f'{type(exc).__name__}: {exc}')
            registration_status.value = (
                f'<span style="color:#c5221f">'
                f'Registration failed: {error_detail}</span>'
            )
            destination_name = locals().get('destination', '<not staged>')
            print(
                f'Registration failed ({type(exc).__name__}) '
                f'for {destination_name}: {exc}'
            )
        finally:
            btn_register_assessment.disabled = False
            registration_task[0] = None


    def on_register_assessment(_):
        """Create and upload a CSV-only-and-analysis-PDF ZIP archive."""
        if registration_task[0] is not None and not registration_task[0].done():
            return
        registration_status.value = '<i>Registering assessment...</i>'
        btn_register_assessment.disabled = True
        registration_task[0] = asyncio.create_task(register_assessment_worker())


    async def generate_report_worker():
        """Persist the current assessment and create its analysis PDF."""
        try:
            report_status.value = '<i>Saving validation data...</i>'
            await wait_for_validation_save()

            screenshot_task = screenshot_save_task[0]
            if screenshot_task is not None and not screenshot_task.done():
                report_status.value = '<i>Waiting for the latest screenshot...</i>'
                await screenshot_task

            report_status.value = '<i>Generating PDF report...</i>'
            report_module = importlib.import_module('validate_spine_analysis')
            report_module = importlib.reload(report_module)
            report_path = await asyncio.to_thread(
                report_module.generate_report,
                images_dir=validation_csv_path.parent,
                csv_path=validation_csv_path,
                fonts_dir=Path('lib') / 'fonts',
                output_dir=validation_csv_path.parent,
            )
            report_status.value = 'Report Generated'
        except Exception as exc:
            error_detail = html.escape(f'{type(exc).__name__}: {exc}')
            report_status.value = (
                f'<span style="color:#c5221f">'
                f'Report generation failed: {error_detail}</span>'
            )
            print(f'Report generation failed: {type(exc).__name__}: {exc}')
        finally:
            btn_generate_report.disabled = False
            report_task[0] = None


    def on_generate_report(_):
        """Create the current assessment PDF without registering it."""
        if report_task[0] is not None and not report_task[0].done():
            return
        btn_generate_report.disabled = True
        report_status.value = '<i>Preparing report...</i>'
        report_task[0] = asyncio.create_task(generate_report_worker())


    def focus_on_section(sec_id):
        """Focus the camera on the whole selected section without a bbox."""
        if sec_id not in sections_points:
            hide_bbox()
            return

        section_points = sections_points[sec_id]
        section_min = section_points.min(axis=0)
        section_max = section_points.max(axis=0)
        section_center = (section_min + section_max) / 2.0
        section_radius = max(float(np.linalg.norm(section_max - section_min)) * 0.6, 1e-6)
        set_camera_on(section_center, section_radius)


    def focus_on_subsection(sec_id, subsection_idx):
        """Focus the camera on a path-length subsection."""
        subsections = section_subsections.get(sec_id, [])
        if not 0 <= subsection_idx < len(subsections):
            hide_bbox()
            return

        subsection = subsections[subsection_idx]
        section_points = sections_points.get(sec_id)
        try:
            points = np.asarray(section_points, dtype=np.float32)
        except (TypeError, ValueError):
            points = None

        if (
            points is None
            or points.ndim != 2
            or points.shape[1] != 3
            or points.shape[0] == 0
            or not np.all(np.isfinite(points))
        ):
            set_camera_on(subsection['center'], subsection['radius'])
            return

        box_extent = points.max(axis=0) - points.min(axis=0)
        if not np.all(np.isfinite(box_extent)) or float(np.max(box_extent)) <= 1e-6:
            set_camera_on(subsection['center'], subsection['radius'])
            return

        view_direction = np.zeros(3, dtype=np.float32)
        view_direction[int(np.argmax(box_extent))] = 1.0
        set_camera_on_oriented(
            subsection['center'], subsection['radius'], view_direction
        )


    # Expand each section's local query radius slightly so vertices on or just
    # beyond the morphology centerline extent are not clipped at the boundary.
    SECTION_QUERY_RADIUS_SCALE = 1.05
    EMPTY_POINT_CLOUD = np.empty((0, 3), dtype=np.float32)


    def clear_section_point_cloud():
        """Hide the full-resolution section cloud while a section is changing."""
        section_point_cloud.positions = EMPTY_POINT_CLOUD
        section_point_cloud.visible = False


    def update_section_point_cloud(sec_id):
        """Show all mesh vertices within the selected section's spatial extent."""
        if sec_id not in sections_points:
            clear_section_point_cloud()
            return

        section_points = sections_points[sec_id]
        section_min = section_points.min(axis=0).astype(np.float32)
        section_max = section_points.max(axis=0).astype(np.float32)
        section_center = (section_min + section_max) * 0.5
        section_half_extent = (section_max - section_min) * 0.5
        section_half_extent = np.maximum(
            section_half_extent * SECTION_QUERY_RADIUS_SCALE,
            np.finfo(np.float32).eps,
        )
        section_min = section_center - section_half_extent
        section_max = section_center + section_half_extent
        local_vertices = mesh_spatial_index.query_box(section_min, section_max)

        # Keep the fast default path at the exact mesh-vertex density. The
        # random surface sampler remains available by increasing the multiplier.
        dense_points = EMPTY_POINT_CLOUD
        if SECTION_DENSE_SAMPLE_MULTIPLIER > 0 and mesh_surface_sampler is not None:
            dense_target = len(local_vertices) * SECTION_DENSE_SAMPLE_MULTIPLIER
            dense_points = mesh_surface_sampler.sample_within_box(
                section_min, section_max, dense_target, section_point_rng
            )
        local_points = (
            np.concatenate([local_vertices, dense_points], axis=0)
            if len(dense_points) else local_vertices
        )

        section_point_cloud.positions = local_points
        section_point_cloud.visible = len(local_points) > 0


    def clear_section_centerline():
        """Hide the reusable selected-section centerline overlay."""
        section_centerline_overlay.visible = False

    def show_section_centerline(sec_id):
        """Update and show the reusable selected-section red overlay."""
        if sec_id not in sections_points:
            clear_section_centerline()
            return

        section_centerline_overlay.vertices = np.asarray(
            sections_points[sec_id], dtype=np.float32
        )
        section_centerline_overlay.color = (
            SECTION_CENTERLINE_HOT_COLOR
            if background_is_black[0]
            else SECTION_CENTERLINE_COLOR
        )
        section_centerline_overlay.width = SECTION_CENTERLINE_WIDTH
        section_centerline_overlay.visible = True


    def clear_subsection_centerline():
        """Hide the reusable orange subsection overlay."""
        subsection_centerline_overlay.visible = False


    def show_subsection_centerline(sec_id, subsection_idx, focus_camera=True):
        """Show and optionally focus the current orange subsection overlay."""
        subsections = section_subsections.get(sec_id, [])
        if not 0 <= subsection_idx < len(subsections):
            clear_subsection_centerline()
            return

        subsection = subsections[subsection_idx]
        subsection_centerline_overlay.vertices = subsection['points']
        subsection_centerline_overlay.color = SUBSECTION_CENTERLINE_COLOR
        subsection_centerline_overlay.width = SUBSECTION_CENTERLINE_WIDTH
        subsection_centerline_overlay.visible = True
        if focus_camera:
            focus_on_subsection(sec_id, subsection_idx)


    def clear_spine_structure():
        """Remove head/neck meshes and restore the selected full-spine mesh."""
        for obj in spine_structure_meshes_k3d:
            try:
                remove_from_plot(obj)
            except Exception:
                pass
        spine_structure_meshes_k3d.clear()
        if spine_structure_base_mesh[0] is not None:
            spine_structure_base_mesh[0].visible = True
        spine_structure_base_mesh[0] = None
        spine_structure_visible[0] = False


    def clear_spine_meshes():
        """Remove all spine mesh objects from the plot."""
        clear_spine_structure()
        for obj in current_spine_meshes_k3d:
            try:
                remove_from_plot(obj)
            except Exception:
                pass
        current_spine_meshes_k3d.clear()
        current_spine_data.clear()
        current_spine_global_ids.clear()
        current_spine_colors.clear()
        previous_colored_spine_idx[0] = None
        spine_colors_initialized[0] = False
        spine_selected[0] = False
        update_spine_analysis_button_state()


    async def load_section(sec_idx):
        """Load a section by focusing the branch, then adding spine meshes incrementally."""
        load_generation[0] += 1
        generation = load_generation[0]

        # Remove the previous section's local point cloud before its spine objects.
        clear_section_point_cloud()
        clear_spine_meshes()
        clear_section_centerline()
        clear_subsection_centerline()
        hide_bbox()

        sec_id, spine_count = spiny_sections[sec_idx]
        current_subsection_idx[0] = 0
        current_spine_idx[0] = 0
        spine_selected[0] = False

        # Focus on the branch before retrieving or displaying any spine meshes.
        if sec_id in sections_points:
            pts = sections_points[sec_id]
            pts_min = pts.min(axis=0)
            pts_max = pts.max(axis=0)
            sec_center = (pts_min + pts_max) / 2.0
            sec_radius = max(float(np.linalg.norm(pts_max - pts_min)) * 0.6, 1e-6)
            set_camera_on(sec_center, sec_radius)

        # Draw the selected section and its first 10 µm subsection.
        show_section_centerline(sec_id)
        show_subsection_centerline(
            sec_id,
            current_subsection_idx[0],
            focus_camera=True,
        )

        update_info()
        # Yield so the branch camera is rendered before spine meshes appear.
        await asyncio.sleep(SPINE_LOAD_DELAY)

        # Prepare all spine meshes concurrently. The K3D objects are still
        # created and added below on the notebook event loop, in stable order.
        spine_indices = list(morphology.spines.spine_indices_for_section(sec_id + 1))
        spine_list = await asyncio.gather(*(
            asyncio.to_thread(morphology.spines.spine_mesh, int(spine_idx))
            for spine_idx in spine_indices
        ))
        if generation != load_generation[0]:
            return

        colors = section_spine_colors.setdefault(
            sec_id, make_distinct_palette(len(spine_list))
        )

        for spine_idx, spine_mesh, color in zip(
            spine_indices, spine_list, colors, strict=True
        ):
            if spine_mesh.is_empty or len(spine_mesh.vertices) == 0:
                continue

            current_spine_data.append(spine_mesh)
            current_spine_global_ids.append(int(spine_idx))
            vertices = np.asarray(spine_mesh.vertices, dtype=np.float32)

            if len(spine_mesh.faces) == 0:
                obj = k3d.points(vertices, point_size=0.05, color=color)
            else:
                faces = np.asarray(spine_mesh.faces, dtype=np.uint32).reshape(-1)
                obj = k3d.mesh(
                    vertices, faces, color=color, opacity=1.0,
                    flat_shading=True, wireframe=False
                )

            add_to_plot(obj)
            current_spine_meshes_k3d.append(obj)
            current_spine_colors.append(color)

        if generation != load_generation[0]:
            return

        # After all meshes load, keep the orange subsection focused and show the
        # full-resolution point cloud for the selected section.
        show_subsection_centerline(
            sec_id,
            current_subsection_idx[0],
            focus_camera=True,
        )
        update_section_point_cloud(sec_id)
        update_info()


    def report_load_error(task):
        """Consume async loader failures so they are reported without task warnings."""
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as exc:
            print(f'Failed to load section meshes: {exc}')


    def start_load_section(sec_idx):
        """Schedule a section load and cancel any stale in-progress load."""
        if loading_task[0] is not None and not loading_task[0].done():
            loading_task[0].cancel()
        task = asyncio.create_task(load_section(sec_idx))
        task.add_done_callback(report_load_error)
        loading_task[0] = task


    def update_spine_colors():
        """Highlight the focused spine and dim the other section spines."""
        selected_idx = current_spine_idx[0]
        previous_idx = previous_colored_spine_idx[0]

        def set_color_if_needed(spine_idx, color):
            if not 0 <= spine_idx < len(current_spine_meshes_k3d):
                return
            spine_object = current_spine_meshes_k3d[spine_idx]
            if spine_object.color != color:
                spine_object.color = color

        def palette_color(spine_idx):
            if spine_idx >= len(current_spine_colors):
                return None
            return current_spine_colors[spine_idx]

        if not spine_selected[0]:
            # A section-level view shows every spine in its assigned palette
            # color until a specific spine is focused.
            for spine_idx in range(len(current_spine_meshes_k3d)):
                color = palette_color(spine_idx)
                if color is not None:
                    set_color_if_needed(spine_idx, color)
            spine_colors_initialized[0] = True
        elif not spine_colors_initialized[0] or previous_idx is None:
            # On first focus, keep the focused spine colorful and dim all peers.
            for spine_idx in range(len(current_spine_meshes_k3d)):
                color = palette_color(spine_idx)
                if color is None:
                    continue
                target_color = (
                    color
                    if spine_idx == selected_idx
                    else UNSELECTED_SPINE_COLOR
                )
                set_color_if_needed(spine_idx, target_color)
            spine_colors_initialized[0] = True
        elif previous_idx != selected_idx:
            # Restore the old focus to gray and restore the new focus's palette
            # color; all other non-focused spines remain gray.
            if previous_idx is not None:
                set_color_if_needed(previous_idx, UNSELECTED_SPINE_COLOR)
            color = palette_color(selected_idx)
            if color is not None:
                set_color_if_needed(selected_idx, color)

        previous_colored_spine_idx[0] = (
            selected_idx
            if spine_selected[0]
            and 0 <= selected_idx < len(current_spine_meshes_k3d)
            else None
        )


    def ensure_spine_analysis_defaults():
        """Set all Yes/No analyses to No when a spine is first selected."""
        if not spine_selected[0]:
            return
        sec_id = spiny_sections[current_section_idx[0]][0]
        key = (sec_id, current_spine_idx[0])
        changed = False
        for results in (
            correct_type_results,
            false_positive_results,
            incomplete_spine_results,
            false_positive_quality_results,
            merged_spine_results,
            split_spine_results,
        ):
            if key not in results:
                results[key] = 'no'
                changed = True
        if changed:
            schedule_validation_save()


    def highlight_current_spine(focus_camera=True):
        """Mark the current spine selected and update color, bbox, and camera."""
        clear_spine_structure()
        spine_idx = current_spine_idx[0]
        spine_selected[0] = (
            0 <= spine_idx < len(current_spine_meshes_k3d)
            and spine_idx < len(current_spine_data)
        )
        if spine_selected[0]:
            ensure_spine_analysis_defaults()
        update_spine_analysis_button_state()
        update_spine_colors()

        if spine_selected[0]:
            # Selection is shown by the bounding box and camera; every spine
            # keeps its own palette color.
            if spine_idx < len(current_spine_data) and current_spine_data[spine_idx] is not None:
                verts = np.asarray(current_spine_data[spine_idx].vertices, dtype=np.float32)
                update_bbox(verts)

                if focus_camera:
                    spine_center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
                    spine_radius = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))) * 0.6
                    spine_radius = max(spine_radius, 2.0)
                    set_camera_on(spine_center, spine_radius)
            else:
                hide_bbox()
        else:
            hide_bbox()


    def on_toggle_spine_structure(_):
        """Toggle rose head/green neck geometry for the selected spine."""
        if spine_structure_visible[0]:
            clear_spine_structure()
            update_spine_colors()
            update_spine_analysis_button_state()
            return

        spine_idx = current_spine_idx[0]
        if not (
            spine_selected[0]
            and 0 <= spine_idx < len(current_spine_meshes_k3d)
            and spine_idx < len(current_spine_global_ids)
        ):
            clear_spine_structure()
            update_spine_analysis_button_state()
            return

        spine_id = int(current_spine_global_ids[spine_idx])
        try:
            neck = morphology.spines.spine_mesh(
                spine_id,
                include_head=False,
            )
            head = morphology.spines.spine_mesh(
                spine_id,
                include_neck=False,
            )
            for component_mesh, color in (
                (neck, SPINE_NECK_STRUCTURE_COLOR),
                (head, SPINE_HEAD_STRUCTURE_COLOR),
            ):
                if component_mesh is None or len(component_mesh.faces) == 0:
                    continue
                obj = k3d.mesh(
                    np.asarray(component_mesh.vertices, dtype=np.float32),
                    np.asarray(component_mesh.faces, dtype=np.uint32).reshape(-1),
                    color=color,
                    opacity=1.0,
                    flat_shading=True,
                    wireframe=False,
                )
                add_to_plot(obj)
                spine_structure_meshes_k3d.append(obj)
        except Exception as exc:
            clear_spine_structure()
            print(f'Could not display spine structure for spine {spine_id}: {exc}')
            update_spine_analysis_button_state()
            return

        if not spine_structure_meshes_k3d:
            clear_spine_structure()
            update_spine_analysis_button_state()
            return

        spine_structure_base_mesh[0] = current_spine_meshes_k3d[spine_idx]
        spine_structure_base_mesh[0].visible = False
        spine_structure_visible[0] = True
        update_spine_analysis_button_state()


    def focus_first_spine():
        """Move to the first loaded spine in the current section."""
        if not current_spine_meshes_k3d:
            update_info()
            return
        current_spine_idx[0] = 0
        highlight_current_spine(focus_camera=True)
        update_info()


    def get_first_unvalidated_spine_index(sec_id, start_idx):
        """Find the next unvalidated spine, wrapping to the first one."""
        n_loaded = len(current_spine_meshes_k3d)
        if n_loaded == 0:
            return None
        for offset in range(1, n_loaded + 1):
            candidate = (start_idx + offset) % n_loaded
            if (sec_id, candidate) not in validation_results:
                return candidate
        return None


    def section_is_fully_validated(sec_id, spine_count):
        """Return whether every expected spine in the section is classified."""
        validated = {
            spine_idx
            for (sid, spine_idx), _ in validation_results.items()
            if sid == sec_id
        }
        return len(validated) >= spine_count


    def validity_status(key):
        """Return only an explicit validity state or the required unset label."""
        status = validation_results.get(key)
        return status if status in {'valid', 'invalid'} else VALIDITY_NOT_SET


    def update_spine_analysis_button_state():
        """Enable analysis actions only when a loaded spine is selected."""
        spine_selected_now = (
            spine_selected[0]
            and bool(current_spine_meshes_k3d)
            and 0 <= current_spine_idx[0] < len(current_spine_data)
        )
        for button in (
            btn_correct_type_yes,
            btn_correct_type_no,
            btn_false_positive_yes,
            btn_false_positive_no,
            btn_incomplete_spines_yes,
            btn_incomplete_spines_no,
            btn_false_positive_quality_yes,
            btn_false_positive_quality_no,
            btn_merged_spine_yes,
            btn_merged_spine_no,
            btn_split_spine_yes,
            btn_split_spine_no,
            btn_spine_done_next,
            btn_show_spine_structure,
        ):
            button.disabled = not spine_selected_now

        structure_review_active = (
            spine_selected_now and spine_structure_visible[0]
        )
        btn_valid_structure_yes.disabled = not structure_review_active
        btn_valid_structure_no.disabled = not structure_review_active

        key = (
            spiny_sections[current_section_idx[0]][0],
            current_spine_idx[0],
        )
        for results, yes_button, no_button in (
            (correct_type_results, btn_correct_type_yes, btn_correct_type_no),
            (false_positive_results, btn_false_positive_yes, btn_false_positive_no),
            (incomplete_spine_results, btn_incomplete_spines_yes, btn_incomplete_spines_no),
            (
                false_positive_quality_results,
                btn_false_positive_quality_yes,
                btn_false_positive_quality_no,
            ),
            (merged_spine_results, btn_merged_spine_yes, btn_merged_spine_no),
            (split_spine_results, btn_split_spine_yes, btn_split_spine_no),
        ):
            value = results.get(key) if spine_selected_now else None
            yes_button.button_style = 'warning' if value == 'yes' else ''
            no_button.button_style = 'success' if value == 'no' else ''

        valid_structure_value = (
            valid_structure_results.get(key)
            if structure_review_active
            else None
        )
        btn_valid_structure_yes.button_style = (
            'warning' if valid_structure_value == 'yes' else ''
        )
        btn_valid_structure_no.button_style = (
            'success' if valid_structure_value == 'no' else ''
        )

        issue_results = (
            false_positive_results,
            incomplete_spine_results,
            false_positive_quality_results,
            merged_spine_results,
            split_spine_results,
        )
        has_defect = spine_selected_now and any(
            results.get(key) == 'yes'
            for results in issue_results
        )
        all_issues_no = spine_selected_now and all(
            results.get(key) == 'no'
            for results in issue_results
        )
        btn_spine_done_next.disabled = not (all_issues_no or has_defect)
        btn_show_spine_structure.disabled = not spine_selected_now
        btn_show_spine_structure.description = (
            'Show Spine Geometry'
            if spine_structure_visible[0]
            else 'Show Spine Structure'
        )

        btn_next_spine.disabled = (
            not bool(current_spine_meshes_k3d)
            or (spine_selected_now and key not in validation_results)
        )
        btn_prev_spine.disabled = (
            len(current_spine_meshes_k3d) <= 1
            or not spine_selected_now
            or current_spine_idx[0] <= 0
        )
        btn_next_spine.disabled = (
            btn_next_spine.disabled
            or len(current_spine_meshes_k3d) <= 1
            or (
                spine_selected_now
                and current_spine_idx[0] >= len(current_spine_meshes_k3d) - 1
            )
        )


    def update_info():
        """Update the info label."""
        update_spine_analysis_button_state()
        sec_idx = current_section_idx[0]
        sec_id, spine_count = spiny_sections[sec_idx]
        spine_idx = current_spine_idx[0]
        n_loaded = len(current_spine_meshes_k3d)

        # Check if current spine already validated.
        key = (sec_id, spine_idx)
        status = validity_status(key)
        # Current section validation progress.
        section_validated = sum(
            1
            for (validated_sec_id, _), validation_status in validation_results.items()
            if validated_sec_id == sec_id
            and validation_status in {'valid', 'invalid'}
        )
        section_complete = section_validated >= spine_count

        # Overall validation progress across sections containing spines.
        total_spines = sum(
            section_spine_count
            for _, section_spine_count in spiny_sections
        )
        validated_spines = sum(
            1
            for validation_status in validation_results.values()
            if validation_status in {'valid', 'invalid'}
        )
        validated_sections = sum(
            1
            for section_id, section_spine_count in spiny_sections
            if sum(
                1
                for (validated_sec_id, _), validation_status in validation_results.items()
                if validated_sec_id == section_id
                and validation_status in {'valid', 'invalid'}
            ) >= section_spine_count
        )
        sections_complete = validated_sections == len(spiny_sections)
        spines_complete = validated_spines == total_spines
        current_subsections = section_subsections.get(sec_id, [])
        subsection_count = len(current_subsections)
        subsection_missing_total = sum(
            subsection_missing_counts.get((sec_id, subsection['index']), 0)
            for subsection in current_subsections
        )
        subsections_checked = sum(
            (sec_id, subsection['index']) in subsection_done_results
            for subsection in current_subsections
        )
        subsections_complete = subsections_checked >= subsection_count
        subsection_review_color = '#188038' if subsections_complete else '#f29900'
        missing_spines_color = (
            '#c5221f'
            if subsection_missing_total > 0
            else '#188038'
            if subsections_complete
            else '#f29900'
        )
        btn_prev_sec.disabled = (
            len(spiny_sections) <= 1
            or sec_idx <= 0
        )
        btn_next_sec.disabled = (
            len(spiny_sections) <= 1
            or sec_idx >= len(spiny_sections) - 1
        )
        btn_prev_subsection.disabled = (
            subsection_count <= 1
            or current_subsection_idx[0] <= 0
        )
        btn_next_subsection.disabled = (
            subsection_count <= 1
            or current_subsection_idx[0] >= subsection_count - 1
        )
        status_colors = {
            'valid': '#188038',
            'invalid': '#c5221f',
            VALIDITY_NOT_SET: '#f29900',
        }
        section_status_color = '#188038' if section_complete else '#f29900'
        section_spines_color = (
            '#188038' if section_validated >= spine_count else '#b06000'
        )
        sections_progress_color = '#188038' if sections_complete else '#f29900'
        spines_progress_color = '#188038' if spines_complete else '#f29900'

        info_label.value = (
            '<b>Current Section Status</b> '
            '<span title="This section shows the status of the current section being validated." '
            'style="cursor:help; color:#5f6368;">&#9432;</span>'
            f'<br><b>Section {sec_id}</b> ({sec_idx + 1}/{len(spiny_sections)}) '
            f'| Spine <b>{spine_idx + 1}/{n_loaded}</b> '
            f'| Validity: '
            f'<span style="color:{status_colors[status]}"><b>{status.title()}</b></span>'
            f'<br>Section Checked: '
            f'<span style="color:{section_status_color}"><b>{"Yes" if section_complete else "No"}</b></span>'
            f' | Spines Checked: '
            f'<span style="color:{section_spines_color}"><b>{section_validated}/{spine_count}</b></span>'
            f'<br>Subsections Review: '
            f'<span style="color:{subsection_review_color}"><b>{subsections_checked}/{subsection_count}</b></span>'
            f' | Missing Spines: '
            f'<span style="color:{missing_spines_color}"><b>{subsection_missing_total}</b></span>'
            '<br><b>Overall Status</b> '
            '<span title="This section summarizes the overall status of all the sections and spines. '
            'This summarizes whether all sections have been analyzed and the results '
            'are ready to be registered." '
            'style="cursor:help; color:#5f6368;">&#9432;</span>'
            f'<br>Sections Validated: '
            f'<span style="color:{sections_progress_color}"><b>{validated_sections}/{len(spiny_sections)}</b></span>'
            f' | Spines Validated: '
            f'<span style="color:{spines_progress_color}"><b>{validated_spines}/{total_spines}</b></span>'
        )

        # Keep the subsection selector synchronized with the current section.
        refresh_subsection_dropdown()
        # Keep the spine selector synchronized with the current loaded section.
        refresh_spine_dropdown()

        # Update stats
        update_stats()


    def update_stats():
        """Show aligned current-section and full-summary HTML tables."""
        stats_output.clear_output()
        with stats_output:
            selected_sec_id, selected_count = spiny_sections[current_section_idx[0]]
            selected_validated = sum(
                1 for (sid, _), _ in validation_results.items()
                if sid == selected_sec_id
            )
            selected_remaining = max(selected_count - selected_validated, 0)
            selected_complete = 'yes' if selected_remaining == 0 else 'no'
            selected_missing = normalize_missing_spines_status(
                section_missing_results.get(selected_sec_id)
            )

            def subsection_missing_total_for_section(sec_id):
                return sum(
                    subsection_missing_counts.get((sec_id, subsection['index']), 0)
                    for subsection in section_subsections.get(sec_id, [])
                )

            def missing_status_display(sec_id, status):
                if status.casefold() == 'missing':
                    return f'{status} [{subsection_missing_total_for_section(sec_id)}]'
                return status

            selected_missing_display = missing_status_display(
                selected_sec_id, selected_missing
            )

            finding_maps = (
                false_positive_results,
                incomplete_spine_results,
                false_positive_quality_results,
                merged_spine_results,
                split_spine_results,
                valid_structure_results,
            )

            def finding_counts_for_section(sec_id):
                return tuple(
                    sum(
                        status == 'yes'
                        for (result_sec_id, _), status in results.items()
                        if result_sec_id == sec_id
                    )
                    for results in finding_maps
                )

            selected_finding_counts = finding_counts_for_section(selected_sec_id)
            summary_rows = []
            for sec_id, spine_count in spiny_sections:
                section_validated = sum(
                    1 for (sid, _), _ in validation_results.items()
                    if sid == sec_id
                )
                remaining = max(spine_count - section_validated, 0)
                complete = 'yes' if remaining == 0 else 'no'
                missing_status = normalize_missing_spines_status(
                    section_missing_results.get(sec_id)
                )
                missing_status_label = missing_status_display(
                    sec_id, missing_status
                )
                finding_counts = finding_counts_for_section(sec_id)
                summary_rows.append(
                    f'<tr><td>{sec_id}</td><td>{spine_count}</td>'
                    f'<td class="status-{complete}">{complete.title()}</td>'
                    f'<td>{remaining}</td>'
                    f'<td>{finding_counts[0]}</td><td>{finding_counts[1]}</td>'
                    f'<td>{finding_counts[2]}</td><td>{finding_counts[3]}</td>'
                    f'<td>{finding_counts[4]}</td>'
                    f'<td>{finding_counts[5]}</td>'
                    f'<td class="section-{missing_status.casefold().replace(" ", "-")}">{missing_status_label}</td></tr>'
                )

            html = f'''
            <style>
                .spine-analysis {{ font-family: sans-serif; width: 100%; }}
                .spine-analysis h4 {{ margin: 8px 0 5px; }}
                .spine-analysis table {{ border-collapse: collapse; width: 100%; }}
                .spine-analysis th, .spine-analysis td {{
                    border: 1px solid #d9d9d9; padding: 5px 8px;
                    text-align: left; white-space: nowrap;
                }}
                .spine-analysis th {{ background: #f2f2f2; font-weight: 600; }}
                .spine-analysis td:nth-child(2),
                .spine-analysis td:nth-child(4),
                .spine-analysis td:nth-child(5),
                .spine-analysis td:nth-child(6),
                .spine-analysis td:nth-child(7),
                .spine-analysis td:nth-child(8),
                .spine-analysis td:nth-child(9),
                .spine-analysis td:nth-child(10) {{ text-align: right; }}
                .spine-analysis .status-yes {{ color: #188038; font-weight: 600; }}
                .spine-analysis .status-no {{ color: #b06000; }}
                .spine-analysis .section-missing {{ color: #c5221f; font-weight: 600; }}
                .spine-analysis .section-no-missing {{ color: #188038; font-weight: 600; }}
                .spine-analysis .section-not-set {{ color: #777; }}
            </style>
            <div class="spine-analysis">
                <h4>Current Section Analysis</h4>
                <table>
                    <thead><tr><th>Section</th><th>Number Spines</th>
                    <th>Validated (Yes or No)</th>
                    <th>Remaining Spines to Validate</th>
                    <th>False Positives</th>
                    <th>Incomplete Spines</th>
                    <th>Falsely Extended Spines</th>
                    <th>Merged Spines</th>
                    <th>Split Spines</th>
                    <th>Valid Structure (Yes)</th>
                    <th>Missing Segmented Spines</th></tr></thead>
                    <tbody><tr><td>{selected_sec_id}</td>
                    <td>{selected_count}</td>
                    <td class="status-{selected_complete}">{selected_complete.title()}</td>
                    <td>{selected_remaining}</td>
                    <td>{selected_finding_counts[0]}</td>
                    <td>{selected_finding_counts[1]}</td>
                    <td>{selected_finding_counts[2]}</td>
                    <td>{selected_finding_counts[3]}</td>
                    <td>{selected_finding_counts[4]}</td>
                    <td>{selected_finding_counts[5]}</td>
                    <td class="section-{selected_missing.casefold().replace(' ', '-')}">{selected_missing_display}</td></tr></tbody>
                </table>
                <h4>Mesh Analysis Summary</h4>
                <table>
                    <thead><tr><th>Section</th><th>Number Spines</th>
                    <th>Validated (Yes or No)</th>
                    <th>Remaining Spines to Validate</th>
                    <th>False Positives</th>
                    <th>Incomplete Spines</th>
                    <th>Falsely Extended Spines</th>
                    <th>Merged Spines</th>
                    <th>Split Spines</th>
                    <th>Valid Structure (Yes)</th>
                    <th>Missing Segmented Spines</th></tr></thead>
                    <tbody>{''.join(summary_rows)}</tbody>
                </table>
            </div>
            '''
            display(HTML(html))


    # ============================================================
    # Navigation callbacks
    # ============================================================

    def on_next_section(_):
        if current_section_idx[0] >= len(spiny_sections) - 1:
            return
        idx = current_section_idx[0] + 1
        current_section_idx[0] = idx
        section_dropdown.value = idx
        start_load_section(idx)

    def on_prev_section(_):
        if current_section_idx[0] <= 0:
            return
        idx = current_section_idx[0] - 1
        current_section_idx[0] = idx
        section_dropdown.value = idx
        start_load_section(idx)

    def on_next_subsection(_):
        """Advance to the next 10 µm subsection in the current section."""
        sec_id = spiny_sections[current_section_idx[0]][0]
        subsection_count = len(section_subsections.get(sec_id, []))
        if (
            subsection_count == 0
            or current_subsection_idx[0] >= subsection_count - 1
        ):
            return
        select_subsection(current_subsection_idx[0] + 1)

    def on_prev_subsection(_):
        """Move to the previous 10 µm subsection in the current section."""
        sec_id = spiny_sections[current_section_idx[0]][0]
        subsection_count = len(section_subsections.get(sec_id, []))
        if subsection_count == 0 or current_subsection_idx[0] <= 0:
            return
        select_subsection(current_subsection_idx[0] - 1)

    def on_missing_subsection(_):
        """Increment the missing-spine count for the current subsection."""
        sec_id = spiny_sections[current_section_idx[0]][0]
        subsection_idx = current_subsection_idx[0]
        if subsection_idx >= len(section_subsections.get(sec_id, [])):
            return
        key = (sec_id, subsection_idx)
        subsection_missing_counts[key] = subsection_missing_counts.get(key, 0) + 1
        # Keep the existing qualitative section status consistent with a
        # positive subsection-level missing-spine observation.
        section_missing_results[sec_id] = MISSING_SPINES_MISSING
        schedule_validation_save()
        refresh_subsection_dropdown()
        refresh_section_dropdown()
        update_info()

    def on_next_spine(_, advance_section=True):
        if not current_spine_meshes_k3d:
            return
        if not spine_selected[0]:
            current_spine_idx[0] = 0
            highlight_current_spine()
            update_info()
            return
        sec_id, spine_count = spiny_sections[current_section_idx[0]]
        next_idx = get_first_unvalidated_spine_index(sec_id, current_spine_idx[0])
        if next_idx is not None:
            current_spine_idx[0] = next_idx
            highlight_current_spine()
            update_info()
            return
        if advance_section and section_is_fully_validated(sec_id, spine_count):
            on_next_section(None)

    def on_prev_spine(_):
        n = len(current_spine_meshes_k3d)
        if n == 0:
            return
        if not spine_selected[0]:
            current_spine_idx[0] = 0
        elif current_spine_idx[0] <= 0:
            return
        else:
            current_spine_idx[0] -= 1
        highlight_current_spine()
        update_info()


    def on_xy(_):
        set_projection([0, 0, 1], [0, 1, 0])

    def on_negative_xy(_):
        set_projection([0, 0, -1], [0, 1, 0])

    def on_xz(_):
        set_projection([0, 1, 0], [0, 0, 1])

    def on_negative_xz(_):
        set_projection([0, -1, 0], [0, 0, 1])

    def on_yz(_):
        set_projection([1, 0, 0], [0, 1, 0])

    def on_negative_yz(_):
        set_projection([-1, 0, 0], [0, 1, 0])

    background_is_black = [False]

    def on_toggle_background(_):
        """Switch the K3D plot background with minimal trait updates."""
        background_is_black[0] = not background_is_black[0]
        centerline_color = (
            SECTION_CENTERLINE_HOT_COLOR
            if background_is_black[0]
            else SECTION_CENTERLINE_COLOR
        )

        # Keep the base morphology lines at their existing gray. Recoloring
        # every section line here caused one K3D update per section and delayed
        # the active subsection refresh on large morphologies.
        plot.background_color = (
            0x000000 if background_is_black[0] else 0xFFFFFF
        )
        if section_centerline_k3d[0].visible:
            section_centerline_k3d[0].color = centerline_color
        if subsection_centerline_overlay.visible:
            subsection_centerline_overlay.color = SUBSECTION_CENTERLINE_COLOR

        btn_toggle_background.description = (
            'White Background' if background_is_black[0]
            else 'Black Background'
        )


    # ============================================================
    # Section-level segmented spine callbacks
    # ============================================================

    def set_section_missing_status(status):
        sec_id = spiny_sections[current_section_idx[0]][0]
        section_missing_results[sec_id] = status
        schedule_validation_save()
        refresh_section_dropdown()
        focus_first_spine()

    def on_confirm_missing_spines(_):
        """Save missing-spine status and advance subsection or spine navigation."""
        sec_id = spiny_sections[current_section_idx[0]][0]
        current_subsections = section_subsections.get(sec_id, [])
        subsection_missing_total = sum(
            subsection_missing_counts.get((sec_id, subsection['index']), 0)
            for subsection in current_subsections
        )
        section_missing_results[sec_id] = (
            MISSING_SPINES_MISSING
            if subsection_missing_total > 0
            else MISSING_SPINES_NO_MISSING
        )
        if 0 <= current_subsection_idx[0] < len(current_subsections):
            subsection_done_results.add((sec_id, current_subsection_idx[0]))
        schedule_validation_save()
        refresh_section_dropdown()
        if (
            current_subsections
            and current_subsection_idx[0] >= len(current_subsections) - 1
        ):
            focus_first_spine()
        else:
            on_next_subsection(None)


    # ============================================================
    # Validation callbacks
    # ============================================================

    def set_spine_analysis_status(results, status, advance=False):
        """Set one analysis value for the current spine and persist it."""
        sec_id = spiny_sections[current_section_idx[0]][0]
        key = (sec_id, current_spine_idx[0])
        results[key] = status
        schedule_validation_save()
        refresh_section_dropdown()
        update_info()
        if advance:
            on_next_spine(None)


    def on_correct_type_yes(_):
        set_spine_analysis_status(correct_type_results, 'yes')


    def on_correct_type_no(_):
        set_spine_analysis_status(correct_type_results, 'no')


    def on_false_positive_yes(_):
        set_spine_analysis_status(false_positive_results, 'yes')


    def on_false_positive_no(_):
        set_spine_analysis_status(false_positive_results, 'no')


    def on_incomplete_spines_yes(_):
        set_spine_analysis_status(incomplete_spine_results, 'yes')


    def on_incomplete_spines_no(_):
        set_spine_analysis_status(incomplete_spine_results, 'no')


    def on_false_positive_quality_yes(_):
        set_spine_analysis_status(false_positive_quality_results, 'yes')


    def on_false_positive_quality_no(_):
        set_spine_analysis_status(false_positive_quality_results, 'no')


    def on_merged_spine_yes(_):
        set_spine_analysis_status(merged_spine_results, 'yes')


    def on_merged_spine_no(_):
        set_spine_analysis_status(merged_spine_results, 'no')


    def on_split_spine_yes(_):
        set_spine_analysis_status(split_spine_results, 'yes')


    def on_split_spine_no(_):
        set_spine_analysis_status(split_spine_results, 'no')


    def on_valid_structure_yes(_):
        set_spine_analysis_status(valid_structure_results, 'yes')


    def on_valid_structure_no(_):
        set_spine_analysis_status(valid_structure_results, 'no')


    def on_spine_done_next(_):
        key = (
            spiny_sections[current_section_idx[0]][0],
            current_spine_idx[0],
        )
        issue_results = (
            false_positive_results,
            incomplete_spine_results,
            false_positive_quality_results,
            merged_spine_results,
            split_spine_results,
        )
        has_defect = any(
            results.get(key) == 'yes'
            for results in issue_results
        )
        set_spine_analysis_status(
            validation_results,
            'invalid' if has_defect else 'valid',
        )
        on_next_spine(None)



    # ============================================================
    # Widgets
    # ============================================================

    navigation_button_width = '124px'
    navigation_button_spacing = '8px'
    status_button_width = '124px'
    btn_prev_sec = widgets.Button(
        description='Prev Section',
        icon='arrow-left',
        layout=widgets.Layout(
            width=navigation_button_width,
            min_width=navigation_button_width,
            max_width=navigation_button_width,
        ),
    )
    btn_next_sec = widgets.Button(
        description='Next Section',
        icon='arrow-right',
        layout=widgets.Layout(
            width=navigation_button_width,
            min_width=navigation_button_width,
            max_width=navigation_button_width,
            margin=f'0 0 0 {navigation_button_spacing}',
        ),
    )
    btn_prev_spine = widgets.Button(
        description='Prev Spine',
        icon='chevron-left',
        layout=widgets.Layout(
            width=navigation_button_width,
            min_width=navigation_button_width,
            max_width=navigation_button_width,
        ),
    )
    btn_next_spine = widgets.Button(
        description='Next Spine',
        icon='chevron-right',
        layout=widgets.Layout(
            width=navigation_button_width,
            min_width=navigation_button_width,
            max_width=navigation_button_width,
            margin=f'0 0 0 {navigation_button_spacing}',
        ),
    )

    btn_correct_type_yes = widgets.Button(
        description='Yes', button_style='warning', disabled=True
    )
    btn_correct_type_no = widgets.Button(
        description='No', button_style='success', disabled=True
    )
    btn_false_positive_yes = widgets.Button(
        description='Yes', button_style='warning', disabled=True
    )
    btn_false_positive_no = widgets.Button(
        description='No', button_style='success', disabled=True
    )
    btn_incomplete_spines_yes = widgets.Button(
        description='Yes', button_style='warning', disabled=True
    )
    btn_incomplete_spines_no = widgets.Button(
        description='No', button_style='success', disabled=True
    )
    btn_false_positive_quality_yes = widgets.Button(
        description='Yes', button_style='warning', disabled=True
    )
    btn_false_positive_quality_no = widgets.Button(
        description='No', button_style='success', disabled=True
    )
    btn_merged_spine_yes = widgets.Button(
        description='Yes', button_style='warning', disabled=True
    )
    btn_merged_spine_no = widgets.Button(
        description='No', button_style='success', disabled=True
    )
    btn_split_spine_yes = widgets.Button(
        description='Yes', button_style='warning', disabled=True
    )
    btn_split_spine_no = widgets.Button(
        description='No', button_style='success', disabled=True
    )
    btn_spine_done_next = widgets.Button(
        description='Spine Done, Next',
        button_style='warning',
        disabled=True,
        layout=widgets.Layout(
            width='258px',
            min_width='258px',
            max_width='258px',
        ),
    )
    btn_show_spine_structure = widgets.Button(
        description='Show Spine Structure',
        button_style='info',
        disabled=True,
        layout=widgets.Layout(
            width='258px',
            min_width='258px',
            max_width='258px',
        ),
    )
    btn_valid_structure_yes = widgets.Button(
        description='Yes', button_style='warning', disabled=True
    )
    btn_valid_structure_no = widgets.Button(
        description='No', button_style='success', disabled=True
    )
    btn_screenshot = widgets.Button(description='ScreenShot', icon='camera')
    btn_register_assessment = widgets.Button(
        description='Register', icon='upload'
    )
    btn_generate_report = widgets.Button(
        description='Generate Report', icon='file-text'
    )
    registration_status = widgets.HTML(value='')
    report_status = widgets.HTML(value='')

    btn_confirm_missing = widgets.Button(
        description='Subsection Done, Next',
        button_style='warning',
        layout=widgets.Layout(
            width='258px',
            min_width='258px',
            max_width='258px',
        ),
    )
    btn_prev_subsection = widgets.Button(
        description='Prev Subsection',
        icon='arrow-left',
        layout=widgets.Layout(
            width=navigation_button_width,
            min_width=navigation_button_width,
            max_width=navigation_button_width,
        ),
    )
    btn_next_subsection = widgets.Button(
        description='Next Subsection',
        icon='arrow-right',
        layout=widgets.Layout(
            width=navigation_button_width,
            min_width=navigation_button_width,
            max_width=navigation_button_width,
            margin=f'0 0 0 {navigation_button_spacing}',
        ),
    )
    btn_missing_subsection = widgets.Button(
        description='Missing +1',
        button_style='danger',
        layout=widgets.Layout(
            width='100px',
            min_width='100px',
            max_width='100px',
            margin='0 0 0 12px',
        ),
    )

    btn_xy = widgets.Button(description='XY', layout=widgets.Layout(width='90px'))
    btn_negative_xy = widgets.Button(description='-XY', layout=widgets.Layout(width='90px'))
    btn_xz = widgets.Button(description='XZ', layout=widgets.Layout(width='90px'))
    btn_negative_xz = widgets.Button(description='-XZ', layout=widgets.Layout(width='90px'))
    btn_yz = widgets.Button(description='YZ', layout=widgets.Layout(width='90px'))
    btn_negative_yz = widgets.Button(description='-YZ', layout=widgets.Layout(width='90px'))
    btn_toggle_background = widgets.Button(
        description='Black Background',
        icon='adjust',
        tooltip='Switch the plot background between white and black',
    )

    # Standard 3D-software axis colors: X red, Y green, Z blue.
    AXIS_COLORS = {'X': '#dc2626', 'Y': '#16a34a', 'Z': '#2563eb'}
    # For each plane, the (horizontal axis, vertical axis, viewing/normal axis).
    PLANE_AXES = {
        'XY': ('X', 'Y', 'Z'),
        'XZ': ('X', 'Z', 'Y'),
        'YZ': ('Y', 'Z', 'X'),
    }

    def projection_svg(plane, direction):
        """Draw a small axis-triad icon: two in-plane axes plus a depth marker.

        The depth marker follows the standard engineering drawing convention:
        a filled dot means the viewing axis points out of the page toward the
        viewer ('+'), and a circled cross means it points into the page,
        away from the viewer ('-').
        """
        horizontal_axis, vertical_axis, normal_axis = PLANE_AXES[plane]
        toward_viewer = direction == '+'
        sign = '+' if toward_viewer else '\u2212'
        horizontal_color = AXIS_COLORS[horizontal_axis]
        vertical_color = AXIS_COLORS[vertical_axis]
        normal_color = AXIS_COLORS[normal_axis]

        origin = (22, 64)
        horizontal_end = (70, 64)
        vertical_end = (22, 14)
        normal_radius = 7
        marker_prefix = f'{plane.lower()}-{direction}'
        h_marker_id = f'arrow-h-{marker_prefix}'
        v_marker_id = f'arrow-v-{marker_prefix}'

        if toward_viewer:
            normal_glyph = (
                f"<circle cx='{origin[0]}' cy='{origin[1]}' r='{normal_radius}' "
                f"fill='{normal_color}'/>"
            )
        else:
            offset = normal_radius * 0.6
            cx, cy = origin
            normal_glyph = (
                f"<circle cx='{cx}' cy='{cy}' r='{normal_radius}' fill='none' "
                f"stroke='{normal_color}' stroke-width='1.6'/>"
                f"<line x1='{cx - offset}' y1='{cy - offset}' "
                f"x2='{cx + offset}' y2='{cy + offset}' "
                f"stroke='{normal_color}' stroke-width='1.6'/>"
                f"<line x1='{cx - offset}' y1='{cy + offset}' "
                f"x2='{cx + offset}' y2='{cy - offset}' "
                f"stroke='{normal_color}' stroke-width='1.6'/>"
            )

        return (
            "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 84 84' "
            "width='84' height='84' role='img' "
            f"aria-label='View along {sign}{normal_axis}, "
            f"looking at the {horizontal_axis}{vertical_axis} plane'>"
            '<defs>'
            f"<marker id='{h_marker_id}' markerWidth='7' markerHeight='7' "
            "refX='6' refY='3.5' orient='auto'>"
            f"<path d='M0,0 L7,3.5 L0,7 Z' fill='{horizontal_color}'/></marker>"
            f"<marker id='{v_marker_id}' markerWidth='7' markerHeight='7' "
            "refX='6' refY='3.5' orient='auto'>"
            f"<path d='M0,0 L7,3.5 L0,7 Z' fill='{vertical_color}'/></marker>"
            '</defs>'
            f"<line x1='{origin[0]}' y1='{origin[1]}' "
            f"x2='{horizontal_end[0]}' y2='{horizontal_end[1]}' "
            f"stroke='{horizontal_color}' stroke-width='2.4' "
            f"marker-end='url(#{h_marker_id})'/>"
            f"<line x1='{origin[0]}' y1='{origin[1]}' "
            f"x2='{vertical_end[0]}' y2='{vertical_end[1]}' "
            f"stroke='{vertical_color}' stroke-width='2.4' "
            f"marker-end='url(#{v_marker_id})'/>"
            f"<text x='{horizontal_end[0] + 6}' y='{horizontal_end[1] + 4}' "
            f"font-family='sans-serif' font-size='13' font-weight='700' "
            f"fill='{horizontal_color}'>{horizontal_axis}</text>"
            f"<text x='{vertical_end[0]}' y='{vertical_end[1] - 5}' "
            "text-anchor='middle' font-family='sans-serif' font-size='13' "
            f"font-weight='700' fill='{vertical_color}'>{vertical_axis}</text>"
            f"{normal_glyph}"
            f"<text x='42' y='80' text-anchor='middle' font-family='sans-serif' "
            f"font-size='10' font-weight='600' fill='{normal_color}'>"
            f"{sign}{normal_axis} axis (view)</text>"
            '</svg>'
        )

    def projection_control(button, plane, direction):
        """Pair a projection button with its inline SVG orientation icon."""
        icon = widgets.HTML(
            value=projection_svg(plane, direction),
            layout=widgets.Layout(width='84px', height='84px'),
        )
        return widgets.VBox(
            [icon, button],
            layout=widgets.Layout(width='96px', align_items='center', margin='0 4px 4px 0'),
        )

    info_label = widgets.HTML(value='<i>Loading...</i>')

    stats_output = widgets.Output(
        layout=widgets.Layout(border='1px solid #ddd', padding='8px', width='100%')
    )
    # Section dropdown
    def build_section_dropdown_options():
        """Build dropdown options with validation status."""
        options = []
        for idx, (sec_id, spine_count) in enumerate(spiny_sections):
            # Count how many spines in this section have been validated
            n_validated = sum(
                1 for (sid, _), _ in validation_results.items() if sid == sec_id
            )
            if n_validated == 0:
                status = ''
            elif n_validated >= spine_count:
                status = ' [Done]'
            else:
                status = f' [{n_validated}/{spine_count}]'
            label = f'Section {sec_id} ({spine_count} spines){status}'
            options.append((label, idx))
        return options

    section_dropdown = widgets.Dropdown(
        options=build_section_dropdown_options(),
        value=0 if spiny_sections else None,
        description='',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='258px')
    )

    def build_subsection_dropdown_options():
        """Build labels for the current section's 10 µm subsections."""
        sec_id = spiny_sections[current_section_idx[0]][0]
        options = []
        for subsection in section_subsections.get(sec_id, []):
            subsection_idx = subsection['index']
            missing_count = subsection_missing_counts.get(
                (sec_id, subsection_idx),
                0,
            )
            subsection_status = (
                '[Done]'
                if (sec_id, subsection_idx) in subsection_done_results
                else '[Not Set]'
            )
            label = (
                f"Subsection {subsection_idx + 1} "
                f"(Missing : {missing_count}) {subsection_status}"
            )
            options.append((label, subsection_idx))
        return options

    subsection_dropdown = widgets.Dropdown(
        options=build_subsection_dropdown_options(),
        value=0 if section_subsections.get(spiny_sections[0][0]) else None,
        description='',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='258px'),
    )

    def refresh_subsection_dropdown():
        """Refresh subsection labels without triggering a new selection."""
        options = build_subsection_dropdown_options()
        selected_idx = current_subsection_idx[0]
        valid_indices = [value for _, value in options]
        updating_subsection_dropdown[0] = True
        try:
            subsection_dropdown.options = options
            subsection_dropdown.value = (
                selected_idx if selected_idx in valid_indices else None
            )
        finally:
            updating_subsection_dropdown[0] = False


    def refresh_section_dropdown():
        """Refresh labels without triggering a section reload."""
        selected_idx = current_section_idx[0]
        updating_section_dropdown[0] = True
        try:
            section_dropdown.options = build_section_dropdown_options()
            section_dropdown.value = selected_idx
        finally:
            updating_section_dropdown[0] = False


    def build_spine_dropdown_options():
        """Build labels with local/global IDs, H5 type, and validity status."""
        sec_id = spiny_sections[current_section_idx[0]][0]
        options = []
        for spine_idx in range(len(current_spine_data)):
            status_label = validity_status((sec_id, spine_idx)).title()
            global_spine_id = (
                current_spine_global_ids[spine_idx]
                if spine_idx < len(current_spine_global_ids)
                else None
            )
            if global_spine_id is None:
                spine_type = 'Unknown'
                global_id_label = 'Unknown'
            else:
                try:
                    raw_spine_type = morphology.spines.spine_type(
                        int(global_spine_id)
                    )
                    raw_spine_type = getattr(raw_spine_type, 'value', raw_spine_type)
                    spine_type = str(raw_spine_type).strip()
                    if spine_type.startswith('SpineType.'):
                        spine_type = spine_type.rsplit('.', 1)[-1]
                    spine_type = spine_type.replace('_', ' ').title() or 'Unknown'
                except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                    spine_type = 'Unknown'
                global_id_label = str(global_spine_id)
            options.append(
                (
                    f'Spine {spine_idx} ({global_id_label}, {spine_type}) '
                    f'[{status_label}]',
                    spine_idx,
                )
            )
        return options


    spine_dropdown = widgets.Dropdown(
        options=[],
        value=None,
        description='',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='260px')
    )


    def refresh_spine_dropdown():
        """Refresh spine labels and preserve the selected spine when possible."""
        selected_idx = current_spine_idx[0]
        updating_spine_dropdown[0] = True
        try:
            options = build_spine_dropdown_options()
            spine_dropdown.options = options
            values = [value for _, value in options]
            selected_value = (
                selected_idx
                if spine_selected[0] and selected_idx in values
                else None
            )
            spine_dropdown.value = selected_value
        finally:
            updating_spine_dropdown[0] = False


    def on_spine_dropdown_change(change):
        """Handle spine dropdown selection."""
        if change['name'] != 'value' or updating_spine_dropdown[0]:
            return
        idx = change['new']
        if idx is not None and (idx != current_spine_idx[0] or not spine_selected[0]):
            current_spine_idx[0] = idx
            highlight_current_spine()
            update_info()


    def select_subsection(subsection_idx, focus_camera=True):
        """Select, highlight, and focus one subsection in the current section."""
        sec_id = spiny_sections[current_section_idx[0]][0]
        if not 0 <= subsection_idx < len(section_subsections.get(sec_id, [])):
            return
        clear_spine_structure()
        current_subsection_idx[0] = subsection_idx
        spine_selected[0] = False
        hide_bbox()
        update_spine_colors()
        update_spine_analysis_button_state()
        show_subsection_centerline(sec_id, subsection_idx, focus_camera=focus_camera)
        update_info()


    def on_subsection_dropdown_change(change):
        """Handle selection of a subsection from the dropdown."""
        if change['name'] != 'value' or updating_subsection_dropdown[0]:
            return
        subsection_idx = change['new']
        if subsection_idx is not None:
            select_subsection(subsection_idx)


    def on_section_dropdown_change(change):
        """Handle section dropdown selection."""
        if change['name'] != 'value' or updating_section_dropdown[0]:
            return
        idx = change['new']
        if idx is not None and idx != current_section_idx[0]:
            current_section_idx[0] = idx
            start_load_section(idx)


    subsection_dropdown.observe(on_subsection_dropdown_change, names='value')
    section_dropdown.observe(on_section_dropdown_change, names='value')
    spine_dropdown.observe(on_spine_dropdown_change, names='value')

    # Connect callbacks
    btn_prev_sec.on_click(on_prev_section)
    btn_next_sec.on_click(on_next_section)
    btn_prev_subsection.on_click(on_prev_subsection)
    btn_next_subsection.on_click(on_next_subsection)
    btn_missing_subsection.on_click(on_missing_subsection)
    btn_prev_spine.on_click(on_prev_spine)
    btn_next_spine.on_click(on_next_spine)
    btn_correct_type_yes.on_click(on_correct_type_yes)
    btn_correct_type_no.on_click(on_correct_type_no)
    btn_false_positive_yes.on_click(on_false_positive_yes)
    btn_false_positive_no.on_click(on_false_positive_no)
    btn_incomplete_spines_yes.on_click(on_incomplete_spines_yes)
    btn_incomplete_spines_no.on_click(on_incomplete_spines_no)
    btn_false_positive_quality_yes.on_click(on_false_positive_quality_yes)
    btn_false_positive_quality_no.on_click(on_false_positive_quality_no)
    btn_merged_spine_yes.on_click(on_merged_spine_yes)
    btn_merged_spine_no.on_click(on_merged_spine_no)
    btn_split_spine_yes.on_click(on_split_spine_yes)
    btn_split_spine_no.on_click(on_split_spine_no)
    btn_spine_done_next.on_click(on_spine_done_next)
    btn_show_spine_structure.on_click(on_toggle_spine_structure)
    btn_valid_structure_yes.on_click(on_valid_structure_yes)
    btn_valid_structure_no.on_click(on_valid_structure_no)
    btn_screenshot.on_click(take_screenshot)
    btn_register_assessment.on_click(on_register_assessment)
    btn_generate_report.on_click(on_generate_report)
    plot.observe(on_screenshot_ready, names='screenshot')
    btn_confirm_missing.on_click(on_confirm_missing_spines)
    btn_xy.on_click(on_xy)
    btn_negative_xy.on_click(on_negative_xy)
    btn_xz.on_click(on_xz)
    btn_negative_xz.on_click(on_negative_xz)
    btn_yz.on_click(on_yz)
    btn_negative_yz.on_click(on_negative_yz)
    btn_toggle_background.on_click(on_toggle_background)


    # ============================================================
    # Layout
    # ============================================================

    row_layout = widgets.Layout(
        display='flex', flex_flow='row wrap', align_items='center', width='100%'
    )
    navigation_button_row_layout = widgets.Layout(
        display='flex',
        flex_flow='row nowrap',
        align_items='center',
        justify_content='flex-end',
        width='260px',
        min_width='260px',
        max_width='260px',
        gap='0px',
    )
    group_layout = widgets.Layout(width='100%', margin='2px 0 4px 0')

    analysis_labels = [
        'Correct Type',
        'False Positive',
        'Incomplete Spine',
        'Falsely Extended',
        'Merged Spine',
        'Split Spine',
        'Valid Structure',
    ]
    analysis_tooltips = {
        'Correct Type': (
            'Does the spine type identified automatically correspond to what do you know?'
        ),
        'False Positive': (
            'The selected object is not a real spine and is likely a segmentation artifact.'
        ),
        'Incomplete Spine': (
            'The spine is present, but its segmented geometry is incomplete or missing part of the spine.'
        ),
        'Falsely Extended': (
            'The segmented geometry extends beyond the actual spine or includes unrelated geometry.'
        ),
        'Merged Spine': (
            'Two or more distinct spines have been incorrectly combined into one segmented object.'
        ),
        'Split Spine': (
            'One actual spine has been incorrectly divided into multiple segmented objects.'
        ),
        'Valid Structure': (
            'Does the displayed rose head and green neck structure match the spine morphology?'
        ),
    }
    # Approximate rendered label width from the longest label, with room for the help icon.
    analysis_label_width = f'{max(map(len, analysis_labels)) * 8 + 4}px'
    analysis_label_layout = widgets.Layout(
        width=analysis_label_width,
        min_width=analysis_label_width,
        max_width=analysis_label_width,
        flex=f'0 0 {analysis_label_width}',
        white_space='nowrap',
    )
    analysis_button_width = '58px'

    def analysis_row(label, yes_button, no_button):
        """Build one compact, aligned single-line analysis row."""
        for button in (yes_button, no_button):
            button.layout = widgets.Layout(
                width=analysis_button_width,
                min_width=analysis_button_width,
                max_width=analysis_button_width,
                padding='0 4px',
            )

        button_group = widgets.HBox(
            [yes_button, no_button],
            layout=widgets.Layout(
                display='flex',
                flex_flow='row nowrap',
                flex='0 0 auto',
                width='auto',
                margin='0',
                gap='4px',
            ),
        )
        label_html = (
            f'<b>{label}</b> '
            f'<span title="{analysis_tooltips[label]}" '
            'style="cursor:help; color:#5f6368;">&#9432;</span>'
        )
        return widgets.HBox(
            [widgets.HTML(value=label_html, layout=analysis_label_layout), button_group],
            layout=widgets.Layout(
                display='flex',
                flex_flow='row nowrap',
                align_items='center',
                justify_content='space-between',
                width='260px',
                min_width='260px',
                max_width='260px',
                margin='2px 0',
            ),
        )

    spine_analysis_controls = widgets.VBox([
        analysis_row(
            'Correct Type',
            btn_correct_type_yes,
            btn_correct_type_no,
        ),
        analysis_row(
            'False Positive',
            btn_false_positive_yes,
            btn_false_positive_no,
        ),
        analysis_row(
            'Incomplete Spine',
            btn_incomplete_spines_yes,
            btn_incomplete_spines_no,
        ),
        analysis_row(
            'Falsely Extended',
            btn_false_positive_quality_yes,
            btn_false_positive_quality_no,
        ),
        analysis_row(
            'Merged Spine',
            btn_merged_spine_yes,
            btn_merged_spine_no,
        ),
        analysis_row(
            'Split Spine',
            btn_split_spine_yes,
            btn_split_spine_no,
        ),
        widgets.HTML(
            value='<div style="border-top:1px solid #d9d9d9; margin:6px 0;"></div>',
            layout=widgets.Layout(width='260px'),
        ),
        btn_show_spine_structure,
        analysis_row(
            'Valid Structure',
            btn_valid_structure_yes,
            btn_valid_structure_no,
        ),
        btn_spine_done_next,
    ], layout=widgets.Layout(width='100%'))

    # The current ipywidgets theme uses a 30 px default button height. Set every
    # button to 27 px (90% of the default) while preserving its existing width,
    # style, and other layout properties.
    button_height = '25px'
    all_buttons = (
        btn_prev_sec,
        btn_next_sec,
        btn_prev_spine,
        btn_next_spine,
        btn_correct_type_yes,
        btn_correct_type_no,
        btn_false_positive_yes,
        btn_false_positive_no,
        btn_incomplete_spines_yes,
        btn_incomplete_spines_no,
        btn_false_positive_quality_yes,
        btn_false_positive_quality_no,
        btn_merged_spine_yes,
        btn_merged_spine_no,
        btn_split_spine_yes,
        btn_split_spine_no,
        btn_show_spine_structure,
        btn_valid_structure_yes,
        btn_valid_structure_no,
        btn_spine_done_next,
        btn_screenshot,
        btn_register_assessment,
        btn_generate_report,
        btn_confirm_missing,
        btn_prev_subsection,
        btn_next_subsection,
        btn_missing_subsection,
        btn_xy,
        btn_negative_xy,
        btn_xz,
        btn_negative_xz,
        btn_yz,
        btn_negative_yz,
        btn_toggle_background,
    )
    for button in all_buttons:
        button.layout.height = button_height

    section_nav = widgets.HBox(
        [btn_prev_sec, btn_next_sec], layout=navigation_button_row_layout
    )
    spine_nav_buttons = widgets.HBox(
        [btn_prev_spine, btn_next_spine],
        layout=navigation_button_row_layout,
    )
    spine_nav = widgets.VBox(
        [spine_dropdown, spine_nav_buttons],
        layout=widgets.Layout(width='100%', align_items='flex-start'),
    )
    section_status_btns = widgets.HBox(
        [btn_confirm_missing],
        layout=widgets.Layout(
            width='100%',
            margin='20px 0 0 0',
        ),
    )

    projection_btns = widgets.HBox(
        [
            widgets.VBox(
                [
                    projection_control(btn_xy, 'XY', '+'),
                    projection_control(btn_negative_xy, 'XY', '-'),
                ],
                layout=widgets.Layout(align_items='center'),
            ),
            widgets.VBox(
                [
                    projection_control(btn_xz, 'XZ', '+'),
                    projection_control(btn_negative_xz, 'XZ', '-'),
                ],
                layout=widgets.Layout(align_items='center'),
            ),
            widgets.VBox(
                [
                    projection_control(btn_yz, 'YZ', '+'),
                    projection_control(btn_negative_yz, 'YZ', '-'),
                ],
                layout=widgets.Layout(align_items='center'),
            ),
        ],
        layout=widgets.Layout(
            display='flex', flex_flow='row nowrap', align_items='flex-start',
            justify_content='flex-end', width='100%'
        ),
    )

    section_selection = widgets.HBox(
        [section_dropdown, section_nav], layout=row_layout
    )
    subsection_nav = widgets.VBox(
        [
            subsection_dropdown,
            widgets.HBox(
                [btn_prev_subsection, btn_next_subsection],
                layout=navigation_button_row_layout,
            ),
            widgets.HBox(
                [
                    widgets.HTML(
                        value='<b>Missing Spine Detected</b>',
                        layout=widgets.Layout(
                            flex='0 0 auto',
                            min_width='0',
                            white_space='nowrap',
                        ),
                    ),
                    btn_missing_subsection,
                ],
                layout=widgets.Layout(
                    display='flex',
                    flex_flow='row nowrap',
                    align_items='center',
                    justify_content='flex-start',
                    width='100%',
                    margin='20px 0 0 0',
                ),
            ),
        ],
        layout=widgets.Layout(width='100%', align_items='flex-start'),
    )
    spine_validation = widgets.VBox(
        [spine_nav, spine_analysis_controls],
        layout=widgets.Layout(width='100%'),
    )

    section_navigation_column = widgets.VBox([
        widgets.HTML(value=(
            '<b>Section Navigation</b> '
            '<span title="Select a section to analyze" '
            'style="cursor:help; color:#5f6368;">&#9432;</span>'
        )),
        section_selection,
        widgets.HTML(
            value=(
                '<b>Subsection Navigation (10 µm)</b> '
                '<span title="Review the current subsection, click Missing Spine (+1) '
                'once for each missing spine, and use Next Subsection to continue." '
                'style="cursor:help; color:#5f6368;">&#9432;</span>'
            ),
            layout=widgets.Layout(margin='6px 0 0 0'),
        ),
        subsection_nav,
        section_status_btns,
    ], layout=widgets.Layout(width='33.333%', padding='0 8px 0 0', box_sizing='border-box'))

    spine_navigation_column = widgets.VBox([
        widgets.HTML(value=(
            '<b>Spine Navigation</b> '
            '<span title="Select a spine to analyze" '
            'style="cursor:help; color:#5f6368;">&#9432;</span>'
        )),
        spine_validation,
    ], layout=widgets.Layout(width='33.333%', padding='0 8px', box_sizing='border-box'))

    visual_verification_section = widgets.VBox([
        widgets.HTML(value=(
            '<b>Visual Verification</b> '
            '<span title="Use the ScreenShot button to save a visual record of the current view. '
            'When only a section is selected, the image captures the section geometry and all '
            'of its spines. When an individual spine is selected, the image captures that spine '
            'for visual verification and assessment." '
            'style="cursor:help; color:#5f6368;">&#9432;</span>'
        )),
        widgets.HBox([btn_screenshot], layout=row_layout),
    ], layout=widgets.Layout(
        width='100%', padding='8px 0 0 0', box_sizing='border-box'
    ))

    registration_controls = widgets.VBox([
        widgets.HTML(value=(
            '<b>Assessment</b> '
            '<span title="Click Register to generate the section-validity figure and register the assessment results." '
            'style="cursor:help; color:#5f6368;">&#9432;</span>'
        )),
        widgets.HBox(
            [btn_generate_report, btn_register_assessment],
            layout=row_layout,
        ),
        report_status,
        registration_status,
    ], layout=widgets.Layout(width='100%', margin='14px 0 0 0'))

    spine_status_column = widgets.VBox([
        info_label,
        visual_verification_section,
        registration_controls,
    ], layout=widgets.Layout(width='33.333%', padding='0 0 0 8px', box_sizing='border-box'))

    navigation_columns = widgets.HBox(
        [section_navigation_column, spine_navigation_column, spine_status_column],
        layout=widgets.Layout(width='100%', align_items='flex-start'),
    )
    section_controls = widgets.VBox(
        [navigation_columns],
        layout=widgets.Layout(width='70%', margin='2px 0 4px 0'),
    )

    projection_controls = widgets.VBox([
        widgets.HTML(value='<b>Projection Views</b>'),
        projection_btns,
        btn_toggle_background,
    ], layout=widgets.Layout(
        width='30%', align_self='flex-start', align_items='flex-end'
    ))

    controls = widgets.HBox(
        [section_controls, projection_controls],
        layout=widgets.Layout(
            width='100%', justify_content='space-between', align_items='flex-start'
        ),
    )

    # Set initial camera to Z+
    plot.camera_auto_fit = False
    plot.camera = [
        float(morph_center[0]), float(morph_center[1]), float(morph_center[2] + morph_radius * 2.5),
        float(morph_center[0]), float(morph_center[1]), float(morph_center[2]),
        0.0, 1.0, 0.0,
    ]

    # Migrate an existing legacy CSV before handing control to the async UI.
    # This keeps headless notebook execution and interactive startup consistent.
    if validation_csv_path.is_file():
        try:
            (
                loaded_spines,
                loaded_correct_type,
                loaded_valid_structure,
                loaded_false_positives,
                loaded_incomplete_spines,
                loaded_false_positive_quality,
                loaded_merged_spines,
                loaded_split_spines,
                loaded_sections,
                loaded_subsection_counts,
                loaded_subsection_done,
            ) = read_validation_csv(validation_csv_path)
            validation_results.update(loaded_spines)
            correct_type_results.update(loaded_correct_type)
            valid_structure_results.update(loaded_valid_structure)
            false_positive_results.update(loaded_false_positives)
            incomplete_spine_results.update(loaded_incomplete_spines)
            false_positive_quality_results.update(loaded_false_positive_quality)
            merged_spine_results.update(loaded_merged_spines)
            split_spine_results.update(loaded_split_spines)
            section_missing_results.update(loaded_sections)
            subsection_missing_counts.update(loaded_subsection_counts)
            subsection_done_results.update(loaded_subsection_done)
            write_validation_csv(
                validation_csv_path,
                snapshot_validation_state(),
            )
        except Exception as exc:
            print(f'Could not migrate validation CSV: {exc}')

    # Display immediately, then restore saved labels before loading the first section.
    #display(widgets.VBox([controls, plot, stats_output]))
    display(controls)
    plot.display()
    # Install the scale bar after PlotView creates the WebGL canvas.
    plot.additional_js_code = scale_bar_js()
    display(stats_output)

    restore_task = asyncio.create_task(restore_and_start())
    restore_task.add_done_callback(report_validation_restore_error)
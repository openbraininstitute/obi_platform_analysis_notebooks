"""Widget layout and presentation logic for the spine proofreading interface.

The data model and morphology/subsection helpers live in
:mod:`proofreading_functions`. This module owns the ipywidgets layout, K3D
viewport, event handlers, rendering, and notebook-facing preview functions.
"""

import asyncio
import base64
import colorsys
import getpass
import html
import importlib
import os
import re
from pathlib import Path

import ipywidgets as widgets
import k3d
import numpy as np
from IPython.display import HTML, clear_output, display

from morph_spines_visualizer.core import data_loading, geometry

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - optional acceleration
    cKDTree = None

try:  # Package import, e.g. ``from . import proofreading_ui``.
    from .proofreading_functions import (
        ALL_ANSWER_KEYS,
        ANALYSIS_FIELDS,
        DesignState,
        ISSUE_FIELDS,
        SECTION_CSV_FIELDS,
        VALIDITY_FIELDS,
        VALID_STRUCTURE_FIELD,
        derive_spine_validity,
        save_validation_state,
        section_summary_rows,
    )
except ImportError:  # Notebook import after adding ``lib`` to sys.path.
    from proofreading_functions import (
        ALL_ANSWER_KEYS,
        ANALYSIS_FIELDS,
        DesignState,
        ISSUE_FIELDS,
        SECTION_CSV_FIELDS,
        VALIDITY_FIELDS,
        VALID_STRUCTURE_FIELD,
        derive_spine_validity,
        save_validation_state,
        section_summary_rows,
    )


FONT_DIRECTORY = Path(__file__).resolve().parent / 'fonts' / 'archivo'

PROJECTION_CELLS = [
    ('xy', 'XY', '+Z AXIS'),
    ('xz', 'XZ', '+Y AXIS'),
    ('yz', 'YZ', '+X AXIS'),
    ('-xy', '-XY', '-Z AXIS'),
    ('-xz', '-XZ', '-Y AXIS'),
    ('-yz', '-YZ', '-X AXIS'),
]


SCALE_BAR_UNIT_LABEL = 'µm'
SCALE_BAR_WORLD_UNITS_PER_DISPLAY_UNIT = 1.0
VIEWPORT_CAMERA_FOV = 90.0
SPINE_UNSELECTED_COLOR = 0xA0A0A0
SPINE_SELECTED_COLOR = 0xFF0000
OBI_LOGO_PATH = Path(__file__).resolve().parent / 'assets' / 'obi_logo.png'
OBI_LOGO_DATA_URI = (
    'data:image/png;base64,'
    + base64.b64encode(OBI_LOGO_PATH.read_bytes()).decode('ascii')
    if OBI_LOGO_PATH.is_file()
    else ''
)


def _resolve_username():
    """Return the active OBI or local username in display form."""
    username = os.environ.get('OBI_USERNAME')
    if not username:
        try:
            username = getpass.getuser()
        except (KeyError, OSError):
            username = 'unknown-user'
    username = str(username).strip()
    return username.upper() if username else 'UNKNOWN-USER'


class _MeshSpatialIndex:
    """Query full-resolution mesh vertices inside axis-aligned boxes."""

    def __init__(self, vertices):
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.tree = (
            cKDTree(self.vertices)
            if cKDTree is not None and len(self.vertices)
            else None
        )
        self.cell_size = None
        self.bins = None
        if self.tree is None and len(self.vertices):
            extent = np.ptp(self.vertices, axis=0)
            self.cell_size = max(
                float(np.max(extent)) / 128.0,
                np.finfo(np.float32).eps,
            )
            cell_coordinates = np.floor(
                self.vertices / self.cell_size
            ).astype(np.int64)
            self.bins = {}
            for vertex_index, cell in enumerate(map(tuple, cell_coordinates)):
                self.bins.setdefault(cell, []).append(vertex_index)

    def query_box(self, box_min, box_max):
        """Return all indexed vertices inside an inclusive box."""
        if not len(self.vertices):
            return np.empty((0, 3), dtype=np.float32)

        box_min = np.asarray(box_min, dtype=np.float32)
        box_max = np.asarray(box_max, dtype=np.float32)
        center = (box_min + box_max) * 0.5
        half_extent = (box_max - box_min) * 0.5
        if self.tree is not None:
            radius = max(
                float(np.max(half_extent)),
                np.finfo(np.float32).eps,
            )
            candidate_indices = self.tree.query_ball_point(
                center, radius, p=np.inf
            )
        else:
            cell_min = np.floor(box_min / self.cell_size).astype(np.int64)
            cell_max = np.floor(box_max / self.cell_size).astype(np.int64)
            candidate_indices = []
            for x in range(int(cell_min[0]), int(cell_max[0]) + 1):
                for y in range(int(cell_min[1]), int(cell_max[1]) + 1):
                    for z in range(int(cell_min[2]), int(cell_max[2]) + 1):
                        candidate_indices.extend(
                            self.bins.get((x, y, z), ())
                        )

        if not candidate_indices:
            return np.empty((0, 3), dtype=np.float32)
        candidate_indices = np.asarray(candidate_indices, dtype=np.intp)
        candidates = self.vertices[candidate_indices]
        inside = np.all(
            (candidates >= box_min) & (candidates <= box_max), axis=1
        )
        return candidates[inside]


def _scale_bar_js():
    """Build the legacy camera-aware browser scale-bar overlay."""
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


# ============================================================
# Fonts / CSS
# ============================================================

def _font_face_css(fonts_dir):
    """Build @font-face rules for bundled Archivo variable font files."""
    fonts_dir = Path(fonts_dir)
    faces = [
        ('Archivo-Variable.ttf', '100 900', 'normal'),
        ('Archivo-Italic-Variable.ttf', '100 900', 'italic'),
    ]
    rules = []
    for filename, weight, style in faces:
        font_path = fonts_dir / filename
        if not font_path.is_file():
            continue
        encoded = base64.b64encode(font_path.read_bytes()).decode('ascii')
        rules.append(
            "@font-face { font-family: 'SpineUI'; "
            f"src: url(data:font/ttf;base64,{encoded}) format('truetype-variations'); "
            f"font-weight: {weight}; font-style: {style}; }}"
        )
    return '\n'.join(rules)


def _build_css(fonts_dir):
    return f"""
<style>
{_font_face_css(fonts_dir)}

.sv-app {{
  --sv-bg: #F7F8FA;
  --sv-panel: #FFFFFF;
  --sv-ink: #14171C;
  --sv-muted: #8A94A3;
  --sv-line: #DADFE6;
  --sv-accent: #D6472E;
  --sv-accent-dark: #B93A24;
  --sv-danger: #C62828;
  --sv-ok: #1F8A55;
  font-family: 'SpineUI', 'Helvetica Neue', Arial, sans-serif;
  color: var(--sv-ink);
  background: var(--sv-bg);
  padding: 14px 26px 12px;
}}
.sv-app, .sv-app * {{ box-sizing: border-box; }}
.sv-app .widget-html {{ margin: 0 !important; }}
.sv-app .widget-html-content {{ padding: 0 !important; }}
.sv-app .widget-box, .sv-app .widget-vbox, .sv-app .widget-hbox, .sv-app .widget-gridbox {{
  overflow: visible !important;
}}

.sv-app .widget-button {{
  border-radius: 0 !important;
  box-shadow: none !important;
  font-family: inherit !important;
  font-size: 12.5px !important;
  font-weight: 600 !important;
  min-height: 27px !important;
}}
.sv-app .sv-btn {{
  background: #fff !important;
  border: 1px solid var(--sv-ink) !important;
  color: var(--sv-ink) !important;
}}
.sv-app .sv-btn:hover {{ background: #EEF0F3 !important; }}
.sv-app .sv-btn-icon {{ padding: 0 !important; text-align: center !important; }}
.sv-app .sv-nav-prev, .sv-app .sv-nav-next {{
  background-repeat: no-repeat !important;
  background-position: center !important;
  background-size: 11px 11px !important;
}}
.sv-app .sv-nav-prev {{
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2314171C' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='15 6 9 12 15 18'/%3E%3C/svg%3E") !important;
}}
.sv-app .sv-nav-next {{
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2314171C' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='9 6 15 12 9 18'/%3E%3C/svg%3E") !important;
}}
.sv-app .sv-nav-prev:disabled, .sv-app .sv-nav-next:disabled {{ opacity: 0.35 !important; }}

.sv-app .sv-btn-primary {{
  background: var(--sv-accent) !important;
  border: 1px solid var(--sv-accent) !important;
  color: #fff !important;
  font-weight: 700 !important;
}}
.sv-app .sv-btn-primary:hover {{ background: var(--sv-accent-dark) !important; }}

.sv-app .sv-toggle {{
  background: #fff !important;
  border: 1px solid var(--sv-ink) !important;
  color: var(--sv-ink) !important;
  min-height: 24px !important;
}}
.sv-app .sv-toggle-no.sv-active {{
  background: var(--sv-ok) !important;
  border-color: var(--sv-ok) !important;
  color: #fff !important;
}}
.sv-app .sv-toggle-yes.sv-active {{
  background: var(--sv-danger) !important;
  border-color: var(--sv-danger) !important;
  color: #fff !important;
}}

.sv-app .widget-dropdown > select {{
  border-radius: 0 !important;
  border: 1px solid var(--sv-ink) !important;
  font-family: inherit !important;
  background: #fff !important;
  color: var(--sv-ink) !important;
  font-size: 12.5px !important;
  box-shadow: none !important;
  height: 27px !important;
}}

.sv-header-row {{
  display: flex; justify-content: space-between; align-items: flex-end;
  border-bottom: 1px solid var(--sv-line);
  padding-bottom: 10px; margin-bottom: 10px;
}}
.sv-brand {{ display: flex; align-items: center; gap: 12px; }}
.sv-brand-logo {{ width: 76px; height: 36px; object-fit: contain; display: block; }}
.sv-header-title {{ font-size: 22px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.01em; }}
.sv-header-subtitle {{
  font-size: 11px; color: var(--sv-muted); text-transform: uppercase;
  letter-spacing: 0.09em; margin-left: 16px;
}}
.sv-stat-group {{ display: flex; gap: 34px; }}
.sv-stat-line {{ display: flex; justify-content: space-between; align-items: baseline; width: 150px; margin-left: auto; margin-bottom: 0; }}
.sv-stat-label {{ font-size: 11px; letter-spacing: 0.09em; color: var(--sv-muted); text-transform: uppercase; }}
.sv-stat-value {{ font-size: 11px; font-weight: 400; }}
.sv-progress-track {{ width: 150px; height: 4px; background: #E4E8ED; margin-left: auto; }}
.sv-progress-fill {{ height: 100%; background: var(--sv-accent); }}

.sv-col-header {{ font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; }}
.sv-col-header-row {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0px; }}
.sv-meta {{ font-size: 10.5px; color: var(--sv-muted); text-transform: uppercase; letter-spacing: 0.07em; }}
.sv-field-label {{
  font-size: 11px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.06em; color: var(--sv-muted); margin: 4px 0 8px;
}}
.sv-divider {{ border-top: 1px solid var(--sv-line); margin: 8px 0; }}

.sv-toggle-row {{ display: flex; justify-content: space-between; align-items: center; margin: 0; font-size: 13px; height: 26px; }}

.sv-missing {{ display: flex; justify-content: space-between; align-items: flex-start; }}
.sv-missing-label {{ font-size: 13px; font-weight: 700; line-height: 1.2; }}
.sv-missing-caption {{ font-size: 10.5px; color: var(--sv-muted);}}
.sv-missing-value {{ font-size: 28px; font-weight: 700; color: var(--sv-accent); line-height: 1; }}

.sv-status-row {{ display: flex; justify-content: space-between; margin: 4px 0; font-size: 13px; }}
.sv-status-row .v {{ font-weight: 700; }}
.sv-flag {{ color: var(--sv-accent); }}
.sv-subsections-complete {{ color: var(--sv-ok); }}
.sv-subsections-pending {{ color: #B06000; }}
.sv-ready {{ font-size: 11.5px; color: var(--sv-muted); margin-top: 8px; }}

.sv-caption {{ font-size: 11px; color: var(--sv-muted); line-height: 1.5; margin-top: 10px; }}

.sv-proj-cell {{ position: relative; border: 1px solid var(--sv-line); text-align: center; padding: 6px 2px 8px; }}
.sv-proj-cell:hover {{ background: #EEF0F3; }}
.sv-proj-cell.sv-active {{ background: var(--sv-ink); border-color: var(--sv-ink); }}
.sv-proj-cell.sv-active:hover {{ background: var(--sv-ink); }}
.sv-proj-label {{ font-size: 13px; font-weight: 700; }}
.sv-proj-cell.sv-active .sv-proj-label {{ color: #fff !important; }}
.sv-proj-caption {{
  font-size: 9.5px; color: var(--sv-muted); letter-spacing: 0.05em;
  text-transform: uppercase; margin-top: 0;
}}
.sv-proj-cell.sv-active .sv-proj-caption {{ color: #C7CCD4; }}
.sv-app .sv-proj-hit {{
  position: absolute !important; inset: 0 !important;
  width: 100% !important; height: 100% !important; min-height: 0 !important;
  margin: 0 !important; padding: 0 !important;
  opacity: 0; border: none !important; background: transparent !important;
  cursor: pointer;
}}
.sv-stats {{
  margin-top: 8px; max-height: 320px; overflow: auto;
  border-top: 1px solid var(--sv-line); padding-top: 6px;
}}
.sv-stats h4 {{ margin: 5px 0 3px; font-size: 11px; text-transform: uppercase; letter-spacing: .03em; }}
.sv-stats table {{
  border-collapse: collapse; width: max-content; min-width: 100%;
  font-size: 10px; line-height: 1.15; font-variant-numeric: tabular-nums;
}}
.sv-stats th, .sv-stats td {{
  border: 1px solid var(--sv-line); padding: 3px 6px; height: 20px;
  text-align: left; vertical-align: middle; white-space: nowrap;
}}
.sv-stats th {{
  position: sticky; top: 0; z-index: 1;
  background: #EEF0F3; font-weight: 700;
}}
.sv-stats td:nth-child(2), .sv-stats td:nth-child(n+4):nth-child(-n+10) {{ text-align: right; }}
.sv-stats .status-yes {{ color: var(--sv-ok); font-weight: 700; }}
.sv-stats .status-no {{ color: #B06000; }}
.sv-stats .section-missing {{ color: var(--sv-accent); font-weight: 700; }}
.sv-stats .section-no-missing {{ color: var(--sv-ok); font-weight: 700; }}
.sv-stats .section-not-set {{ color: var(--sv-muted); }}
</style>
"""


# ============================================================
# The preview app
# ============================================================

class SpineValidationDesign:
    """Builds and wires the redesigned control panel widget tree."""

    def __init__(self, mesh_path=None, morphology_path=None, validation_csv_path=None):
        self.state = DesignState(
            mesh_path=mesh_path,
            morphology_path=morphology_path,
            validation_csv_path=validation_csv_path,
        )
        self.username = _resolve_username()
        self._silent = False
        self._analysis_visible = False
        self._section_points = {}
        self._section_radii = {}
        self._section_centerlines = {}
        self._section_terminal_markers = None
        self._section_spine_objects = []
        self._section_spine_objects_by_index = []
        self._section_spine_vertices = []
        self._section_spine_vertices_by_index = []
        self._section_spine_normals_by_index = []
        self._spine_structure_objects = []
        self._spine_structure_base_object = None
        self._spine_selected = False
        self._section_load_task = None
        self._section_load_generation = 0
        self._section_geometry_visible = True
        self._screenshot_request_path = None
        self._screenshot_request_token = 0
        self._screenshot_save_task = None
        self._report_task = None
        self._persistence_dirty = False
        self._last_persistence_error = None
        self._build_widgets()
        self._initialize_viewport()
        self._wire_events()
        self.root = self._layout()
        self.refresh()
        self._start_section_load(self.state.section_index)

    # -- construction -----------------------------------------------------

    def _make_toggle_row(self, key, label):
        yes_btn = widgets.Button(description='Yes', layout=widgets.Layout(width='58px', height='26px'))
        no_btn = widgets.Button(description='No', layout=widgets.Layout(width='58px', height='26px'))
        yes_btn.add_class('sv-toggle')
        yes_btn.add_class('sv-toggle-yes')
        no_btn.add_class('sv-toggle')
        no_btn.add_class('sv-toggle-no')
        self.toggle_buttons[key] = (yes_btn, no_btn)
        label_html = widgets.HTML(value=f'<span>{label}</span>', layout=widgets.Layout(flex='1 1 auto'))
        row = widgets.HBox(
            [label_html, widgets.HBox([yes_btn, no_btn], layout=widgets.Layout(gap='4px'))],
            layout=widgets.Layout(width='100%', align_items='center', margin='2px 0'),
        )
        self.toggle_rows[key] = row
        return row

    def _build_widgets(self):
        self.header_html = widgets.HTML()

        # Section column
        self.section_header_html = widgets.HTML()
        self.section_dropdown = widgets.Dropdown(layout=widgets.Layout(width='calc(100% - 80px)', height='27px', margin='0 4px 0 0'))
        self.section_dropdown.add_class('sv-section-dropdown')
        self.btn_prev_section = widgets.Button(description='', layout=widgets.Layout(width='34px', height='27px'))
        self.btn_next_section = widgets.Button(description='', layout=widgets.Layout(width='34px', height='27px'))
        self.btn_prev_section.add_class('sv-btn')
        self.btn_prev_section.add_class('sv-btn-icon')
        self.btn_prev_section.add_class('sv-nav-prev')
        self.btn_next_section.add_class('sv-btn')
        self.btn_next_section.add_class('sv-btn-icon')
        self.btn_next_section.add_class('sv-nav-next')
        self.btn_show_section_geometry = widgets.Button(
            description='Hide Section Geometry',
            layout=widgets.Layout(
                width='calc(100% - 4px)',
                align_self='flex-start',
                min_width='0',
            ),
        )
        self.btn_show_section_geometry.add_class('sv-btn')

        self.subsection_header_html = widgets.HTML()
        self.subsection_dropdown = widgets.Dropdown(layout=widgets.Layout(width='calc(100% - 80px)', height='27px', margin='0 4px 0 0'))
        self.btn_prev_subsection = widgets.Button(description='', layout=widgets.Layout(width='34px', height='27px'))
        self.btn_next_subsection = widgets.Button(description='', layout=widgets.Layout(width='34px', height='27px'))
        self.btn_prev_subsection.add_class('sv-btn')
        self.btn_prev_subsection.add_class('sv-btn-icon')
        self.btn_prev_subsection.add_class('sv-nav-prev')
        self.btn_next_subsection.add_class('sv-btn')
        self.btn_next_subsection.add_class('sv-btn-icon')
        self.btn_next_subsection.add_class('sv-nav-next')

        self.missing_html = widgets.HTML()
        self.btn_flag_missing = widgets.Button(
            description='Flag missing spine +1', layout=widgets.Layout(width='100%')
        )
        self.btn_flag_missing.add_class('sv-btn')
        self.btn_subsection_done = widgets.Button(
            description='Subsection done, next', layout=widgets.Layout(width='100%')
        )
        self.btn_subsection_done.add_class('sv-btn-primary')

        # Spine review column
        self.spine_review_header = widgets.HTML()
        self.spine_dropdown = widgets.Dropdown(layout=widgets.Layout(width='calc(100% - 80px)', height='27px', margin='0 4px 0 0'))
        self.spine_dropdown.add_class('sv-spine-dropdown')
        self.btn_prev_spine = widgets.Button(description='', layout=widgets.Layout(width='34px', height='27px'))
        self.btn_next_spine = widgets.Button(description='', layout=widgets.Layout(width='34px', height='27px'))
        self.btn_prev_spine.add_class('sv-btn')
        self.btn_prev_spine.add_class('sv-btn-icon')
        self.btn_prev_spine.add_class('sv-nav-prev')
        self.btn_next_spine.add_class('sv-btn')
        self.btn_next_spine.add_class('sv-btn-icon')
        self.btn_next_spine.add_class('sv-nav-next')

        self.toggle_rows = {}
        self.toggle_buttons = {}
        for key, label in ANALYSIS_FIELDS:
            self._make_toggle_row(key, label)
        self._make_toggle_row(*VALID_STRUCTURE_FIELD)

        self.btn_show_structure = widgets.Button(
            description='Show Spine Structure', layout=widgets.Layout(width='100%')
        )
        self.btn_show_structure.add_class('sv-btn')
        self.btn_spine_done = widgets.Button(
            description='Spine done, next', layout=widgets.Layout(width='100%')
        )
        self.btn_spine_done.add_class('sv-btn-primary')

        # Status column
        self.status_html = widgets.HTML()
        self.btn_screenshot = widgets.Button(
            description='Screenshot',
            layout=widgets.Layout(width='calc(50% - 3px)'),
        )
        self.btn_toggle_analysis = widgets.Button(
            description='Hide Analysis',
            layout=widgets.Layout(width='calc(50% - 3px)'),
        )
        self.btn_generate_report = widgets.Button(
            description='Generate report',
            layout=widgets.Layout(width='calc(50% - 3px)'),
        )
        self.btn_register = widgets.Button(
            description='Register',
            layout=widgets.Layout(width='calc(50% - 3px)'),
        )
        for btn in (
            self.btn_screenshot,
            self.btn_toggle_analysis,
            self.btn_generate_report,
            self.btn_register,
        ):
            btn.add_class('sv-btn')
        self.btn_register.add_class('sv-btn-primary')
        self.ready_html = widgets.HTML()
        self.stats_html = widgets.HTML()

        # Projection column
        self.proj_header_html = widgets.HTML(value='<div class="sv-col-header">Visualization</div>')
        self.proj_cells = {}
        proj_buttons = []
        for code, main_label, sub_label in PROJECTION_CELLS:
            label_html = widgets.HTML(
                value=(
                    f'<div class="sv-proj-label">{main_label}</div>'
                    f'<div class="sv-proj-caption">{sub_label}</div>'
                )
            )
            btn = widgets.Button(description='', layout=widgets.Layout(width='100%'))
            btn.add_class('sv-proj-hit')
            cell = widgets.VBox([label_html, btn], layout=widgets.Layout(width='100%'))
            cell.add_class('sv-proj-cell')
            self.proj_cells[code] = (cell, btn)
            proj_buttons.append(cell)
        self.projection_grid = widgets.GridBox(
            proj_buttons,
            layout=widgets.Layout(
                width='100%',
                grid_template_columns='repeat(3, 1fr)',
                grid_template_rows='repeat(2, 1fr)',
            ),
        )
        self.btn_white_background = widgets.Button(
            description='Black background', layout=widgets.Layout(width='100%', margin='10px 0 0')
        )
        self.btn_white_background.add_class('sv-btn')

    def _install_scale_bar(self):
        """Install the legacy overlay once on the final K3D plot."""
        self.plot.additional_js_code = _scale_bar_js()

    def _initialize_viewport(self):
        """Create the K3D viewport and add real morphology context."""
        self.plot = k3d.plot(
            grid_visible=False,
            grid_auto_fit=False,
            camera_auto_fit=False,
            menu_visibility=False,
            camera_mode='trackball',
            camera_up_axis='y',
            camera_fov=VIEWPORT_CAMERA_FOV,
            background_color=0xF7F8FA,
            height=600,
        )
        self._section_points = {}
        self._section_radii = {}
        self._section_centerlines = {}
        self._section_terminal_markers = None
        self._section_spine_objects = []
        self._section_spine_objects_by_index = []
        self._section_spine_vertices = []
        self._section_spine_vertices_by_index = []
        self._section_spine_normals_by_index = []
        self._spine_structure_objects = []
        self._spine_structure_base_object = None
        self._selected_section_overlay = None
        self._subsection_overlay = None
        self._spine_bbox_lines = []
        self._global_mesh_object = None
        self._mesh_spatial_index = None
        self._section_point_cloud = None
        morphology = self.state.morphology
        if morphology is None:
            self.plot = build_sample_viewport(height=600)
            self._install_scale_bar()
            return

        raw_sections = geometry.get_sections_points(morphology)
        self._section_points = {
            int(section_id): np.asarray(points, dtype=np.float32)
            for section_id, points in raw_sections.items()
            if np.asarray(points).ndim == 2 and np.asarray(points).shape[1] == 3
        }
        self._section_radii = {}
        for section in morphology.morphology.sections:
            section_data = np.asarray(section.points, dtype=np.float32)
            if (
                section_data.ndim == 2
                and section_data.shape[1] >= 4
                and section_data.shape[0] >= 2
            ):
                self._section_radii[int(section.id)] = section_data[:, 3]
        for section_id, points in self._section_points.items():
            if len(points) == 0:
                continue
            line = k3d.line(
                points,
                width=1.5,
                color=0x888888,
                shader='simple',
                name=f'Morphology section {section_id}',
            )
            self.plot += line
            self._section_centerlines[section_id] = line

        dummy_vertices, dummy_faces = self._build_variable_radius_polyline_mesh(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            np.array([0.01, 0.01], dtype=np.float32),
        )
        selected_overlay = k3d.mesh(
            dummy_vertices,
            dummy_faces.reshape(-1),
            color=0xFF0000,
            opacity=0.3,
            flat_shading=True,
            wireframe=False,
            name='Selected morphology section with varying radius',
        )
        selected_overlay.visible = False
        self.plot += selected_overlay
        self._selected_section_overlay = selected_overlay

        subsection_overlay = k3d.line(
            np.zeros((2, 3), dtype=np.float32),
            width=0.05,
            color=0xFF8C00,
            shader='mesh',
            radial_segments=4,
            name='Current morphology subsection',
        )
        subsection_overlay.visible = False
        self.plot += subsection_overlay
        self._subsection_overlay = subsection_overlay

        bbox_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        bbox_dummy = np.zeros((2, 3), dtype=np.float32)
        self._spine_bbox_lines = []
        for edge_index, _edge in enumerate(bbox_edges):
            bbox_line = k3d.line(
                bbox_dummy,
                color=self._spine_bbox_color(),
                shader='simple',
                width=0.3,
                name=f'Selected spine bounding box {edge_index}',
            )
            bbox_line.visible = False
            bbox_line.opacity = 0.5
            self.plot += bbox_line
            self._spine_bbox_lines.append(bbox_line)

        if self.state.mesh_path is not None:
            vertices = data_loading.load_mesh_vertices_pylmesh(
                str(self.state.mesh_path), scale_factor=1e-3
            )
            vertices = np.asarray(vertices, dtype=np.float32)
            vertices = vertices[np.all(np.isfinite(vertices), axis=1)]
            if len(vertices):
                self._mesh_spatial_index = _MeshSpatialIndex(vertices)
                sample_count = max(1, int(round(len(vertices) * 0.15)))
                sample_indices = np.linspace(
                    0, len(vertices) - 1, sample_count, dtype=np.intp
                )
                self._global_mesh_object = k3d.points(
                    vertices[sample_indices],
                    point_size=0.06,
                    color=0x87CEEB,
                    opacity=0.45,
                    shader='flat',
                )
                self.plot += self._global_mesh_object
                self._section_point_cloud = k3d.points(
                    np.empty((0, 3), dtype=np.float32),
                    point_size=0.025,
                    color=0x2F80ED,
                    opacity=0.8,
                    shader='flat',
                    name='Selected section mesh points',
                )
                self._section_point_cloud.visible = False
                self.plot += self._section_point_cloud

        self._install_scale_bar()

    def _selected_section_id(self, index=None):
        index = self.state.section_index if index is None else index
        section = self.state.sections[index]
        return int(section.get('section_id', section['index']))

    def _clear_section_point_cloud(self):
        """Hide the full-resolution mesh points for the previous section."""
        if self._section_point_cloud is None:
            return
        self._section_point_cloud.positions = np.empty(
            (0, 3), dtype=np.float32
        )
        self._section_point_cloud.visible = False

    def _update_section_point_cloud(self, section_id):
        """Show all mesh vertices inside the selected section's expanded bounds."""
        if (
            self._section_point_cloud is None
            or self._mesh_spatial_index is None
        ):
            self._clear_section_point_cloud()
            return

        points = self._section_points.get(section_id)
        if points is None:
            self._clear_section_point_cloud()
            return
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            self._clear_section_point_cloud()
            return
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) == 0:
            self._clear_section_point_cloud()
            return

        section_min = points.min(axis=0)
        section_max = points.max(axis=0)
        section_center = (section_min + section_max) * 0.5
        section_half_extent = np.maximum(
            (section_max - section_min) * 0.5 * 1.05,
            np.finfo(np.float32).eps,
        )
        local_points = self._mesh_spatial_index.query_box(
            section_center - section_half_extent,
            section_center + section_half_extent,
        )
        self._section_point_cloud.positions = local_points
        self._section_point_cloud.visible = len(local_points) > 0

    def _camera_aspect_ratio(self):
        """Return the rendered viewport aspect ratio when it is available."""
        try:
            width = float(getattr(self.plot, 'width'))
            height = float(getattr(self.plot, 'height'))
        except (AttributeError, TypeError, ValueError):
            return 1.0
        if not np.isfinite(width) or not np.isfinite(height) or width <= 0 or height <= 0:
            return 1.0
        return width / height

    def _camera_distance_for_points(
        self, points, center, view_direction, camera_up, fallback_radius
    ):
        """Fit finite geometry in the perspective frustum with a small margin."""
        points = np.asarray(points, dtype=np.float32)
        center = np.asarray(center, dtype=np.float32)
        view = np.asarray(view_direction, dtype=np.float32)
        up = np.asarray(camera_up, dtype=np.float32)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or len(points) == 0
            or center.shape != (3,)
            or view.shape != (3,)
            or up.shape != (3,)
            or not np.all(np.isfinite(points))
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(view))
            or not np.all(np.isfinite(up))
        ):
            return max(float(fallback_radius), 1e-6) * 1.5

        view_norm = float(np.linalg.norm(view))
        view = view / view_norm if view_norm > 1e-6 else None
        if view is None:
            return max(float(fallback_radius), 1e-6) * 1.5
        up = up - np.dot(up, view) * view
        up_norm = float(np.linalg.norm(up))
        if up_norm <= 1e-6:
            return max(float(fallback_radius), 1e-6) * 1.5
        up /= up_norm
        right = np.cross(-view, up)
        right_norm = float(np.linalg.norm(right))
        if right_norm <= 1e-6:
            return max(float(fallback_radius), 1e-6) * 1.5
        right /= right_norm

        half_fov = np.deg2rad(VIEWPORT_CAMERA_FOV * 0.5)
        tan_vertical = float(np.tan(half_fov))
        tan_horizontal = max(self._camera_aspect_ratio(), 1e-6) * tan_vertical
        relative = points - center
        depth_offset = relative @ view
        vertical_offset = np.abs(relative @ up)
        horizontal_offset = np.abs(relative @ right)
        required_distance = max(
            float(np.max(depth_offset + vertical_offset / tan_vertical)),
            float(np.max(depth_offset + horizontal_offset / tan_horizontal)),
        )
        nearest_depth = float(np.max(depth_offset))
        near_margin = max(float(fallback_radius) * 0.05, 1e-3)
        required_distance = max(required_distance, nearest_depth + near_margin)
        return max(required_distance, 1e-6) * 1.05

    @staticmethod
    def _finite_focus_points(*point_sets):
        """Combine finite 3D point sets for camera fitting."""
        finite_sets = []
        for points in point_sets:
            try:
                points = np.asarray(points, dtype=np.float32)
            except (TypeError, ValueError):
                continue
            if points.ndim == 2 and points.shape[1] == 3 and len(points):
                points = points[np.all(np.isfinite(points), axis=1)]
                if len(points):
                    finite_sets.append(points)
        return (
            np.concatenate(finite_sets, axis=0)
            if finite_sets
            else np.empty((0, 3), dtype=np.float32)
        )

    def _focus_section(self, section_id):
        """Focus the camera on the selected section without clipping geometry."""
        points = self._section_points.get(section_id)
        if points is None or len(points) == 0:
            return
        focus_points = [points]
        if (
            self._selected_section_overlay is not None
            and self._selected_section_overlay.visible
        ):
            focus_points.append(self._selected_section_overlay.vertices)
        if self._section_point_cloud is not None and self._section_point_cloud.visible:
            focus_points.append(self._section_point_cloud.positions)
        focus_points = self._finite_focus_points(*focus_points)
        if len(focus_points) == 0:
            return
        bounds_min = focus_points.min(axis=0)
        bounds_max = focus_points.max(axis=0)
        center = (bounds_min + bounds_max) * 0.5
        radius = max(float(np.linalg.norm(bounds_max - bounds_min)) * 0.6, 1e-6)
        view_direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        distance = self._camera_distance_for_points(
            focus_points, center, view_direction, camera_up, radius
        )
        self.plot.camera_auto_fit = False
        position = center + view_direction * distance
        self.plot.camera = position.tolist() + center.tolist() + camera_up.tolist()

    def _show_selected_section(self, section_id):
        # The persistent centerline is the complete section path. Recolor it
        # directly so selection remains visibly red even when tube geometry is
        # unavailable or is rendered over the line.
        for centerline_id, centerline in self._section_centerlines.items():
            color = 0xFF0000 if centerline_id == section_id else 0x888888
            if centerline.color != color:
                centerline.color = color
            centerline.opacity = 1.0

        if self._section_terminal_markers is not None:
            try:
                self.plot -= self._section_terminal_markers
            except Exception:
                pass
            self._section_terminal_markers = None

        points = self._section_points.get(section_id)
        valid_points = np.empty((0, 3), dtype=np.float32)
        if points is not None:
            points = np.asarray(points, dtype=np.float32)
            if points.ndim == 2 and points.shape[1] == 3:
                valid_points = points[np.all(np.isfinite(points), axis=1)]
        if len(valid_points):
            self._section_terminal_markers = k3d.points(
                np.asarray([valid_points[0], valid_points[-1]], dtype=np.float32),
                point_size=0.25,
                color=self._spine_bbox_color(),
                shader='3d',
                name=f'Morphology section {section_id} terminal points',
            )
            self.plot += self._section_terminal_markers

        if self._selected_section_overlay is None:
            return
        radii = self._section_radii.get(section_id)
        if points is None or radii is None or len(points) == 0:
            self._selected_section_overlay.visible = False
            return
        tube_vertices, tube_faces = self._build_variable_radius_polyline_mesh(
            points,
            radii,
        )
        if len(tube_vertices) == 0 or len(tube_faces) == 0:
            self._selected_section_overlay.visible = False
            return
        self._selected_section_overlay.vertices = tube_vertices
        self._selected_section_overlay.indices = tube_faces.reshape(-1)
        self._selected_section_overlay.color = 0xFF0000
        self._selected_section_overlay.opacity = 0.3
        self._selected_section_overlay.visible = self._section_geometry_visible
        self._focus_section(section_id)

    def _toggle_section_geometry(self, _button):
        """Show or hide the focused section's variable-radius tube."""
        if self._selected_section_overlay is None:
            return
        self._section_geometry_visible = not self._section_geometry_visible
        self._selected_section_overlay.visible = self._section_geometry_visible
        self.refresh()

    def _set_camera_oriented(
        self, center, radius, view_direction, focus_points=None,
        preferred_up=None,
    ):
        """Focus along an axis while fitting the target geometry in view."""
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
            self._focus_section(self._selected_section_id())
            return
        view /= view_norm
        focus_points = self._finite_focus_points(focus_points)
        if preferred_up is not None:
            preferred_up = np.asarray(preferred_up, dtype=np.float32)
            if (
                preferred_up.shape != (3,)
                or not np.all(np.isfinite(preferred_up))
            ):
                preferred_up = None
        candidates = (
            [preferred_up]
            if preferred_up is not None
            else np.eye(3, dtype=np.float32)[
                np.argsort(np.abs(np.eye(3) @ view))
            ]
        )
        for candidate in candidates:
            up = candidate - np.dot(candidate, view) * view
            up_norm = float(np.linalg.norm(up))
            if up_norm <= 1e-6:
                continue
            up /= up_norm
            distance = (
                self._camera_distance_for_points(
                    focus_points, center, view, up, radius
                )
                if len(focus_points)
                else max(float(radius), 1e-6) * 1.5
            )
            self.plot.camera_auto_fit = False
            position = center + view * distance
            self.plot.camera = position.tolist() + center.tolist() + up.tolist()
            return
        self._focus_section(self._selected_section_id())

    @staticmethod
    def _subsection_extent_line(points):
        """Return the ordered subsection start/end line and its direction."""
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            return None
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) == 0:
            return None
        start = points[0]
        end = points[-1]
        extent = end - start
        length = float(np.linalg.norm(extent))
        if length <= 1e-6:
            return start, end, start, None, length, points
        direction = extent / length
        center = (start + end) * 0.5
        return start, end, center, direction, length, points

    def _focus_subsection(self, section_id, subsection_index):
        """Focus on the subsection start-to-end extent without manual rotation."""
        section = self.state.section
        subsections = section['subsections']
        if not 0 <= subsection_index < len(subsections):
            return
        subsection = subsections[subsection_index]
        points = self._section_points.get(section_id)
        if points is None or len(points) == 0:
            self._focus_section(section_id)
            return
        extent_line = self._subsection_extent_line(subsection['points'])
        if extent_line is None:
            self._focus_section(section_id)
            return
        _start, _end, center, line_direction, _length, focus_points = extent_line
        if line_direction is None:
            extent = points.max(axis=0) - points.min(axis=0)
            view_direction = np.zeros(3, dtype=np.float32)
            view_direction[int(np.argmax(extent))] = 1.0
            self._set_camera_oriented(
                center, subsection['radius'], view_direction,
                focus_points=focus_points,
            )
            return

        reference_axes = np.eye(3, dtype=np.float32)
        reference_axis = reference_axes[
            np.argmin(np.abs(reference_axes @ line_direction))
        ]
        view_direction = np.cross(line_direction, reference_axis)
        view_norm = float(np.linalg.norm(view_direction))
        if view_norm <= 1e-6:
            self._focus_section(section_id)
            return
        view_direction /= view_norm
        self._set_camera_oriented(
            center,
            subsection['radius'],
            view_direction,
            focus_points=focus_points,
            # Keep the subsection direction on the screen's horizontal axis.
            preferred_up=reference_axis,
        )

    def _show_selected_subsection(self, subsection_index, focus_camera=True):
        """Update the orange subsection overlay and optionally focus it."""
        if self._subsection_overlay is None:
            return
        section_id = self._selected_section_id()
        subsections = self.state.section['subsections']
        if not 0 <= subsection_index < len(subsections):
            self._subsection_overlay.visible = False
            return
        points = np.asarray(subsections[subsection_index]['points'], dtype=np.float32)
        if points.ndim != 2 or points.shape[0] == 0:
            self._subsection_overlay.visible = False
            return
        self._subsection_overlay.vertices = points
        self._subsection_overlay.color = 0xFF8C00
        self._subsection_overlay.width = 0.05
        self._subsection_overlay.visible = True
        if focus_camera:
            self._focus_subsection(section_id, subsection_index)

    def _clear_selected_subsection(self):
        if self._subsection_overlay is not None:
            self._subsection_overlay.visible = False

    def _selected_spine_object(self, spine_index=None):
        """Return the full-spine K3D object for a local spine index."""
        spine_index = self.state.spine_index if spine_index is None else spine_index
        if not 0 <= spine_index < len(self._section_spine_objects_by_index):
            return None
        return self._section_spine_objects_by_index[spine_index]

    def _clear_spine_geometry(self):
        """Remove temporary head/neck objects and restore the full spine."""
        for obj in self._spine_structure_objects:
            try:
                self.plot -= obj
            except Exception:
                pass
        self._spine_structure_objects.clear()
        if self._spine_structure_base_object is not None:
            self._spine_structure_base_object.visible = True
        self._spine_structure_base_object = None
        self.state.show_structure = False

    def _show_spine_geometry(self):
        """Toggle colored neck/head geometry for the selected real spine."""
        if self.state.show_structure:
            self._clear_spine_geometry()
            self._set_status('Spine geometry hidden.')
            self.refresh()
            return

        if self.state.morphology is None or not self._spine_selected:
            return
        spine_index = self.state.spine_index
        base_object = self._selected_spine_object(spine_index)
        if base_object is None:
            return

        spine_id = int(self.state.spine['global_id'])
        try:
            neck = self.state.morphology.spines.spine_mesh(
                spine_id,
                include_head=False,
            )
            head = self.state.morphology.spines.spine_mesh(
                spine_id,
                include_neck=False,
            )
            for component_mesh, color in (
                (neck, 0x64C864),
                (head, 0xFF6464),
            ):
                if component_mesh is None:
                    continue
                vertices = np.asarray(
                    getattr(component_mesh, 'vertices', []), dtype=np.float32
                )
                faces = np.asarray(
                    getattr(component_mesh, 'faces', []), dtype=np.uint32
                )
                if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
                    continue
                if faces.size == 0:
                    continue
                obj = k3d.mesh(
                    vertices,
                    faces.reshape(-1),
                    color=color,
                    opacity=1.0,
                    flat_shading=True,
                    wireframe=False,
                )
                self.plot += obj
                self._spine_structure_objects.append(obj)
        except Exception as exc:
            self._clear_spine_geometry()
            self._set_status(f'Could not display spine geometry: {exc}')
            self.refresh()
            return

        if not self._spine_structure_objects:
            self._clear_spine_geometry()
            self._set_status('No head or neck geometry available.')
            self.refresh()
            return

        self._spine_structure_base_object = base_object
        base_object.visible = False
        self.state.show_structure = True
        self._set_status('Spine geometry shown.')
        self.refresh()

    def _spine_bbox_color(self):
        """Return the high-contrast bounding-box color for the plot background."""
        return 0xFF0000 if self.state.white_background else 0xFFFF00

    def _hide_spine_bbox(self):
        for line in self._spine_bbox_lines:
            line.visible = False

    def _update_spine_bbox(self, vertices):
        """Update the reusable background-aware box around selected spine vertices."""
        if vertices is None:
            self._hide_spine_bbox()
            return
        vertices = np.asarray(vertices, dtype=np.float32)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or vertices.shape[0] == 0
            or not np.all(np.isfinite(vertices))
        ):
            self._hide_spine_bbox()
            return
        bmin = vertices.min(axis=0)
        bmax = vertices.max(axis=0)
        x0, y0, z0 = bmin
        x1, y1, z1 = bmax
        corners = np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ], dtype=np.float32)
        bbox_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for line, (start, end) in zip(self._spine_bbox_lines, bbox_edges):
            line.color = self._spine_bbox_color()
            line.vertices = np.array(
                [corners[start], corners[end]], dtype=np.float32
            )
            line.visible = True

    def _clear_section_spines(self):
        self._clear_spine_geometry()
        self._hide_spine_bbox()
        for obj in self._section_spine_objects:
            try:
                self.plot -= obj
            except Exception:
                pass
        self._section_spine_objects.clear()
        self._section_spine_objects_by_index.clear()
        self._section_spine_vertices.clear()
        self._section_spine_vertices_by_index.clear()
        self._section_spine_normals_by_index.clear()

    @staticmethod
    def _spine_normal_from_transformation(transformation):
        """Return the global spine-tip axis from a morph-spines transform."""
        if isinstance(transformation, BaseException) or transformation is None:
            return None
        try:
            rotation, _translation = transformation
            normal = rotation.apply(np.array([0.0, 1.0, 0.0], dtype=np.float32))
            normal = np.asarray(normal, dtype=np.float32)
        except Exception:
            return None
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            return None
        if float(np.linalg.norm(normal)) <= 1e-6:
            return None
        return normal

    @staticmethod
    def _camera_up_from_spine_normal(normal, view_direction):
        """Project a spine orientation into a valid camera-up direction."""
        try:
            view = np.asarray(view_direction, dtype=np.float32)
        except (TypeError, ValueError):
            view = np.array([], dtype=np.float32)
        view_norm = float(np.linalg.norm(view)) if view.size else 0.0
        if view.shape != (3,) or not np.all(np.isfinite(view)) or view_norm <= 1e-6:
            view = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            view /= view_norm

        try:
            candidate = np.asarray(normal, dtype=np.float32) if normal is not None else None
        except (TypeError, ValueError):
            candidate = None
        if candidate is not None and candidate.shape == (3,) and np.all(np.isfinite(candidate)):
            candidate = candidate - np.dot(candidate, view) * view
            candidate_norm = float(np.linalg.norm(candidate))
            if candidate_norm > 1e-6:
                return (candidate / candidate_norm).tolist()

        fallback_candidates = [
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            *np.eye(3, dtype=np.float32)[np.argsort(np.abs(np.eye(3) @ view))],
        ]
        for candidate in fallback_candidates:
            candidate = candidate - np.dot(candidate, view) * view
            candidate_norm = float(np.linalg.norm(candidate))
            if candidate_norm > 1e-6:
                return (candidate / candidate_norm).tolist()
        return [0.0, 1.0, 0.0]

    def _focus_spine(self, spine_index):
        """Focus the camera on one loaded spine, matching the legacy UI."""
        if not 0 <= spine_index < len(self._section_spine_vertices_by_index):
            return
        vertices = self._section_spine_vertices_by_index[spine_index]
        if vertices is None:
            return
        vertices = np.asarray(vertices, dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
            return
        self._update_spine_colors()
        bounds_min = vertices.min(axis=0)
        bounds_max = vertices.max(axis=0)
        self._update_spine_bbox(vertices)
        center = (bounds_min + bounds_max) * 0.5
        radius = max(float(np.linalg.norm(bounds_max - bounds_min)) * 0.6, 2.0)
        view_direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        normal = (
            self._section_spine_normals_by_index[spine_index]
            if spine_index < len(self._section_spine_normals_by_index)
            else None
        )
        camera_up = self._camera_up_from_spine_normal(normal, view_direction)
        distance = self._camera_distance_for_points(
            vertices,
            center,
            view_direction,
            np.asarray(camera_up, dtype=np.float32),
            radius,
        )
        self.plot.camera_auto_fit = False
        position = center + view_direction * distance
        self.plot.camera = position.tolist() + center.tolist() + camera_up

    def _ensure_spine_analysis_defaults(self, spine=None):
        """Default unset validity answers to No when a spine is selected."""
        spine = self.state.spine if spine is None else spine
        changed = False
        answers = spine.setdefault('answers', {})
        for key in VALIDITY_FIELDS:
            if answers.get(key) not in {'yes', 'no'}:
                answers[key] = 'no'
                changed = True
        if changed:
            self._save_state()
        return changed

    def _focus_first_spine(self):
        """Match the legacy final-subsection action by focusing spine zero."""
        for spine_index, vertices in enumerate(self._section_spine_vertices_by_index):
            if vertices is not None:
                self.state.spine_index = spine_index
                self._spine_selected = True
                self._ensure_spine_analysis_defaults()
                self._focus_spine(spine_index)
                self.refresh()
                return

    def _update_spine_colors(self):
        """Keep full-spine meshes gray except for the focused spine."""
        selected_index = (
            self.state.spine_index if self._spine_selected else None
        )
        for spine_index, spine_object in enumerate(
            self._section_spine_objects_by_index
        ):
            if spine_object is None:
                continue
            color = (
                SPINE_SELECTED_COLOR
                if spine_index == selected_index
                else SPINE_UNSELECTED_COLOR
            )
            if spine_object.color != color:
                spine_object.color = color

    @staticmethod
    def _spine_color(_index):
        """Return the default gray color for an unselected full spine."""
        return SPINE_UNSELECTED_COLOR

    @staticmethod
    def _build_variable_radius_polyline_mesh(points, radii, radial_segments=8):
        """Build a triangle tube around a centerline with per-point radii."""
        points = np.asarray(points, dtype=np.float32)
        radii = np.asarray(radii, dtype=np.float32).reshape(-1)
        empty_vertices = np.empty((0, 3), dtype=np.float32)
        empty_faces = np.empty((0, 3), dtype=np.uint32)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or len(points) != len(radii)
            or len(points) < 2
        ):
            return empty_vertices, empty_faces

        valid = np.all(np.isfinite(points), axis=1) & np.isfinite(radii)
        points = points[valid]
        radii = radii[valid]
        if len(points) < 2:
            return empty_vertices, empty_faces
        radii = np.maximum(radii, np.finfo(np.float32).eps)
        radial_segments = max(int(radial_segments), 3)

        tangents = np.zeros_like(points)
        for point_index in range(len(points)):
            if point_index == 0:
                tangent = points[1] - points[0]
            elif point_index == len(points) - 1:
                tangent = points[-1] - points[-2]
            else:
                tangent = points[point_index + 1] - points[point_index - 1]
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm <= np.finfo(np.float32).eps:
                for candidate_index in range(1, len(points)):
                    candidate = points[candidate_index] - points[point_index]
                    candidate_norm = float(np.linalg.norm(candidate))
                    if candidate_norm > np.finfo(np.float32).eps:
                        tangent = candidate
                        tangent_norm = candidate_norm
                        break
            if tangent_norm <= np.finfo(np.float32).eps:
                tangent = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                tangent_norm = 1.0
            tangents[point_index] = tangent / tangent_norm

        def initial_normal(tangent):
            reference = np.eye(3, dtype=np.float32)[
                int(np.argmin(np.abs(tangent)))
            ]
            normal = np.cross(tangent, reference)
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm <= np.finfo(np.float32).eps:
                reference = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                normal = np.cross(tangent, reference)
                normal_norm = float(np.linalg.norm(normal))
            return normal / max(normal_norm, np.finfo(np.float32).eps)

        normals = np.zeros_like(points)
        binormals = np.zeros_like(points)
        normals[0] = initial_normal(tangents[0])
        binormals[0] = np.cross(tangents[0], normals[0])
        for point_index in range(1, len(points)):
            normal = normals[point_index - 1]
            normal = normal - np.dot(normal, tangents[point_index]) * tangents[point_index]
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm <= np.finfo(np.float32).eps:
                normal = initial_normal(tangents[point_index])
            else:
                normal /= normal_norm
            normals[point_index] = normal
            binormals[point_index] = np.cross(tangents[point_index], normal)

        angles = np.linspace(
            0.0, 2.0 * np.pi, radial_segments, endpoint=False, dtype=np.float32
        )
        circle_cos = np.cos(angles)
        circle_sin = np.sin(angles)
        rings = np.stack([
            points[:, None, :]
            + radii[:, None, None] * (
                circle_cos[None, :, None] * normals[:, None, :]
                + circle_sin[None, :, None] * binormals[:, None, :]
            )
        ], axis=0)[0]
        vertices = rings.reshape(-1, 3)

        faces = []
        for point_index in range(len(points) - 1):
            ring_start = point_index * radial_segments
            next_ring_start = (point_index + 1) * radial_segments
            for radial_index in range(radial_segments):
                next_radial_index = (radial_index + 1) % radial_segments
                first = ring_start + radial_index
                second = ring_start + next_radial_index
                third = next_ring_start + next_radial_index
                fourth = next_ring_start + radial_index
                faces.extend(((first, second, third), (first, third, fourth)))

        start_center = len(vertices)
        end_center = start_center + 1
        vertices = np.vstack((vertices, points[0], points[-1])).astype(np.float32)
        for radial_index in range(radial_segments):
            next_radial_index = (radial_index + 1) % radial_segments
            faces.append((start_center, next_radial_index, radial_index))
            end_ring = (len(points) - 1) * radial_segments
            faces.append((end_center, end_ring + radial_index, end_ring + next_radial_index))
        return vertices, np.asarray(faces, dtype=np.uint32)

    @staticmethod
    def _spine_mesh_object(spine_mesh, color):
        if spine_mesh is None or getattr(spine_mesh, 'is_empty', False):
            return None
        vertices = np.asarray(spine_mesh.vertices, dtype=np.float32)
        if len(vertices) == 0:
            return None
        faces = np.asarray(getattr(spine_mesh, 'faces', []), dtype=np.uint32)
        if faces.size == 0:
            return k3d.points(vertices, point_size=0.05, color=color)
        return k3d.mesh(
            vertices,
            faces.reshape(-1),
            color=color,
            opacity=1.0,
            flat_shading=True,
            wireframe=False,
        )

    async def _load_section_spines(self, section_index, generation):
        """Load one section's spine meshes without blocking the notebook UI."""
        if self.state.morphology is None:
            return
        section_id = self._selected_section_id(section_index)
        spine_ids = list(
            self.state.morphology.spines.spine_indices_for_section(section_id + 1)
        )
        spine_meshes = await asyncio.gather(*(
            asyncio.to_thread(self.state.morphology.spines.spine_mesh, int(spine_id))
            for spine_id in spine_ids
        ))
        spine_transformations = getattr(
            self.state.morphology.spines, 'spine_transformations', None
        )
        if callable(spine_transformations):
            spine_transforms = await asyncio.gather(*(
                asyncio.to_thread(spine_transformations, int(spine_id))
                for spine_id in spine_ids
            ), return_exceptions=True)
        else:
            spine_transforms = [None] * len(spine_ids)
        if generation != self._section_load_generation:
            return
        for spine_index, spine_mesh in enumerate(spine_meshes):
            while len(self._section_spine_vertices_by_index) <= spine_index:
                self._section_spine_vertices_by_index.append(None)
            while len(self._section_spine_objects_by_index) <= spine_index:
                self._section_spine_objects_by_index.append(None)
            while len(self._section_spine_normals_by_index) <= spine_index:
                self._section_spine_normals_by_index.append(None)
            transform = (
                spine_transforms[spine_index]
                if spine_index < len(spine_transforms)
                else None
            )
            self._section_spine_normals_by_index[spine_index] = (
                self._spine_normal_from_transformation(transform)
            )
            obj = self._spine_mesh_object(
                spine_mesh, self._spine_color(spine_index)
            )
            if obj is None:
                continue
            vertices = np.asarray(spine_mesh.vertices, dtype=np.float32)
            self.plot += obj
            self._section_spine_objects.append(obj)
            self._section_spine_objects_by_index[spine_index] = obj
            self._section_spine_vertices.append(vertices)
            self._section_spine_vertices_by_index[spine_index] = vertices
        self._show_selected_subsection(self.state.subsection_index)
        if self._spine_selected:
            self._focus_spine(self.state.spine_index)
        self._update_spine_colors()
        self._set_status(f'Section {section_id} loaded.')
        self.refresh()

    def _section_load_error(self, task):
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as exc:
            self._set_status(f'Could not load section: {exc}')
            self.refresh()

    def _start_section_load(self, section_index):
        """Start a cancellable, generation-guarded load for one section."""
        self._section_load_generation += 1
        generation = self._section_load_generation
        if self._section_load_task is not None and not self._section_load_task.done():
            self._section_load_task.cancel()
        self._clear_section_spines()
        self._clear_section_point_cloud()
        self._clear_selected_subsection()
        # Every section load starts exactly like the legacy workflow: at the
        # first subsection and first spine, before any spine meshes are loaded.
        self.state.subsection_index = 0
        self.state.spine_index = 0
        self._spine_selected = False
        section_id = self._selected_section_id(section_index)
        self._show_selected_section(section_id)
        self._update_section_point_cloud(section_id)
        # Assign the first subsection geometry before focusing the camera. The
        # later async spine load re-applies this same subsection focus only; it
        # must not replace it with a spine focus.
        self._show_selected_subsection(0, focus_camera=True)
        if self.state.morphology is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._set_status('Section ready; start a notebook event loop to load meshes.')
            return
        self._section_load_task = loop.create_task(
            self._load_section_spines(section_index, generation)
        )
        self._section_load_task.add_done_callback(self._section_load_error)

    def _set_projection(self, code):
        directions = {
            'xy': ([0, 0, 1], [0, 1, 0]),
            'xz': ([0, 1, 0], [0, 0, 1]),
            'yz': ([1, 0, 0], [0, 1, 0]),
            '-xy': ([0, 0, -1], [0, 1, 0]),
            '-xz': ([0, -1, 0], [0, 0, 1]),
            '-yz': ([-1, 0, 0], [0, 1, 0]),
        }
        direction, up = directions[code]
        camera = np.asarray(self.plot.camera, dtype=np.float32)
        if camera.size < 6:
            self.plot.camera_auto_fit = True
            return
        target = camera[3:6]
        distance = max(float(np.linalg.norm(camera[:3] - target)), 1.0)
        position = target + np.asarray(direction, dtype=np.float32) * distance
        self.plot.camera_auto_fit = False
        self.plot.camera = position.tolist() + target.tolist() + up

    def _set_plot_background(self):
        self.plot.background_color = 0xF7F8FA if self.state.white_background else 0x14171C
        bbox_color = self._spine_bbox_color()
        for line in self._spine_bbox_lines:
            line.color = bbox_color
        if self._section_terminal_markers is not None:
            self._section_terminal_markers.color = bbox_color

    def _screenshot_output_dir(self):
        """Return the legacy mesh-adjacent directory used for screenshots."""
        mesh_path = self.state.mesh_path
        if mesh_path is None:
            return None
        return mesh_path.with_name(f'{mesh_path.stem}_proofreading')

    def _screenshot_path(self):
        """Reserve the next section/spine/view-specific screenshot path."""
        output_dir = self._screenshot_output_dir()
        if output_dir is None:
            raise RuntimeError('No mesh output path is available.')
        output_dir.mkdir(parents=True, exist_ok=True)

        section_id = self._selected_section_id()
        if self._spine_selected and self._selected_spine_object() is not None:
            spine_id = int(self.state.spine['global_id'])
            view_marker = 'structure' if self.state.show_structure else 'error'
            prefix = f'section_{section_id}_spine_{spine_id}_{view_marker}_'
        else:
            prefix = f'section_{section_id}_error_'

        pattern = re.compile(rf'^{re.escape(prefix)}(\d+)\.png$')
        sequence_numbers = [
            int(match.group(1))
            for candidate in output_dir.glob(f'{prefix}*.png')
            if (match := pattern.match(candidate.name)) is not None
        ]
        next_sequence = max(sequence_numbers, default=0) + 1
        return output_dir / f'{prefix}{next_sequence}.png'

    @staticmethod
    def _write_screenshot_file(encoded_image, destination):
        """Decode and write a captured WebGL PNG without blocking the UI."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(base64.b64decode(encoded_image, validate=True))

    def _live_canvas_capture_js(self, request_token):
        """Build frontend code that captures the canvas with the scale bar."""
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

    Promise.resolve(K3DInstance.render(true))
        .then(() => new Promise((resolve) => window.requestAnimationFrame(resolve)))
        .then(() => {{
            const source = K3DInstance.getWorld().renderer.domElement;
            if (!source) {{
                throw new Error('The K3D WebGL canvas is unavailable.');
            }}

            const sourceRect = source.getBoundingClientRect();
            if (!(source.width > 0 && source.height > 0
                    && sourceRect.width > 0 && sourceRect.height > 0)) {{
                throw new Error('The K3D canvas has no drawable dimensions.');
            }}

            const host = source.parentElement || source;
            const overlay = host.querySelector('#obi-scale-bar-overlay');
            if (!overlay) {{
                throw new Error('The K3D scale-bar overlay is unavailable.');
            }}
            const label = overlay.querySelector('#obi-scale-bar-overlay-label');
            const bar = overlay.querySelector('#obi-scale-bar-overlay-bar');
            if (!label || !bar) {{
                throw new Error('The K3D scale-bar contents are unavailable.');
            }}

            const output = document.createElement('canvas');
            output.width = source.width;
            output.height = source.height;
            const context = output.getContext('2d');
            if (!context) {{
                throw new Error('A 2D screenshot context is unavailable.');
            }}
            context.drawImage(source, 0, 0);

            const scaleX = source.width / sourceRect.width;
            const scaleY = source.height / sourceRect.height;
            const overlayRect = overlay.getBoundingClientRect();
            const labelRect = label.getBoundingClientRect();
            const barRect = bar.getBoundingClientRect();
            const overlayStyle = getComputedStyle(overlay);
            const labelStyle = getComputedStyle(label);
            const barStyle = getComputedStyle(bar);
            const cssX = (rect) => rect.left - sourceRect.left;
            const cssY = (rect) => rect.top - sourceRect.top;

            context.save();
            context.scale(scaleX, scaleY);

            const overlayX = cssX(overlayRect);
            const overlayY = cssY(overlayRect);
            context.fillStyle = overlayStyle.backgroundColor;
            if (typeof context.roundRect === 'function') {{
                context.beginPath();
                context.roundRect(
                    overlayX,
                    overlayY,
                    overlayRect.width,
                    overlayRect.height,
                    parseFloat(overlayStyle.borderRadius) || 0
                );
                context.fill();
            }} else {{
                context.fillRect(
                    overlayX,
                    overlayY,
                    overlayRect.width,
                    overlayRect.height
                );
            }}

            context.fillStyle = labelStyle.color;
            context.font = labelStyle.fontWeight + ' '
                + labelStyle.fontSize + ' ' + labelStyle.fontFamily;
            context.textBaseline = 'alphabetic';
            context.fillText(
                label.textContent || '',
                cssX(labelRect),
                cssY(labelRect) + (parseFloat(labelStyle.fontSize) || 12)
            );

            context.fillStyle = barStyle.backgroundColor;
            context.fillRect(
                cssX(barRect),
                cssY(barRect),
                barRect.width,
                barRect.height
            );
            const borderWidth = parseFloat(barStyle.borderTopWidth) || 0;
            if (borderWidth > 0) {{
                context.strokeStyle = barStyle.borderTopColor;
                context.lineWidth = borderWidth;
                context.strokeRect(
                    cssX(barRect) + borderWidth / 2,
                    cssY(barRect) + borderWidth / 2,
                    barRect.width - borderWidth,
                    barRect.height - borderWidth
                );
            }}
            context.restore();

            output.toBlob((blob) => {{
                if (!blob) {{
                    reportError(new Error('The browser could not encode the screenshot.'));
                    return;
                }}
                const reader = new FileReader();
                reader.onloadend = () => {{
                    const result = typeof reader.result === 'string' ? reader.result : '';
                    const separator = result.indexOf(',');
                    const encoded = separator >= 0 ? result.slice(separator + 1) : '';
                    if (!encoded) {{
                        reportError(new Error('The encoded screenshot is empty.'));
                        return;
                    }}
                    widgetView.model.save(
                        'screenshot',
                        `__K3D_CAPTURE__${{requestToken}}:${{encoded}}`,
                        {{patch: true}}
                    );
                }};
                reader.onerror = () => reportError(
                    reader.error || new Error('PNG read failed.')
                );
                reader.readAsDataURL(blob);
            }}, 'image/png');
        }})
        .catch(reportError);
}})();
"""

    async def _save_screenshot_file(self, encoded_image, destination, request_token):
        """Persist a captured PNG without blocking the notebook event loop."""
        try:
            await asyncio.to_thread(
                self._write_screenshot_file,
                encoded_image,
                destination,
            )
            if self._screenshot_request_token == request_token:
                self._set_status(f'Screenshot saved: {destination.name}')
        except Exception as exc:
            if self._screenshot_request_token == request_token:
                self._set_status(f'Failed to save screenshot: {exc}')
        finally:
            if self._screenshot_request_token == request_token:
                self._screenshot_request_path = None
                self.btn_screenshot.disabled = False
                self.refresh()

    def _on_screenshot_ready(self, change):
        """Receive the PNG payload returned by the K3D frontend."""
        payload = change.get('new')
        if not isinstance(payload, str) or not payload:
            return

        error_prefix = '__K3D_CAPTURE_ERROR__'
        if payload.startswith(error_prefix):
            token_text, _, message = payload[len(error_prefix):].partition(':')
            try:
                request_token = int(token_text)
            except ValueError:
                return
            if request_token != self._screenshot_request_token:
                return
            self._screenshot_request_path = None
            self.btn_screenshot.disabled = False
            self._set_status(f'Failed to capture screenshot: {message}')
            self.refresh()
            return

        success_prefix = '__K3D_CAPTURE__'
        if not payload.startswith(success_prefix):
            return
        token_text, separator, encoded_image = payload[len(success_prefix):].partition(':')
        if not separator or not encoded_image:
            return
        try:
            request_token = int(token_text)
        except ValueError:
            return
        if request_token != self._screenshot_request_token:
            return
        destination = self._screenshot_request_path
        if destination is None:
            return
        try:
            self._screenshot_save_task = asyncio.create_task(
                self._save_screenshot_file(
                    encoded_image,
                    destination,
                    request_token,
                )
            )
        except RuntimeError as exc:
            self._screenshot_request_path = None
            self.btn_screenshot.disabled = False
            self._set_status(f'Failed to save screenshot: {exc}')
            self.refresh()

    def _take_screenshot(self, _button):
        """Capture the displayed K3D canvas using the legacy live-canvas flow."""
        if self._screenshot_request_path is not None:
            return
        if self.state.mesh_path is None:
            self._set_status('Screenshot unavailable: no mesh output path.')
            self.refresh()
            return
        if self.state.morphology is None:
            self._set_status('Screenshot unavailable: no morphology loaded.')
            self.refresh()
            return

        try:
            screenshot_path = self._screenshot_path()
        except Exception as exc:
            self._set_status(f'Failed to prepare screenshot: {exc}')
            self.refresh()
            return

        self._screenshot_request_token += 1
        request_token = self._screenshot_request_token
        self._screenshot_request_path = screenshot_path
        self.btn_screenshot.disabled = True
        self._set_status('Capturing screenshot...')
        self.refresh()
        try:
            # Run capture in the existing K3D PlotView, matching the legacy UI.
            self.plot.additional_js_code = self._live_canvas_capture_js(request_token)
        except Exception as exc:
            self._screenshot_request_path = None
            self.btn_screenshot.disabled = False
            self._set_status(f'Failed to request screenshot: {exc}')
            self.refresh()

    def _wire_events(self):
        self.section_dropdown.observe(self._on_section_dropdown, names='value')
        self.btn_prev_section.on_click(lambda _b: self._step_section(-1))
        self.btn_next_section.on_click(lambda _b: self._step_section(1))
        self.btn_show_section_geometry.on_click(self._toggle_section_geometry)

        self.subsection_dropdown.observe(self._on_subsection_dropdown, names='value')
        self.btn_prev_subsection.on_click(lambda _b: self._step_subsection(-1))
        self.btn_next_subsection.on_click(lambda _b: self._step_subsection(1))

        self.btn_flag_missing.on_click(self._on_flag_missing)
        self.btn_subsection_done.on_click(self._on_subsection_done)

        self.spine_dropdown.observe(self._on_spine_dropdown, names='value')
        self.btn_prev_spine.on_click(lambda _b: self._step_spine(-1))
        self.btn_next_spine.on_click(lambda _b: self._step_spine(1))

        for key, (yes_btn, no_btn) in self.toggle_buttons.items():
            yes_btn.on_click(lambda _b, key=key: self._on_toggle(key, 'yes'))
            no_btn.on_click(lambda _b, key=key: self._on_toggle(key, 'no'))

        self.btn_show_structure.on_click(self._on_show_structure)
        self.btn_spine_done.on_click(self._on_spine_done)

        self.btn_screenshot.on_click(self._take_screenshot)
        self.btn_toggle_analysis.on_click(self._on_toggle_analysis)
        self.plot.observe(self._on_screenshot_ready, names='screenshot')
        self.btn_generate_report.on_click(self._on_generate_report)
        self.btn_register.on_click(self._on_register)

        for code, (_cell, btn) in self.proj_cells.items():
            btn.on_click(lambda _b, code=code: self._on_projection(code))
        self.btn_white_background.on_click(self._on_toggle_background)

    def _layout(self):
        section_col = widgets.VBox([
            self.section_header_html,
            widgets.HBox([self.section_dropdown, self.btn_prev_section, self.btn_next_section],
                         layout=widgets.Layout(width='100%', gap='4px', align_items='center')),
            self.btn_show_section_geometry,
            widgets.HTML(value='<div class="sv-divider"></div>'),
            self.subsection_header_html,
            widgets.HBox([self.subsection_dropdown, self.btn_prev_subsection, self.btn_next_subsection],
                         layout=widgets.Layout(width='100%', gap='4px', align_items='center')),
            widgets.HTML(value='<div class="sv-divider"></div>'),
            self.missing_html,
            widgets.HTML(value='<div style="height:4px"></div>'),
            self.btn_flag_missing,
            widgets.HTML(value='<div style="height:8px"></div>'),
            self.btn_subsection_done,
        ], layout=widgets.Layout(width='25%', padding='0 22px 0 0'))
        section_col.add_class('sv-col')

        review_col = widgets.VBox([
            self.spine_review_header,
            widgets.HBox([self.spine_dropdown, self.btn_prev_spine, self.btn_next_spine],
                         layout=widgets.Layout(width='100%', gap='4px', margin='0 0 6px', align_items='center')),
            *[self.toggle_rows[key] for key, _ in ANALYSIS_FIELDS],
            widgets.HTML(value='<div class="sv-divider"></div>'),
            self.btn_show_structure,
            self.toggle_rows[VALID_STRUCTURE_FIELD[0]],
            widgets.HTML(value='<div style="height:5px"></div>'),
            self.btn_spine_done,
        ], layout=widgets.Layout(width='30%', padding='0 22px'))
        review_col.add_class('sv-col')

        status_col = widgets.VBox([
            widgets.HTML(value='<div class="sv-col-header">Status</div>'),
            self.status_html,
            widgets.HTML(value='<div class="sv-divider"></div>'),
            widgets.HTML(value='<div class="sv-field-label" style="margin-top:0">Verification &amp; output</div>'),
            widgets.HBox([self.btn_screenshot, self.btn_toggle_analysis],
                         layout=widgets.Layout(width='100%', gap='6px', margin='0 0 6px')),
            widgets.HBox([self.btn_generate_report, self.btn_register],
                         layout=widgets.Layout(width='100%')),
            self.ready_html,
        ], layout=widgets.Layout(width='22%', padding='0 22px'))
        status_col.add_class('sv-col')

        projection_col = widgets.VBox([
            self.proj_header_html,
            self.projection_grid,
            self.btn_white_background,
        ], layout=widgets.Layout(width='23%', padding='0 0 0 22px'))
        projection_col.add_class('sv-col')

        body = widgets.HBox(
            [section_col, review_col, status_col, projection_col],
            layout=widgets.Layout(width='100%', align_items='flex-start'),
        )

        viewport = widgets.VBox(
            [
                widgets.HTML(value='<div class="sv-col-header">Morphology viewport</div>'),
                self.plot,
            ],
            layout=widgets.Layout(width='100%', margin='14px 0 0'),
        )

        root = widgets.VBox(
            [self.header_html, body, viewport, self.stats_html],
            layout=widgets.Layout(width='100%'),
        )
        root.add_class('sv-app')
        return root

    # -- event handlers -----------------------------------------------------

    def _select_section(self, section_index):
        if not self._save_state_required():
            self.refresh()
            return
        self.state.section_index = max(
            0, min(section_index, len(self.state.sections) - 1)
        )
        self.state.subsection_index = 0
        self.state.spine_index = 0
        self.refresh()
        self._start_section_load(self.state.section_index)

    def _step_section(self, delta):
        self._select_section(self.state.section_index + delta)

    def _on_section_dropdown(self, change):
        if self._silent or change['name'] != 'value' or change['new'] is None:
            return
        self._select_section(change['new'])

    def _select_subsection(self, subsection_index):
        if not self._save_state_required():
            self.refresh()
            return
        subsections = self.state.section['subsections']
        self.state.subsection_index = max(
            0, min(subsection_index, len(subsections) - 1)
        )
        self._clear_spine_geometry()
        self._spine_selected = False
        self._update_spine_colors()
        self._hide_spine_bbox()
        self.refresh()
        self._show_selected_subsection(self.state.subsection_index)

    def _step_subsection(self, delta):
        self._select_subsection(self.state.subsection_index + delta)

    def _on_subsection_dropdown(self, change):
        if self._silent or change['name'] != 'value' or change['new'] is None:
            return
        self._select_subsection(change['new'])

    def _on_flag_missing(self, _button):
        subsection = self.state.subsection
        subsection['missing_count'] += 1
        self.state.section['missing_status'] = 'Missing'
        self._set_status('Missing spine flagged.')
        if not self._save_state_required():
            self.refresh()
            return
        self.refresh()
        self._show_selected_subsection(self.state.subsection_index, focus_camera=False)

    def _on_subsection_done(self, _button):
        """Match validate_spines_core's subsection completion workflow."""
        section = self.state.section
        current_subsections = section['subsections']
        current_index = self.state.subsection_index
        subsection_missing_total = sum(
            sub['missing_count'] for sub in current_subsections
        )
        section['missing_status'] = (
            'Missing' if subsection_missing_total > 0 else 'No Missing'
        )
        if 0 <= current_index < len(current_subsections):
            current_subsections[current_index]['done'] = True
        if not self._save_state_required():
            self.refresh()
            return

        # Equivalent to refresh_section_dropdown() in the reference UI: make
        # the completed subsection and section missing status visible before
        # navigating to the next action.
        self._set_status('Subsection marked done.')
        self.refresh()
        if (
            current_subsections
            and current_index >= len(current_subsections) - 1
        ):
            # The reference keeps the final subsection selected and focuses
            # the first loaded spine only after the final subsection is done.
            self.state.spine_index = 0
            self._set_status('Section subsections marked done.')
            self.refresh()
            self._focus_first_spine()
        else:
            # Equivalent to on_next_subsection(None) in validate_spines_core.
            self._step_subsection(1)

    def _spine_review_is_enabled(self):
        """Return whether all current-section subsections have been reviewed."""
        subsections = self.state.section['subsections']
        return bool(subsections) and all(subsection['done'] for subsection in subsections)

    def _step_spine(self, delta):
        if not self._spine_review_is_enabled():
            return
        if not self._save_state_required():
            self.refresh()
            return
        spines = self.state.section['spines']
        self._clear_spine_geometry()
        self.state.spine_index = max(0, min(self.state.spine_index + delta, len(spines) - 1))
        self._spine_selected = True
        self._ensure_spine_analysis_defaults()
        self.refresh()
        self._focus_spine(self.state.spine_index)

    def _on_spine_dropdown(self, change):
        if (
            self._silent
            or change['name'] != 'value'
            or change['new'] is None
            or not self._spine_review_is_enabled()
        ):
            return
        if not self._save_state_required():
            self.refresh()
            return
        self._clear_spine_geometry()
        self.state.spine_index = change['new']
        self._spine_selected = True
        self._ensure_spine_analysis_defaults()
        self.refresh()
        self._focus_spine(self.state.spine_index)

    def _on_toggle(self, key, value):
        if not self._spine_review_is_enabled():
            return
        spine = self.state.spine
        spine['answers'][key] = value
        spine['validity'] = None
        spine['checked'] = False
        if not self._save_state_required():
            self.refresh()
            return
        self.refresh()

    def _on_show_structure(self, _button):
        if not self._spine_review_is_enabled():
            return
        self._show_spine_geometry()

    def _next_unchecked_spine_index(self, start_index):
        """Find the next incomplete spine in the current section."""
        spines = self.state.section['spines']
        if not spines:
            return None
        for offset in range(1, len(spines) + 1):
            candidate = (start_index + offset) % len(spines)
            if not spines[candidate]['checked']:
                return candidate
        return None

    def _next_incomplete_section_index(self):
        """Find the next incomplete section, wrapping around once."""
        sections = self.state.sections
        if not sections:
            return None
        for offset in range(1, len(sections) + 1):
            candidate = (self.state.section_index + offset) % len(sections)
            if not self.state.section_is_checked(sections[candidate]):
                return candidate
        return None

    def _on_spine_done(self, _button):
        if not self._spine_review_is_enabled():
            return
        section = self.state.section
        spines = section['spines']
        current_index = self.state.spine_index
        current_spine = spines[current_index]
        self._clear_spine_geometry()
        derived_validity = derive_spine_validity(current_spine)
        if derived_validity is None:
            self._set_status(
                'Answer every defect-analysis question before marking the spine done.'
            )
            self.refresh()
            return
        current_spine['validity'] = derived_validity
        current_spine['checked'] = True
        if not self._save_state_required():
            self.refresh()
            return

        next_spine_index = self._next_unchecked_spine_index(current_index)
        if next_spine_index is not None:
            self.state.spine_index = next_spine_index
            self._spine_selected = True
            self._ensure_spine_analysis_defaults()
            self._set_status('Spine marked done.')
            self.refresh()
            self._focus_spine(next_spine_index)
            return

        # All spines in the section are now complete. Persist the derived
        # section state and update the visible status before changing sections.
        section_completed = self.state.section_is_checked(section)
        if not section_completed:
            self._set_status('Spine marked done.')
            self.refresh()
            return

        completed_section_id = section.get('section_id', section['index'])
        next_section_index = self._next_incomplete_section_index()
        if next_section_index is None:
            self._spine_selected = True
            self._set_status(
                f'Section {completed_section_id} completed. All sections are complete.'
            )
            self.refresh()
            self._focus_spine(current_index)
            return

        next_section_id = self.state.sections[next_section_index].get(
            'section_id', self.state.sections[next_section_index]['index']
        )
        navigation_status = (
            f'Section {completed_section_id} completed. '
            f'Moving to section {next_section_id}.'
        )
        self._select_section(next_section_index)
        # Re-apply subsection zero focus after the new section state has been
        # committed. This keeps the final-spine transition deterministic even
        # when section loading is asynchronous.
        self.state.subsection_index = 0
        self._show_selected_subsection(0, focus_camera=True)
        self._set_status(navigation_status)
        self.refresh()

    def _on_projection(self, code):
        self.state.current_view = code
        self._set_projection(code)
        self.refresh()

    def _on_toggle_background(self, _button):
        self.state.white_background = not self.state.white_background
        self._set_plot_background()
        self.refresh()

    def _set_status(self, message):
        self.state.status_message = message

    def _save_state(self):
        """Persist a complete validation snapshot and track failures."""
        if self.state.validation_csv_path is None:
            self._persistence_dirty = False
            self._last_persistence_error = None
            return None
        try:
            path = save_validation_state(self.state)
        except Exception as exc:
            self._persistence_dirty = True
            self._last_persistence_error = exc
            self._set_status(
                f'UNSAVED: could not write validation data '
                f'({type(exc).__name__}: {exc})'
            )
            return None
        self._persistence_dirty = path is None
        self._last_persistence_error = None
        return path

    def _save_state_required(self):
        """Save before continuing; block navigation when persistence fails."""
        if self.state.validation_csv_path is None:
            return True
        return self._save_state() is not None

    def _on_register(self, _button):
        if not self._save_state_required():
            self.refresh()
            return
        self._set_status('Assessment registered.')
        self.refresh()

    async def _generate_report_worker(self):
        """Persist the assessment and create the legacy analysis PDF."""
        try:
            self._set_status('Saving validation data...')
            self.refresh()
            csv_path = self._save_state()
            if csv_path is None:
                raise RuntimeError('Report generation requires a validation CSV path.')
            csv_path = Path(csv_path)

            screenshot_task = self._screenshot_save_task
            if screenshot_task is not None and not screenshot_task.done():
                self._set_status('Waiting for the latest screenshot...')
                self.refresh()
                await screenshot_task

            self._set_status('Generating PDF report...')
            self.refresh()
            try:
                report_module = importlib.import_module('validate_spine_analysis')
            except ModuleNotFoundError as exc:
                if exc.name != 'validate_spine_analysis' or not __package__:
                    raise
                from . import validate_spine_analysis as report_module
            report_module = importlib.reload(report_module)
            report_path = await asyncio.to_thread(
                report_module.generate_report,
                images_dir=csv_path.parent,
                csv_path=csv_path,
                fonts_dir=Path(__file__).resolve().parent / 'fonts',
                output_dir=csv_path.parent,
            )
            report_path = Path(report_path)
            if not report_path.is_file():
                raise FileNotFoundError(
                    f'Report generation did not produce: {report_path}'
                )
            self._set_status(f'Report generated: {report_path.name}')
        except Exception as exc:
            error_detail = html.escape(f'{type(exc).__name__}: {exc}')
            self._set_status(f'Report generation failed: {error_detail}')
        finally:
            self.btn_generate_report.disabled = False
            self._report_task = None
            self.refresh()

    def _on_toggle_analysis(self, _button):
        """Toggle visibility of the full mesh analysis summary."""
        self._analysis_visible = not self._analysis_visible
        self.refresh()

    def _on_generate_report(self, _button):
        """Create the current assessment PDF without blocking the UI."""
        if self._report_task is not None and not self._report_task.done():
            return
        if self.state.validation_csv_path is None:
            self._set_status('Report unavailable: no validation output path.')
            self.refresh()
            return
        self.btn_generate_report.disabled = True
        self._set_status('Preparing report...')
        self.refresh()
        try:
            self._report_task = asyncio.create_task(self._generate_report_worker())
        except RuntimeError as exc:
            self.btn_generate_report.disabled = False
            self._set_status(f'Report generation failed: {exc}')
            self.refresh()

    # -- rendering -----------------------------------------------------

    def refresh(self):
        s = self.state
        s.clamp_indices()
        section = s.section
        subsection = s.subsection
        spine = s.spine
        spine_count = len(section['spines'])
        checked_count = s.section_checked_count(section)
        subsection_count = len(section['subsections'])
        subsections_done = sum(1 for sub in section['subsections'] if sub['done'])
        missing_total = sum(sub['missing_count'] for sub in section['subsections'])
        flag_count = s.spine_flag_count(spine)

        self._silent = True
        try:
            self._refresh_header()
            self._refresh_section_column(section, subsection, subsection_count)
            self._refresh_spine_review(section, spine, spine_count)
            self._refresh_status(section, checked_count, spine_count, subsections_done, subsection_count,
                                  missing_total, flag_count)
            self._refresh_projection()
            self._refresh_stats()
        finally:
            self._silent = False

    def _refresh_header(self):
        s = self.state
        sections_checked = s.sections_checked_count()
        spines_checked = s.total_spines_checked()
        sections_pct = 100.0 * sections_checked / s.total_sections if s.total_sections else 0.0
        spines_pct = 100.0 * spines_checked / s.total_spines if s.total_spines else 0.0
        neuron_label = html.escape(str(s.neuron_id))
        username_label = html.escape(self.username)
        logo_html = (
            f'<img class="sv-brand-logo" src="{OBI_LOGO_DATA_URI}" '
            'alt="Open Brain Institute logo">'
            if OBI_LOGO_DATA_URI
            else ''
        )
        self.header_html.value = f"""
        <div class="sv-header-row">
          <div class="sv-brand">
            {logo_html}
            <div>
              <span class="sv-header-title">Spine Validation</span>
              <span class="sv-header-subtitle">Neuron {neuron_label} · {username_label}</span>
            </div>
          </div>
          <div class="sv-stat-group">
            <div>
              <div class="sv-stat-line">
                <span class="sv-stat-label">Sections</span>
                <span class="sv-stat-value">{sections_checked} / {s.total_sections}</span>
              </div>
              <div class="sv-progress-track"><div class="sv-progress-fill" style="width:{sections_pct:.1f}%"></div></div>
            </div>
            <div>
              <div class="sv-stat-line">
                <span class="sv-stat-label">Spines</span>
                <span class="sv-stat-value">{spines_checked} / {s.total_spines}</span>
              </div>
              <div class="sv-progress-track"><div class="sv-progress-fill" style="width:{spines_pct:.1f}%"></div></div>
            </div>
          </div>
        </div>
        """

    def _refresh_section_column(self, section, subsection, subsection_count):
        s = self.state
        section_id = section.get('section_id', section['index'])
        section_option_colors = []
        for option_index, sec in enumerate(s.sections, start=1):
            status = str(sec.get('missing_status', 'Not Set')).strip().casefold()
            if status.startswith('missing'):
                color = '#c5221f'
            elif status in {'no missing', 'no missing spines'}:
                color = '#188038'
            else:
                color = '#777777'
            section_option_colors.append(
                f'.sv-app .sv-section-dropdown select option:nth-child({option_index}) '
                f'{{ color: {color} !important; font-weight: 600; }}'
            )
        section_option_css = ''.join(section_option_colors)
        self.section_header_html.value = f"""
        <style>{section_option_css}</style>
        <div class="sv-col-header-row">
          <span class="sv-col-header">Section</span>
          <span class="sv-meta">Section {section_id}</span>
        </div>
        """
        section_options = [
            (
                f"Section {sec.get('section_id', sec['index'])} "
                f"({len(sec['spines'])} spines — "
                f"{s.section_checked_count(sec)}/{len(sec['spines'])}) "
                f"[{sec.get('missing_status', 'Not Set')}]",
                sec['index'],
            )
            for sec in s.sections
        ]
        self.section_dropdown.options = section_options
        self.section_dropdown.value = s.section_index
        self.btn_prev_section.disabled = s.section_index == 0
        self.btn_next_section.disabled = s.section_index == len(s.sections) - 1
        self.btn_show_section_geometry.disabled = (
            self._selected_section_overlay is None
        )
        self.btn_show_section_geometry.description = (
            'Hide Section Geometry'
            if self._section_geometry_visible
            else 'Show Section Geometry'
        )

        self.subsection_header_html.value = f"""
        <div class="sv-col-header-row">
          <span class="sv-field-label" style="margin:0">Subsection</span>
          <span class="sv-meta">{subsection['index'] + 1} / {subsection_count}</span>
        </div>
        """
        subsection_options = [
            (
                f"Subsection {sub['index'] + 1} (Missing : {sub['missing_count']}) "
                f"[{'Done' if sub['done'] else 'Not Set'}]",
                sub['index'],
            )
            for sub in section['subsections']
        ]
        self.subsection_dropdown.options = subsection_options
        self.subsection_dropdown.value = s.subsection_index
        self.btn_prev_subsection.disabled = subsection_count <= 1 or s.subsection_index == 0
        self.btn_next_subsection.disabled = (
            subsection_count <= 1 or s.subsection_index == subsection_count - 1
        )

        self.missing_html.value = f"""
        <div class="sv-missing">
          <div>
            <div class="sv-missing-label">Missing spines</div>
            <div class="sv-missing-caption">flagged in this section</div>
          </div>
          <div class="sv-missing-value">{subsection['missing_count']}</div>
        </div>
        """

    def _refresh_spine_review(self, section, spine, spine_count):
        spine_review_enabled = self._spine_review_is_enabled()
        spine_option_colors = []
        for option_index, sp in enumerate(section['spines'], start=1):
            status = str(self.state.spine_validity(sp)).strip().casefold()
            if status == 'valid':
                color = '#188038'
            elif status == 'invalid':
                color = '#c5221f'
            else:
                color = '#777777'
            spine_option_colors.append(
                f'.sv-app .sv-spine-dropdown select option:nth-child({option_index}) '
                f'{{ color: {color} !important; font-weight: 600; }}'
            )
        spine_option_css = ''.join(spine_option_colors)
        spine_options = [
            (
                f"Spine {sp['index'] + 1} ({sp['global_id']}) — "
                f"{sp['type']} [{str(self.state.spine_validity(sp)).title()}]",
                sp['index'],
            )
            for sp in section['spines']
        ]
        self.spine_dropdown.options = spine_options
        self.spine_dropdown.value = self.state.spine_index
        self.spine_dropdown.disabled = not spine_review_enabled
        self.btn_prev_spine.disabled = (
            not spine_review_enabled or self.state.spine_index == 0
        )
        self.btn_next_spine.disabled = (
            not spine_review_enabled
            or self.state.spine_index == spine_count - 1
        )
        for yes_btn, no_btn in self.toggle_buttons.values():
            yes_btn.disabled = not spine_review_enabled
            no_btn.disabled = not spine_review_enabled
        geometry_available = (
            spine_review_enabled
            and self.state.morphology is not None
            and self._spine_selected
            and self._selected_spine_object() is not None
        )
        self.btn_show_structure.disabled = not geometry_available
        self.btn_spine_done.disabled = (
            not spine_review_enabled
            or derive_spine_validity(spine) is None
        )

        self.spine_review_header.value = f"""
        <style>{spine_option_css}</style>
        <div class="sv-col-header-row">
          <span class="sv-col-header">Spine review</span>
          <span class="sv-meta">Spine {self.state.spine_index + 1} of {spine_count} ({spine['global_id']}) · {spine['type']}</span>
        </div>
        """

        for key in ALL_ANSWER_KEYS:
            yes_btn, no_btn = self.toggle_buttons[key]
            answer = spine['answers'][key]
            yes_btn.remove_class('sv-active')
            no_btn.remove_class('sv-active')
            if answer == 'yes':
                yes_btn.add_class('sv-active')
            else:
                no_btn.add_class('sv-active')

        self.btn_show_structure.description = (
            'Show Spine Geometry' if self.state.show_structure else 'Show Spine Structure'
        )

    def _refresh_status(self, section, checked_count, spine_count, subsections_done, subsection_count,
                         missing_total, flag_count):
        s = self.state
        section_checked = s.section_is_checked(section)
        subsection_status_class = (
            'sv-subsections-complete'
            if subsection_count > 0 and subsections_done == subsection_count
            else 'sv-subsections-pending'
        )
        spine_status_class = (
            'sv-subsections-complete'
            if spine_count > 0 and checked_count == spine_count
            else 'sv-subsections-pending'
        )
        missing_status_class = (
            'sv-subsections-complete' if missing_total == 0 else 'sv-flag'
        )
        flag_status_class = (
            'sv-subsections-complete' if flag_count == 0 else 'sv-flag'
        )
        self.status_html.value = f"""
        <div class="sv-status-row"><span>Section checked</span><span class="v">{'Yes' if section_checked else 'No'}</span></div>
        <div class="sv-status-row"><span>Subsections reviewed</span><span class="v {subsection_status_class}">{subsections_done} of {subsection_count}</span></div>
        <div class="sv-status-row"><span>Missing spines</span><span class="v {missing_status_class}">{missing_total}</span></div>
        <div class="sv-status-row"><span>Spines checked</span><span class="v {spine_status_class}">{checked_count} of {spine_count}</span></div>
        <div class="sv-status-row"><span>Flags raised on this spine</span><span class="v {flag_status_class}">{flag_count}</span></div>
        """
        self.ready_html.value = f'<div class="sv-ready">{s.status_message}</div>'

    def _refresh_projection(self):
        for code, (cell, _btn) in self.proj_cells.items():
            if code == self.state.current_view:
                cell.add_class('sv-active')
            else:
                cell.remove_class('sv-active')
        self.btn_white_background.description = (
            'Black background' if self.state.white_background else 'White background'
        )

    def _refresh_stats(self):
        """Render the legacy current-section and mesh summary tables."""
        self.btn_toggle_analysis.description = (
            'Hide Analysis' if self._analysis_visible else 'Show Analysis'
        )
        rows = section_summary_rows(self.state)
        if not rows:
            self.stats_html.value = ''
            return

        def missing_display(section, value):
            if value != 'Missing':
                return value
            count = sum(
                int(subsection.get('missing_count', 0))
                for subsection in section.get('subsections', [])
            )
            return f'Missing [{count}]'

        def render_table(title, table_rows):
            header_html = ''.join(
                f'<th>{html.escape(str(field))}</th>'
                for field in SECTION_CSV_FIELDS
            )
            body_rows = []
            for row, section in table_rows:
                cells = []
                for field in SECTION_CSV_FIELDS:
                    value = row[field]
                    if field == 'Missing Segmented Spines':
                        value = missing_display(section, value)
                    classes = []
                    if field == 'Validated (Yes or No)':
                        classes.append(f"status-{str(value).casefold()}")
                    elif field == 'Missing Segmented Spines':
                        classes.append(
                            f"section-{str(row[field]).casefold().replace(' ', '-')}"
                        )
                    class_attribute = (
                        f' class="{" ".join(classes)}"' if classes else ''
                    )
                    cells.append(
                        f'<td{class_attribute}>{html.escape(str(value))}</td>'
                    )
                body_rows.append(f'<tr>{"".join(cells)}</tr>')
            return (
                f'<h4>{html.escape(title)}</h4>'
                f'<table><thead><tr>{header_html}</tr></thead>'
                f'<tbody>{"".join(body_rows)}</tbody></table>'
            )

        current_section = self.state.section
        current_row = rows[self.state.section_index]
        mesh_analysis = ''
        if self._analysis_visible:
            mesh_analysis = render_table(
                'Mesh Analysis Summary',
                list(zip(rows, self.state.sections)),
            )
        self.stats_html.value = (
            '<div class="sv-stats">'
            + render_table('Current Section Analysis', [(current_row, current_section)])
            + mesh_analysis
            + '</div>'
        )



def build_sample_viewport(seed=3, height=420):
    """Build a throwaway K3D plot used when no real morphology is supplied."""
    rng = np.random.default_rng(seed)

    # A smoothed random walk stands in for a dendrite backbone.
    steps = rng.normal(scale=1.0, size=(200, 3))
    backbone = np.cumsum(steps, axis=0).astype(np.float32)
    backbone -= backbone.mean(axis=0)

    # Spine "heads" scattered near the backbone, colored by mock valid/invalid status.
    anchor_idx = rng.integers(0, len(backbone), size=60)
    offsets = rng.normal(scale=1.5, size=(60, 3))
    spine_points = (backbone[anchor_idx] + offsets).astype(np.float32)
    valid_mask = rng.random(60) > 0.3
    colors = np.where(valid_mask, 0x168A55, 0xD6472E).astype(np.uint32)

    plot = k3d.plot(
        height=height,
        background_color=0xF7F8FA,
        grid_visible=False,
        camera_auto_fit=True,
        camera_fov=VIEWPORT_CAMERA_FOV,
    )
    plot += k3d.line(backbone, color=0x14171C, width=0.08, shader='mesh')
    plot += k3d.points(
        spine_points, colors=colors, point_size=0.6, shader='3d',
    )
    return plot


# ============================================================
# Public entry points
# ============================================================

def build_preview(mesh_path=None, morphology_path=None, fonts_dir=None, validation_csv_path=None):
    """Build the redesigned widget tree, optionally from real morphology data."""
    fonts_dir = FONT_DIRECTORY if fonts_dir is None else Path(fonts_dir)
    display(HTML(_build_css(fonts_dir)))
    app = SpineValidationDesign(
        mesh_path=mesh_path,
        morphology_path=morphology_path,
        validation_csv_path=validation_csv_path,
    )
    return app.root


def display_preview(
    mesh_path,
    morphology_path,
    fonts_dir=None,
    show_sample_viewport=False,
    validation_csv_path=None,
):
    """Display the redesigned control panel for the supplied dataset.

    The section dropdown is populated from the same spiny-section helper used
    by the production validation workflow. The sample viewport remains an
    optional design aid and is disabled by default for real datasets.
    """
    clear_output(wait=True)
    root = build_preview(
        mesh_path=mesh_path,
        morphology_path=morphology_path,
        fonts_dir=fonts_dir,
        validation_csv_path=validation_csv_path,
    )
    display(root)

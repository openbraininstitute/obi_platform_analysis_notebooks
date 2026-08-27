"""Standalone visual redesign of the Spine Validation control panel.

This module is a self-contained UI preview: it reproduces the flat,
grid-lined "Spine Proofreading Utility" layout (header stats, Section /
Spine Review / Status / Projection columns, footer) with mock data so the
look and feel can be reviewed and clicked through inside a notebook,
independent of the real validation pipeline in ``validate_spines_core.py``.

Usage (from a notebook, with ``lib`` on ``sys.path``):

    import claude_design
    claude_design.display_preview()
"""

import base64
import random
from pathlib import Path

import ipywidgets as widgets
import k3d
import numpy as np
from IPython.display import HTML, display


FONT_DIRECTORY = Path(__file__).resolve().parent / 'fonts' / 'archivo'

ANALYSIS_FIELDS = [
    ('correct_type', 'Correct type'),
    ('false_positive', 'False positive'),
    ('incomplete_spine', 'Incomplete spine'),
    ('falsely_extended', 'Falsely extended'),
    ('merged_spine', 'Merged spine'),
    ('split_spine', 'Split spine'),
]
ISSUE_FIELDS = [key for key, _ in ANALYSIS_FIELDS if key != 'correct_type']
VALID_STRUCTURE_FIELD = ('valid_structure', 'Valid structure')
ALL_ANSWER_KEYS = [key for key, _ in ANALYSIS_FIELDS] + [VALID_STRUCTURE_FIELD[0]]

SPINE_TYPES = ['thin', 'stubby', 'mushroom', 'branched', 'filopodia']

PROJECTION_CELLS = [
    ('xy', 'XY', '+Z AXIS'),
    ('xz', 'XZ', '+Y AXIS'),
    ('yz', 'YZ', '+X AXIS'),
    ('-xy', '-XY', '-Z AXIS'),
    ('-xz', '-XZ', '-Y AXIS'),
    ('-yz', '-YZ', '-X AXIS'),
]

TOTAL_SECTIONS = 63
TOTAL_SPINES = 1047
NEURON_ID = '864691134886335738'


# ============================================================
# Fonts / CSS
# ============================================================

def _font_face_css(fonts_dir):
    """Build @font-face rules for the bundled Archivo variable font files, if present.

    Archivo is an open-source (SIL OFL) Google Font used as an Helvetica-Neue-
    style grotesque; the variable-font files cover the full weight range in
    one file each (upright and italic), so ``font-weight: 400..700`` in the
    CSS below is resolved by the browser rather than needing separate statics.
    """
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
  --sv-ok: #1F8A55;
  font-family: 'SpineUI', 'Helvetica Neue', Arial, sans-serif;
  color: var(--sv-ink);
  background: var(--sv-bg);
  padding: 14px 26px 12px;
}}
.sv-app, .sv-app * {{ box-sizing: border-box; }}
.sv-app .widget-html, .sv-app .widget-dropdown {{ margin: 0 !important; }}
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
.sv-app .sv-toggle.sv-active {{
  background: var(--sv-accent) !important;
  border-color: var(--sv-accent) !important;
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
</style>
"""


# ============================================================
# Mock data
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

    return {'index': section_index, 'spines': spines, 'subsections': subsections}, spine_count


def _build_mock_sections(seed=7):
    rng = random.Random(seed)
    sections = []
    global_id = 1
    for i in range(TOTAL_SECTIONS):
        section, spine_count = _make_section(i, rng, global_id)
        sections.append(section)
        global_id += spine_count
    return sections


# ============================================================
# State + derived values
# ============================================================

class DesignState:
    """Mock, in-memory state driving the preview UI."""

    def __init__(self):
        self.neuron_id = NEURON_ID
        self.sections = _build_mock_sections()
        self.section_index = 0
        self.subsection_index = 3
        self.spine_index = 0
        self.show_structure = False
        self.white_background = True
        self.current_view = 'xy'
        self.status_message = 'Ready.'

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
        answers = spine['answers']
        if any(value is None for value in answers.values()):
            return 'unset'
        invalid = (
            answers['correct_type'] == 'no'
            or answers['valid_structure'] == 'no'
            or any(answers[key] == 'yes' for key in ISSUE_FIELDS)
        )
        return 'invalid' if invalid else 'valid'

    def spine_flag_count(self, spine):
        return sum(1 for key in ISSUE_FIELDS if spine['answers'][key] == 'yes')

    def section_checked_count(self, section):
        return sum(1 for spine in section['spines'] if spine['checked'])

    def section_is_checked(self, section):
        return all(spine['checked'] for spine in section['spines'])

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


# ============================================================
# The preview app
# ============================================================

class SpineValidationDesign:
    """Builds and wires the redesigned control panel widget tree."""

    def __init__(self):
        self.state = DesignState()
        self._silent = False
        self._build_widgets()
        self._wire_events()
        self.root = self._layout()
        self.refresh()

    # -- construction -----------------------------------------------------

    def _make_toggle_row(self, key, label):
        yes_btn = widgets.Button(description='Yes', layout=widgets.Layout(width='58px', height='26px'))
        no_btn = widgets.Button(description='No', layout=widgets.Layout(width='58px', height='26px'))
        yes_btn.add_class('sv-toggle')
        no_btn.add_class('sv-toggle')
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
        self.section_dropdown = widgets.Dropdown(layout=widgets.Layout(width='calc(100% - 76px)', height='27px'))
        self.btn_prev_section = widgets.Button(description='', layout=widgets.Layout(width='34px', height='27px'))
        self.btn_next_section = widgets.Button(description='', layout=widgets.Layout(width='34px', height='27px'))
        self.btn_prev_section.add_class('sv-btn')
        self.btn_prev_section.add_class('sv-btn-icon')
        self.btn_prev_section.add_class('sv-nav-prev')
        self.btn_next_section.add_class('sv-btn')
        self.btn_next_section.add_class('sv-btn-icon')
        self.btn_next_section.add_class('sv-nav-next')

        self.subsection_header_html = widgets.HTML()
        self.subsection_dropdown = widgets.Dropdown(layout=widgets.Layout(width='calc(100% - 76px)', height='27px'))
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
        self.spine_dropdown = widgets.Dropdown(layout=widgets.Layout(width='calc(100% - 76px)', height='27px'))
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
            description='Show spine structure', layout=widgets.Layout(width='100%')
        )
        self.btn_show_structure.add_class('sv-btn')
        self.btn_spine_done = widgets.Button(
            description='Spine done, next', layout=widgets.Layout(width='100%')
        )
        self.btn_spine_done.add_class('sv-btn-primary')

        # Status column
        self.status_html = widgets.HTML()
        self.btn_screenshot = widgets.Button(description='Screenshot', layout=widgets.Layout(width='100%'))
        self.btn_generate_report = widgets.Button(description='Generate report', layout=widgets.Layout(width='100%'))
        self.btn_register = widgets.Button(description='Register', layout=widgets.Layout(width='calc(50% - 3px)'))
        for btn in (self.btn_screenshot, self.btn_generate_report, self.btn_register):
            btn.add_class('sv-btn')
        self.ready_html = widgets.HTML()

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
            description='White background', layout=widgets.Layout(width='100%', margin='10px 0 0')
        )
        self.btn_white_background.add_class('sv-btn')

    def _wire_events(self):
        self.section_dropdown.observe(self._on_section_dropdown, names='value')
        self.btn_prev_section.on_click(lambda _b: self._step_section(-1))
        self.btn_next_section.on_click(lambda _b: self._step_section(1))

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

        self.btn_screenshot.on_click(lambda _b: self._set_status('Screenshot saved.'))
        self.btn_generate_report.on_click(lambda _b: self._set_status('Report generated.'))
        self.btn_register.on_click(lambda _b: self._set_status('Assessment registered.'))

        for code, (_cell, btn) in self.proj_cells.items():
            btn.on_click(lambda _b, code=code: self._on_projection(code))
        self.btn_white_background.on_click(self._on_toggle_background)

    def _layout(self):
        section_col = widgets.VBox([
            self.section_header_html,
            widgets.HBox([self.section_dropdown, self.btn_prev_section, self.btn_next_section],
                         layout=widgets.Layout(width='100%', gap='4px', align_items='center')),
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
            widgets.HBox([self.btn_screenshot, self.btn_generate_report],
                         layout=widgets.Layout(width='100%', gap='6px', margin='0 0 6px')),
            widgets.HBox([self.btn_register], layout=widgets.Layout(width='100%')),
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

        root = widgets.VBox(
            [self.header_html, body],
            layout=widgets.Layout(width='100%'),
        )
        root.add_class('sv-app')
        return root

    # -- event handlers -----------------------------------------------------

    def _step_section(self, delta):
        self.state.section_index += delta
        self.state.section_index = max(0, min(self.state.section_index, len(self.state.sections) - 1))
        self.state.subsection_index = 0
        self.state.spine_index = 0
        self.refresh()

    def _on_section_dropdown(self, change):
        if self._silent or change['name'] != 'value' or change['new'] is None:
            return
        self.state.section_index = change['new']
        self.state.subsection_index = 0
        self.state.spine_index = 0
        self.refresh()

    def _step_subsection(self, delta):
        subsections = self.state.section['subsections']
        self.state.subsection_index = max(0, min(self.state.subsection_index + delta, len(subsections) - 1))
        self.refresh()

    def _on_subsection_dropdown(self, change):
        if self._silent or change['name'] != 'value' or change['new'] is None:
            return
        self.state.subsection_index = change['new']
        self.refresh()

    def _on_flag_missing(self, _button):
        self.state.subsection['missing_count'] += 1
        self._set_status('Missing spine flagged.')
        self.refresh()

    def _on_subsection_done(self, _button):
        self.state.subsection['done'] = True
        subsections = self.state.section['subsections']
        self.state.subsection_index = min(self.state.subsection_index + 1, len(subsections) - 1)
        self._set_status('Subsection marked done.')
        self.refresh()

    def _step_spine(self, delta):
        spines = self.state.section['spines']
        self.state.spine_index = max(0, min(self.state.spine_index + delta, len(spines) - 1))
        self.refresh()

    def _on_spine_dropdown(self, change):
        if self._silent or change['name'] != 'value' or change['new'] is None:
            return
        self.state.spine_index = change['new']
        self.refresh()

    def _on_toggle(self, key, value):
        spine = self.state.spine
        spine['answers'][key] = value
        spine['checked'] = all(spine['answers'][k] is not None for k in ALL_ANSWER_KEYS)
        self.refresh()

    def _on_show_structure(self, _button):
        self.state.show_structure = not self.state.show_structure
        self.refresh()

    def _on_spine_done(self, _button):
        spines = self.state.section['spines']
        self.state.spine_index = min(self.state.spine_index + 1, len(spines) - 1)
        self._set_status('Spine marked done.')
        self.refresh()

    def _on_projection(self, code):
        self.state.current_view = code
        self.refresh()

    def _on_toggle_background(self, _button):
        self.state.white_background = not self.state.white_background
        self.refresh()

    def _set_status(self, message):
        self.state.status_message = message

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
        finally:
            self._silent = False

    def _refresh_header(self):
        s = self.state
        sections_checked = s.sections_checked_count()
        spines_checked = s.total_spines_checked()
        sections_pct = 100.0 * sections_checked / TOTAL_SECTIONS
        spines_pct = 100.0 * spines_checked / TOTAL_SPINES
        self.header_html.value = f"""
        <div class="sv-header-row">
          <div>
            <span class="sv-header-title">Spine Validation</span>
            <span class="sv-header-subtitle">Neuron {s.neuron_id}</span>
          </div>
          <div class="sv-stat-group">
            <div>
              <div class="sv-stat-line">
                <span class="sv-stat-label">Sections</span>
                <span class="sv-stat-value">{sections_checked} / {TOTAL_SECTIONS}</span>
              </div>
              <div class="sv-progress-track"><div class="sv-progress-fill" style="width:{sections_pct:.1f}%"></div></div>
            </div>
            <div>
              <div class="sv-stat-line">
                <span class="sv-stat-label">Spines</span>
                <span class="sv-stat-value">{spines_checked} / {TOTAL_SPINES}</span>
              </div>
              <div class="sv-progress-track"><div class="sv-progress-fill" style="width:{spines_pct:.1f}%"></div></div>
            </div>
          </div>
        </div>
        """

    def _refresh_section_column(self, section, subsection, subsection_count):
        s = self.state
        self.section_header_html.value = f"""
        <div class="sv-col-header-row">
          <span class="sv-col-header">Section</span>
          <span class="sv-meta">Section {section['index']}</span>
        </div>
        """
        section_options = [
            (
                f"Section {sec['index']} — {len(sec['spines'])} spines — "
                f"{s.section_checked_count(sec)}/{len(sec['spines'])} checked",
                sec['index'],
            )
            for sec in s.sections
        ]
        self.section_dropdown.options = section_options
        self.section_dropdown.value = s.section_index
        self.btn_prev_section.disabled = s.section_index == 0
        self.btn_next_section.disabled = s.section_index == len(s.sections) - 1

        self.subsection_header_html.value = f"""
        <div class="sv-col-header-row">
          <span class="sv-field-label" style="margin:0">Subsection</span>
          <span class="sv-meta">{subsection['index'] + 1} / {subsection_count}</span>
        </div>
        """
        subsection_options = [
            (
                f"Subsection {sub['index'] + 1} — {sub['missing_count']} missing — "
                f"{'done' if sub['done'] else 'pending'}",
                sub['index'],
            )
            for sub in section['subsections']
        ]
        self.subsection_dropdown.options = subsection_options
        self.subsection_dropdown.value = s.subsection_index
        self.btn_prev_subsection.disabled = s.subsection_index == 0
        self.btn_next_subsection.disabled = s.subsection_index == subsection_count - 1

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
        spine_options = [
            (f"Spine {sp['index'] + 1} — {sp['type']} — {self.state.spine_validity(sp)}", sp['index'])
            for sp in section['spines']
        ]
        self.spine_dropdown.options = spine_options
        self.spine_dropdown.value = self.state.spine_index
        self.btn_prev_spine.disabled = self.state.spine_index == 0
        self.btn_next_spine.disabled = self.state.spine_index == spine_count - 1

        self.spine_review_header.value = f"""
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
            elif answer == 'no':
                no_btn.add_class('sv-active')

        self.btn_show_structure.description = (
            'Hide spine structure' if self.state.show_structure else 'Show spine structure'
        )

    def _refresh_status(self, section, checked_count, spine_count, subsections_done, subsection_count,
                         missing_total, flag_count):
        s = self.state
        section_checked = s.section_is_checked(section)
        self.status_html.value = f"""
        <div class="sv-status-row"><span>Section checked</span><span class="v">{'Yes' if section_checked else 'No'}</span></div>
        <div class="sv-status-row"><span>Subsections reviewed</span><span class="v">{subsections_done} of {subsection_count}</span></div>
        <div class="sv-status-row"><span>Missing spines</span><span class="v sv-flag">{missing_total}</span></div>
        <div class="sv-status-row"><span>Spines checked</span><span class="v">{checked_count} of {spine_count}</span></div>
        <div class="sv-status-row"><span>Flags raised on this spine</span><span class="v sv-flag">{flag_count}</span></div>
        """
        self.ready_html.value = f'<div class="sv-ready">{s.status_message}</div>'

    def _refresh_projection(self):
        for code, (cell, _btn) in self.proj_cells.items():
            if code == self.state.current_view:
                cell.add_class('sv-active')
            else:
                cell.remove_class('sv-active')
        self.btn_white_background.description = (
            'White background' if self.state.white_background else 'Dark background'
        )


def build_sample_viewport(seed=3, height=420):
    """Build a throwaway k3d plot with a random dendrite-like backbone and
    scattered "spine" points, just to preview how a real k3d viewport would
    sit underneath this control panel. Not driven by the panel's state.
    """
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
    )
    plot += k3d.line(backbone, color=0x14171C, width=0.08, shader='mesh')
    plot += k3d.points(
        spine_points, colors=colors, point_size=0.6, shader='3d',
    )
    return plot


# ============================================================
# Public entry points
# ============================================================

def build_preview(fonts_dir=None):
    """Build the preview widget tree (CSS is injected as a side effect)."""
    fonts_dir = FONT_DIRECTORY if fonts_dir is None else Path(fonts_dir)
    display(HTML(_build_css(fonts_dir)))
    app = SpineValidationDesign()
    return app.root


def display_preview(fonts_dir=None, show_sample_viewport=True):
    """Inject styling and display the redesigned control panel with mock data.

    With ``show_sample_viewport`` (default), a random k3d plot is rendered
    below the panel, purely to check how a real viewport would look and
    behave alongside this layout.
    """
    display(build_preview(fonts_dir=fonts_dir))
    if show_sample_viewport:
        display(build_sample_viewport())

"""Code behind simulation_analysis_dataframes.ipynb: load any simulation campaign of the platform,
give access to its spikes, and plot them.

The notebook calls `load`, `describe`, `neuron_table`, `summary`, and the `RasterApp` methods
`explorer`, `statistics`, `overview` and `group_table`; the bluepysnap notebook calls `selection_panel`
in place of `statistics`. Every plot of the app follows one shared selection of neurons (by property
and/or by id). Without a client, `load` makes one with `connect`.

Any circuit simulation works: point-neuron (Brian2) or morphologically detailed (NEURON), one or
several node populations, any stimuli and reports, and a scan over any number of parameters.
Virtual populations are inputs, not simulated neurons: the spikes a stimulus replays into them are
shown with the stimuli, apart from the simulated neurons' spikes.

In the platform's sandboxes, the files of entities are mounted under /data: with a client that has
a LocalAssetStore there, files are linked rather than downloaded, so large campaigns cost no download.
"""

import hashlib
import json
import logging
import math
import os
import re
import shutil
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache, partial
from pathlib import Path

import h5py
import httpx2
import ipywidgets as widgets
import libsonata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# entitysdk logs every request, signed download URLs included; keep them out of the outputs.
logging.getLogger("httpx").setLevel(logging.WARNING)

_HERE = Path(__file__).resolve().parent
# Files go to HOME/data, saved figures to HOME/plots: the notebook's folder, which is the working
# directory when the platform extracted this file into a folder next to the notebook.
HOME = Path.cwd() if _HERE.parent == Path.cwd().resolve() else _HERE
MOUNT = Path("/data")  # where the platform's sandboxes mount the files of entities
DOMAINS = {"staging": "staging.openbraininstitute.org", "production": "www.openbraininstitute.org"}
API_HOSTS = {"staging": "https://staging.cell-a.openbraininstitute.org",
             "production": "https://cell-a.openbraininstitute.org"}
MANIFEST_VERSION = 2  # 2 added replayed spikes and the circuit's population types
REPORT_LIMIT_MB = 100  # larger report files are downloaded only with load(..., reports=True); mounted ones are linked
MAX_OVERVIEW = 24  # simulations drawn in the overview
MAX_GRID = 36  # simulations drawn in the overview of a scan over two parameters, a table of rasters
MAX_TRACES = 50  # traces drawn per report
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)
GROUP_BY = ["super_class", "mtype", "cell_class", "layer", "synapse_class", "etype", "region"]
INPUT_FIELDS = {"module", "input_type", "node_set", "delay", "duration", "dt",
                "represents_physical_electrode", "spike_file"}  # everything else is a parameter
UNITS = {"amp_start": "nA", "amp_end": "nA", "mean": "nA", "sigma": "nA", "frequency": "Hz",
         "rate": "Hz", "width": "ms", "percent_start": "%", "percent_end": "%"}

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#a3a19b"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 10, "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2, "font.size": 9,
    "legend.frameon": False, "grid.color": GRID, "grid.linewidth": 0.8,
})
STYLE = {"description_width": "initial"}


def _fmt(value):
    return f"{value:g}" if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)


def _plural(n, word):
    return f"{n:,} {word}{'' if n == 1 else 's'}"


# --- Connecting to the platform ------------------------------------------------------------------


def current_environment():
    """'staging' or 'production': the platform this sandbox belongs to."""
    return os.environ.get("OBI_ENVIRONMENT", "production")


_TOKENS = {}  # environment -> a token given to connect()


def _bearer(environment):
    if environment in _TOKENS:
        return _TOKENS[environment]
    from obi_auth import get_token

    return get_token(environment=environment, auth_mode="daf")


class _RetryTransport(httpx2.HTTPTransport):
    """Asks again when the platform answers a GET with 502, 503 or 504: it is sometimes
    unavailable for a few seconds."""

    def handle_request(self, request):
        for wait in (1, 2, 4, 8):
            response = super().handle_request(request)
            if request.method not in ("GET", "HEAD") or response.status_code not in (502, 503, 504):
                return response
            response.close()
            time.sleep(wait)
        return super().handle_request(request)


def connect(environment=None, token=None, local_store=None):
    """An entitysdk client for the platform. If you need to log in, click the link that appears. In
    the platform's sandboxes, it links files from the mounted file system rather than download them."""
    from entitysdk import Client, LocalAssetStore
    from entitysdk.token_manager import TokenFromFunction

    if local_store is None and MOUNT.is_dir():
        local_store = LocalAssetStore(prefix=str(MOUNT))
    environment = environment or current_environment()
    if token:
        _TOKENS[environment] = token
    _bearer(environment)  # log in now rather than in the middle of a download
    return Client(environment=environment, http_client=httpx2.Client(transport=_RetryTransport()),
                  token_manager=TokenFromFunction(lambda: _bearer(environment)), local_store=local_store)


def _projects(environment, token=None):
    """(virtual lab id, project id) of every project you belong to."""
    with httpx2.Client(transport=_RetryTransport()) as http:
        response = http.get(f"{API_HOSTS[environment]}/api/virtual-lab-manager/virtual-labs/projects",
                            params={"size": 100}, timeout=30,
                            headers={"authorization": f"Bearer {token or _bearer(environment)}"})
    response.raise_for_status()
    return [(p["virtual_lab_id"], p["id"]) for p in response.json()["data"]["results"]]


def _token_of(client):
    """The token an entitysdk client signs its requests with, so that one login serves every request."""
    try:
        return client._token_manager.get_token()
    except Exception:
        return None


def _find(client, environment, entity_id, hint):
    """The campaign or simulation with this id, and the project context that can read it. Looks in
    the project a link names, the client's project, the sandbox's, public entities, then every other
    project you belong to."""
    from entitysdk import ProjectContext, models
    from entitysdk.exception import EntitySDKError

    def contexts():
        own = (os.environ.get("OBI_VLAB_ID"), os.environ.get("OBI_PROJECT_ID"))
        chosen = getattr(client, "project_context", None)
        chosen = (str(chosen.virtual_lab_id), str(chosen.project_id)) if chosen else None
        tried = [pair for pair in dict.fromkeys([hint, chosen, own]) if pair and all(pair)]
        for vlab, project in tried:
            yield ProjectContext(virtual_lab_id=vlab, project_id=project)
        yield None  # public entities
        for vlab, project in _projects(environment, _token_of(client)):
            if (vlab, project) not in tried:
                yield ProjectContext(virtual_lab_id=vlab, project_id=project)

    for context in contexts():
        for entity_type in (models.SimulationCampaign, models.Simulation):
            try:
                return client.get_entity(entity_id=entity_id, entity_type=entity_type,
                                         project_context=context), context
            except EntitySDKError as error:
                if "404" not in str(error):
                    raise
    raise LookupError(f"No simulation campaign or simulation {entity_id} in your projects on {environment}")


def _parse(entity):
    """The id in an id or a platform link, and the (virtual lab, project) the link names, if any."""
    ids = UUID.findall(str(entity))
    if not ids:
        raise ValueError(f"No id in {entity!r}: give a campaign or simulation id, or its platform link")
    lab = re.search(rf"virtual-lab/({UUID.pattern})/({UUID.pattern})", str(entity), re.IGNORECASE)
    return ids[-1].lower(), (lab.group(1).lower(), lab.group(2).lower()) if lab else None


def link(environment, entity_id):
    return f"https://{DOMAINS[environment]}/app/entity/{entity_id}"


# --- Downloading ---------------------------------------------------------------------------------


def _resolve(path, manifest):
    """A circuit config path with its $VARIABLES expanded, relative to the circuit's folder."""
    for _ in range(5):
        for key, value in manifest.items():
            path = path.replace(key, value)
    return os.path.normpath(path)


def _download(entity_id, hint, folder, client, environment, reports):
    """Fetch the configs, spikes and reports of every simulation that has a result, plus the
    circuit's node files and node sets, and describe them in folder/manifest.json. A file that the
    platform mounts in this sandbox is linked rather than downloaded."""
    from entitysdk import models
    from entitysdk.exception import EntitySDKError
    from entitysdk.types import FetchFileStrategy
    from entitysdk.utils.asset import resolve_asset_path

    entity, context = _find(client, environment, entity_id, hint)
    if isinstance(entity, models.Simulation):
        simulations = [entity]
        campaign = client.get_entity(entity_id=entity.simulation_campaign_id,
                                     entity_type=models.SimulationCampaign, project_context=context)
    else:
        campaign = entity
        simulations = list(client.search_entity(entity_type=models.Simulation, project_context=context,
                                                query={"simulation_campaign_id": campaign.id}))

    store = getattr(client, "_local_store", None)  # the mounted file system, in the platform's sandboxes
    fetched = {"linked": 0, "downloaded": 0, "MB": 0.0}

    def fetch(owner, owner_type, asset, target, size, asset_path=None, limit_mb=None):
        """Link the file when it is mounted, else download it unless it is over limit_mb. False if it
        was left out."""
        if target.is_symlink() and not target.exists():  # a link into a mount this sandbox does not have
            target.unlink()
        if target.exists() and target.stat().st_size == size:
            return True
        mounted = store is not None and store.path_exists(resolve_asset_path(asset, directory_file=asset_path))
        if not mounted and limit_mb is not None and size > limit_mb * 1e6:
            return False
        if target.is_symlink() or target.exists():
            target.unlink()  # an out-of-date copy
        target.parent.mkdir(parents=True, exist_ok=True)
        client.fetch_file(entity_id=owner.id, entity_type=owner_type, asset_id=asset, output_path=target,
                          asset_path=asset_path, project_context=context,
                          strategy=FetchFileStrategy.link_or_download)
        if target.is_symlink():
            fetched["linked"] += 1
        else:
            fetched["downloaded"] += 1
            fetched["MB"] += size / 1e6
        return True

    print(f"Getting the files of '{campaign.name}': {_plural(len(simulations), 'simulation')}")
    entries, skipped, replayed = [], [], set()
    for sim in simulations:
        sim_dir = folder / f"simulation_{str(sim.id)[:8]}"
        for asset in sim.assets:
            if asset.label in ("sonata_simulation_config", "custom_node_sets", "replay_spikes"):
                fetch(sim, models.Simulation, asset, sim_dir / asset.path, asset.size)
            if asset.label == "replay_spikes":  # spikes that a stimulus replays, usually of virtual neurons
                with h5py.File(sim_dir / asset.path) as f:
                    replayed |= set(f.get("spikes", {}))
        result = client.search_entity(entity_type=models.SimulationResult, project_context=context,
                                      query={"simulation_id": sim.id}).first()
        spike_file, report_files = None, {}
        for asset in (result.assets if result else []):
            if asset.is_directory:
                continue
            if asset.label == "spike_report":
                spike_file = asset.path
                fetch(result, models.SimulationResult, asset, sim_dir / asset.path, asset.size)
                continue
            report_files[asset.path] = asset.size
            if reports is not False and not fetch(result, models.SimulationResult, asset, sim_dir / asset.path,
                                                  asset.size, limit_mb=None if reports is True else REPORT_LIMIT_MB):
                skipped.append(f"{asset.path} ({asset.size / 1e6:,.0f} MB)")
        entries.append({
            "id": str(sim.id), "name": sim.name, "scan_parameters": sim.scan_parameters or {},
            "dir": sim_dir.name, "result_id": str(result.id) if result else None,
            "spike_file": spike_file, "report_files": report_files, "circuit_id": str(sim.entity_id),
        })
    if skipped:
        print(f"Not downloaded, over {REPORT_LIMIT_MB} MB each and not mounted here: "
              f"{', '.join(sorted(set(skipped)))}. Pass reports=True to load() to get them.")

    # The circuit: only its config, node files and node sets; never the (large) edge files.
    circuit_entry = None
    try:
        circuit = client.get_entity(entity_id=entries[0]["circuit_id"] if entries else campaign.entity_id,
                                    entity_type=models.Circuit, project_context=context)
    except EntitySDKError as error:
        if "404" not in str(error):
            raise
        circuit = None  # e.g. a single-neuron model rather than a circuit: no node properties
    directory = next((a for a in circuit.assets if a.label == "sonata_circuit"), None) if circuit else None
    if directory:
        circuit_dir = folder.parent / "circuits" / str(circuit.id)
        listing = client.list_directory(entity_id=circuit.id, entity_type=models.Circuit,
                                        asset_id=directory.id, project_context=context).files

        def get(relative):
            target = circuit_dir / relative
            fetch(circuit, models.Circuit, directory, target, listing[Path(relative)].size, asset_path=relative)
            return target

        config = json.loads(get("circuit_config.json").read_text())
        variables = config.get("manifest", {})
        spiking = set()
        for entry in entries:
            if entry["spike_file"]:
                spiking |= set(_read_spike_file(folder / entry["dir"] / entry["spike_file"]))
        # Simulated populations, and virtual ones that spike, go into the node table. Of the other
        # virtual populations, only those whose spikes are replayed are read: for their size.
        populations, sources, types = {}, {}, {}
        for network in config.get("networks", {}).get("nodes", []):
            nodes_file = _resolve(network["nodes_file"], variables)
            for name, spec in network.get("populations", {}).items():
                types[name] = spec.get("type", "")
                if name in spiking or spec.get("type") != "virtual":
                    populations[name] = nodes_file
                elif name in replayed:
                    sources[name] = nodes_file
        for files in (populations, sources):
            for name, nodes_file in list(files.items()):
                if Path(nodes_file) in listing:
                    get(nodes_file)
                else:
                    del files[name]
        node_sets = _resolve(config["node_sets_file"], variables) if "node_sets_file" in config else None
        if node_sets and Path(node_sets) in listing:
            get(node_sets)
        else:
            node_sets = None
        circuit_entry = {"id": str(circuit.id), "name": circuit.name,
                         "dir": os.path.relpath(circuit_dir, folder), "populations": populations,
                         "node_sets": node_sets, "types": types, "sources": sources}
    if fetched["linked"] or fetched["downloaded"]:
        print(f"  {_plural(fetched['linked'], 'file')} linked from the mounted file system, "
              f"{fetched['downloaded']:,} downloaded ({fetched['MB']:,.1f} MB)")

    manifest = {
        "version": MANIFEST_VERSION, "environment": environment, "id": entity_id,
        "kind": "simulation" if isinstance(entity, models.Simulation) else "campaign",
        "campaign": {"id": str(campaign.id), "name": campaign.name,
                     "description": campaign.description or "",
                     "scan_parameters": campaign.scan_parameters or {}},
        "project": {"virtual_lab_id": str(context.virtual_lab_id),
                    "project_id": str(context.project_id)} if context else None,
        "circuit": circuit_entry,
        "downloaded": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "simulations": entries,
    }
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))


# --- The loaded scan -----------------------------------------------------------------------------


@dataclass(repr=False, eq=False)
class Input:
    """One stimulus of a simulation, as in its SONATA config."""

    name: str
    module: str
    input_type: str
    node_set: str
    delay: float
    duration: float
    params: dict  # the module's own parameters, e.g. amp_start, frequency, rate
    targets: np.ndarray  # the neurons it targets, as rows of scan.nodes
    sources: dict = field(default_factory=dict)  # population -> (node ids, times): the spikes it replays

    @property
    def end(self):
        return self.delay + self.duration

    @property
    def replayed_times(self):
        """The times (ms) of the spikes it replays, sorted."""
        return np.sort(np.concatenate([t for _, t in self.sources.values()])) if self.sources else np.array([])

    @property
    def lane(self):
        """Inputs repeated at several times (Step_0, Step_1, ...) share one lane in the explorer."""
        return re.sub(r"_\d+$", "", self.name), self.module, self.node_set

    def amplitude(self):
        values = [abs(v) for k, v in self.params.items()
                  if k in ("amp_start", "amp_end", "percent_start", "percent_end") and isinstance(v, (int, float))]
        return max(values) if values else None

    def waveform(self, t):
        """The input's time course at times t (ms), in its own units, or None if it has no simple shape."""
        p, on = self.params, (t >= self.delay) & (t < self.end)
        if self.module in ("linear", "relative_linear"):
            key = "amp" if self.module == "linear" else "percent"
            start = p.get(f"{key}_start", 0.0)
            end = p.get(f"{key}_end", start)
            return np.where(on, start + (end - start) * (t - self.delay) / max(self.duration, 1e-9), 0.0)
        if self.module == "pulse" and p.get("frequency"):
            period = 1000 / p["frequency"]
            pulse = ((t - self.delay) % period) < p.get("width", period / 2)
            return np.where(on & pulse, p.get("amp_start", 1.0), 0.0)
        if self.module == "sinusoidal" and p.get("frequency"):
            phase = 2 * np.pi * p["frequency"] * (t - self.delay) / 1000
            return np.where(on, p.get("amp_start", 1.0) * np.sin(phase), 0.0)
        return None

    def spans(self):
        """When it pushes: the positive half-cycles of a sinusoid, the pulses of a pulse train, the
        bursts of the spikes it replays (spikes less than 5 ms apart), else its window."""
        p = self.params
        if self.sources:
            times = self.replayed_times
            if not len(times):
                return []
            breaks = np.flatnonzero(np.diff(times) > 5.0)
            starts, ends = times[np.r_[0, breaks + 1]], times[np.r_[breaks, len(times) - 1]]
            return [(a, max(b, a + 1.0)) for a, b in zip(starts, ends)]
        if self.module in ("sinusoidal", "pulse") and p.get("frequency"):
            period = 1000 / p["frequency"]
            width = period / 2 if self.module == "sinusoidal" else p.get("width", period / 2)
            if self.duration / period <= 200:
                return [(s, min(s + width, self.end)) for s in np.arange(self.delay, self.end, period)]
        return [(self.delay, self.end)]

    def describe(self):
        details = [f"{k} {_fmt(v)}{' ' + UNITS[k] if k in UNITS else ''}" for k, v in self.params.items()
                   if isinstance(v, (int, float, str)) and not isinstance(v, bool)]
        for population, (ids, times) in self.sources.items():
            details.append(f"{_plural(len(times), 'spike')} of {_plural(len(np.unique(ids)), 'neuron')} of {population}")
        return ", ".join([self.module, *details])


@dataclass(repr=False, eq=False)
class Report:
    """One report (recording) of a simulation."""

    name: str
    cells: str  # the node set recorded
    variable: str
    unit: str
    file: Path
    size: int  # bytes, on the platform
    traces: object = None  # optional reader for report_traces: (neurons, max_traces) -> its return value

    @property
    def downloaded(self):
        """The file is here: linked from the mounted file system, or downloaded."""
        return self.file.exists()


@dataclass(repr=False, eq=False)
class Simulation:
    """One simulation: its parameters, config, stimuli, reports and spikes."""

    label: str
    params: dict  # scan parameter -> value
    id: str
    name: str
    result_id: str
    folder: Path
    spike_file: Path
    config: dict  # the SONATA simulation config
    neurons: np.ndarray  # the neuron of each spike, as a row of scan.nodes
    times: np.ndarray  # the time of each spike (ms)
    inputs: list
    reports: dict  # name -> Report
    stimulated: np.ndarray  # neurons targeted by at least one input, as rows of scan.nodes

    def __repr__(self):
        return f"<Simulation {self.label}: {_plural(len(self.times), 'spike')}, {self.tstop:g} ms>"

    @property
    def tstop(self):
        return float(self.config["run"]["tstop"])

    @property
    def windows(self):
        """The time windows (ms) during which at least one input is on."""
        return _merge([(max(0.0, i.delay), min(i.end, self.tstop)) for i in self.inputs])

    @property
    def stimulus_end(self):
        return max((end for _, end in self.windows), default=0.0)

    def shading(self):
        """Spans to shade behind rasters: when inputs push, leaving out those on for the whole run.
        A replay input counts as on for the whole run when its bursts of spikes cover nearly all of it."""
        spans = []
        for i in self.inputs:
            mine = _merge([(max(0.0, a), min(b, self.tstop)) for a, b in i.spans()])
            on = sum(b - a for a, b in mine) if i.sources else i.duration
            if on < 0.99 * self.tstop:
                spans += mine
        shortest = self.tstop / 200  # so that a single replayed spike still shows
        return _merge([(a, max(b, a + shortest)) for a, b in spans])


def _merge(spans):
    merged = []
    for start, end in sorted(s for s in spans if s[1] > s[0]):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [tuple(s) for s in merged]


@dataclass(repr=False, eq=False)
class Scan:
    """The spikes of every simulation in a campaign, and the circuit's node properties."""

    name: str
    description: str
    environment: str
    entity_id: str  # what was loaded: a campaign or one simulation
    campaign_id: str
    project: object  # the entitysdk ProjectContext that reads the campaign (None if public)
    circuit: dict  # id and name, or None
    nodes: pd.DataFrame  # one row per neuron: population, node_id and the circuit's node properties
    properties: list  # categorical node properties for selecting and grouping neurons
    id_columns: list  # node properties that name single neurons: an integer unique to each, e.g. a database id
    sims: list
    scan_keys: list  # scan parameters, in full ("stimuli.Name.parameter")
    short_keys: dict  # scan parameter -> the shortest unambiguous name
    plots_dir: Path
    missing: list  # names of simulations that have no result yet
    offsets: dict  # population -> its first row in nodes
    population_types: dict = field(default_factory=dict)  # every node population of the circuit -> its type
    source_sizes: dict = field(default_factory=dict)  # population whose spikes are replayed -> its size
    spike_counts: pd.DataFrame = field(init=False)  # neuron x simulation, neurons that fire only
    stimulus_counts: pd.DataFrame = field(init=False)  # the same, during the stimulus windows
    responders: np.ndarray = field(init=False)  # neurons that fire in at least one simulation
    stimulated: np.ndarray = field(init=False)  # neurons targeted by an input in any simulation

    def __post_init__(self):
        self.responders = np.unique(np.concatenate([s.neurons for s in self.sims])) if self.sims else np.array([], int)
        def counts(select):
            columns = {}
            for sim in self.sims:
                positions = np.searchsorted(self.responders, sim.neurons[select(sim)])
                columns[sim.label] = np.bincount(positions, minlength=len(self.responders))
            return pd.DataFrame(columns, index=self.responders)

        self.spike_counts = counts(lambda sim: np.ones(len(sim.times), dtype=bool))
        self.stimulus_counts = counts(lambda sim: self._in_windows(sim, sim.times))
        self.stimulated = (np.unique(np.concatenate([s.stimulated for s in self.sims]))
                           if self.sims else np.array([], int))

    def __repr__(self):
        return (f"<Scan '{self.name}': {_plural(len(self.sims), 'simulation')}, "
                f"{_plural(len(self.nodes), 'neuron')}>")

    def _in_windows(self, sim, times):
        windows = sim.windows or [(0.0, sim.tstop)]
        return np.logical_or.reduce([(times >= a) & (times < b) for a, b in windows])

    def stimulus_duration(self, sim):
        """Total time (ms) with at least one input on; the whole run when there is no input."""
        return sum(b - a for a, b in sim.windows) or sim.tstop

    @property
    def focal(self):
        """True when the inputs target a minority of the neurons, so 'stimulated' is a useful group."""
        return 0 < len(self.stimulated) < 0.5 * len(self.nodes)

    @property
    def group_by(self):
        """A good default property to group neurons by."""
        preferred = [p for p in GROUP_BY if p in self.properties]
        if preferred:
            return preferred[0]
        return min(self.properties, key=lambda p: self.nodes[p].nunique())

    def populations(self):
        """One row per node population of the circuit: its type, its size, and what is done with it.
        Virtual populations are not simulated; a stimulus may replay spikes from them."""
        replayed_by = {}
        for sim in self.sims:
            for i in sim.inputs:
                for population in i.sources:
                    replayed_by.setdefault(population, {})[i.name] = None
        sizes = self.nodes["population"].value_counts()
        rows = []
        for name in dict.fromkeys([*self.population_types, *self.offsets]):
            kind = self.population_types.get(name, "")
            if name in self.offsets:
                use = "virtual, but its spikes are recorded: analysed" if kind == "virtual" else "simulated; its spikes are analysed"
            elif name in replayed_by:
                use = f"not simulated; spikes replayed by {', '.join(replayed_by[name])}, shown with the stimuli"
            elif kind == "virtual":
                use = "not simulated; no spikes replayed"
            else:
                use = "no node file: left out"
            size = sizes.get(name, self.source_sizes.get(name))
            rows.append({"population": name, "type": kind or "?", "neurons": "" if size is None else f"{size:,}",
                         "in this notebook": use})
        return pd.DataFrame(rows, columns=["population", "type", "neurons", "in this notebook"])

    def simulation(self, key):
        """A simulation by position, label or id."""
        if isinstance(key, Simulation):
            return key
        if isinstance(key, (int, np.integer)):
            return self.sims[key]
        return next(s for s in self.sims if key in (s.label, s.id))

    def link(self, entity_id=None):
        return link(self.environment, entity_id or self.entity_id)

    def neuron_name(self, row):
        """How a neuron is written in the id box: its node id, or population:node_id."""
        population, node_id = self.nodes["population"].iat[row], int(self.nodes["node_id"].iat[row])
        return node_id if len(self.offsets) == 1 else f"{population}:{node_id}"

    def find(self, tokens, column="node_id"):
        """Neurons (rows of nodes) whose integer `column` holds the values listed, and the tokens not
        recognised. Node ids may be written population:node_id. Takes a list or a string of tokens."""
        if isinstance(tokens, str):
            tokens = re.findall(r"[^\s,;:]+:\d+|-?\d+", tokens)
        tokens = list(map(str, tokens))
        if column != "node_id":
            values = self.nodes[column].to_numpy()
            limits = np.iinfo(values.dtype)
            numbers = {t: int(t) for t in tokens if re.fullmatch(r"-?\d+", t) and limits.min <= int(t) <= limits.max}
            hit = np.isin(values, np.array(list(numbers.values()), dtype=values.dtype))  # same dtype: exact for 64-bit ids
            present = set(values[hit].tolist())
            return np.flatnonzero(hit), [t for t in tokens if numbers.get(t) not in present]
        sizes = self.nodes["population"].value_counts()
        found, unknown = [], []
        for token in tokens:
            population, _, number = token.rpartition(":")
            if not number.isdigit():
                unknown.append(token)
                continue
            number = int(number)
            if population:
                rows = [self.offsets[population] + number] if number < sizes.get(population, 0) else []
            else:
                rows = [offset + number for p, offset in self.offsets.items() if number < sizes[p]]
            if rows:
                found.extend(rows)
            else:
                unknown.append(token)
        return np.unique(np.asarray(found, dtype=np.int64)), unknown

    def top_neurons(self, n=5):
        """The n neurons with the most spikes over all simulations, not counting stimulated ones when others fire."""
        totals = self.spike_counts.sum(axis=1)
        others = totals[~np.isin(totals.index, self.stimulated)]
        picked = (others if len(others) else totals).sort_values(ascending=False, kind="stable").head(n)
        return [self.neuron_name(row) for row in picked.index]

    def spikes(self, simulation=None, properties=()):
        """One row per spike: simulation, population, node_id, time_ms, and the node properties named."""
        sims = self.sims if simulation is None else [self.simulation(simulation)]
        neurons = np.concatenate([s.neurons for s in sims]) if sims else np.array([], int)
        table = pd.DataFrame({
            "simulation": pd.Categorical(np.repeat([s.label for s in sims], [len(s.times) for s in sims]),
                                         categories=[s.label for s in self.sims]),
            "population": self.nodes["population"].to_numpy()[neurons],
            "node_id": self.nodes["node_id"].to_numpy()[neurons],
            "time_ms": np.concatenate([s.times for s in sims]) if sims else np.array([]),
        })
        for name in properties:
            table[name] = self.nodes[name].iloc[neurons].to_numpy()
        return table


def _intact(folder, manifest):
    """No link to a fetched file dangles. Links into the mounted file system resolve only in a
    sandbox that mounts the same files: a project's private files are mounted only in its own."""
    circuit = manifest["circuit"]
    folders = [folder, *([folder / circuit["dir"]] if circuit else [])]
    return not any(p.is_symlink() and not p.exists() for f in folders if f.exists() for p in f.rglob("*"))


def load(entity, client=None, refresh=False, reports="auto"):
    """Load a simulation campaign, or a single simulation, given its id or its platform link.

    The files go to data/<id>/ next to the notebook, once. In the platform's sandboxes, a file that
    is mounted there is linked rather than downloaded. Simulations that had no result yet are looked
    for again on every load; refresh=True looks again for everything. reports="auto" downloads
    report files up to REPORT_LIMIT_MB each, True all of them, False none; mounted reports are linked
    whatever their size.
    """
    entity_id, hint = _parse(entity)
    folder = HOME / "data" / entity_id
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    environment = manifest["environment"] if manifest else current_environment()
    stale = (manifest is None or refresh or manifest.get("version", 1) < MANIFEST_VERSION
             or any(not s["result_id"] for s in manifest["simulations"])
             or not _intact(folder, manifest)
             or (reports is True and any(not (folder / s["dir"] / name).exists()
                                         for s in manifest["simulations"] for name in s["report_files"])))
    if stale:
        try:
            _download(entity_id, hint, folder, client or connect(environment), environment, reports)
        except Exception as error:
            if manifest is None:
                raise
            print(f"Could not check the platform for new results ({type(error).__name__}: {error}); "
                  f"using the files fetched on {manifest['downloaded'][:10]}")
        manifest = json.loads(manifest_path.read_text())
    scan = _build(manifest, folder)
    _print_loaded(scan)
    return scan


def _sortable(value):
    return (0, float(value), "") if isinstance(value, (int, float)) and not isinstance(value, bool) else (1, 0.0, str(value))


def _short_keys(keys):
    """The shortest dotted suffix of each scan parameter that no other parameter ends with."""
    short = {}
    for key in keys:
        parts = key.split(".")
        for n in range(1, len(parts) + 1):
            suffix = ".".join(parts[-n:])
            if not any(other != key and (other == suffix or other.endswith("." + suffix)) for other in keys):
                short[key] = suffix
                break
        else:
            short[key] = key
    return short


def _read_nodes(storage_path, population):
    """The node table of one population: node_id and every attribute; text attributes as categories."""
    pop = libsonata.NodeStorage(str(storage_path)).open_population(population)
    selection = pop.select_all()
    table = {"node_id": np.arange(pop.size)}
    for name in sorted(pop.attribute_names):
        if name in pop.enumeration_names:
            labels = pop.enumeration_values(name)
            codes = pop.get_enumeration(name, selection).astype(np.int64)
            if len(set(labels)) == len(labels):
                table[name] = pd.Categorical.from_codes(codes, categories=labels)
                continue
            table[name] = pd.Categorical(np.asarray(labels, dtype=object)[codes])
            continue
        values = pop.get_attribute(name, selection)
        table[name] = pd.Categorical(values) if values.dtype.kind in "OSU" else values
    return pop, pd.DataFrame(table)


def _read_spike_file(path):
    """{population: (node ids, times in ms)} from a SONATA spike file."""
    with h5py.File(path) as f:
        spikes = f.get("spikes")
        if spikes is None:
            return {}
        if isinstance(spikes.get("timestamps"), h5py.Dataset):  # older files have no population groups
            ids = spikes["node_ids"] if "node_ids" in spikes else spikes["gids"]
            return {"default": (ids[:].astype(np.int64), spikes["timestamps"][:].astype(float))}
        return {name: (group["node_ids"][:].astype(np.int64), group["timestamps"][:].astype(float))
                for name, group in spikes.items() if isinstance(group, h5py.Group) and "timestamps" in group}


def _with_bare_populations(tables, spikes):
    """The circuit's node tables, plus a bare one for any spiking population the circuit lacks."""
    known = {name for name, _ in tables}
    tables = list(tables)
    for name in sorted({p for s in spikes for p in s} - known):
        sizes = [int(s[name][0].max()) + 1 for s in spikes if name in s and len(s[name][0])]
        tables.append((name, pd.DataFrame({"node_id": np.arange(max(sizes, default=0))})))
    return tables


def _node_table(tables):
    """The populations' node tables as one, the first row of each population, the categorical
    properties for selecting and grouping neurons, and the columns that name single neurons."""
    offsets, start = {}, 0
    for name, table in tables:
        offsets[name] = start
        start += len(table)
    nodes = (pd.concat([t.assign(population=name) for name, t in tables], ignore_index=True) if tables
             else pd.DataFrame({"population": pd.Series([], dtype=object), "node_id": np.array([], dtype=np.int64)}))
    for column in nodes.columns:  # concatenating populations turns categories into objects
        if nodes[column].dtype == object:  # a property some populations lack is "(none)" in their neurons
            nodes[column] = pd.Categorical(nodes[column].astype(object).where(nodes[column].notna(), "(none)").astype(str))
        elif isinstance(nodes[column].dtype, pd.CategoricalDtype) and nodes[column].isna().any():
            if "(none)" not in nodes[column].cat.categories:
                nodes[column] = nodes[column].cat.add_categories("(none)")
            nodes[column] = nodes[column].fillna("(none)")
    nodes = nodes[["population", "node_id", *[c for c in nodes.columns if c not in ("population", "node_id")]]]

    properties, id_columns = [], []
    for column in nodes.columns:
        if column == "node_id":
            continue
        n_values = nodes[column].nunique()
        if isinstance(nodes[column].dtype, pd.CategoricalDtype):
            if 2 <= n_values and (n_values <= 0.5 * len(nodes) or n_values <= 50):
                properties.append(column)
        elif nodes[column].dtype.kind in "iu":
            if n_values == len(nodes) and "id" in column:
                id_columns.append(column)
            elif 2 <= n_values <= 50:
                nodes[column] = pd.Categorical(nodes[column])
                properties.append(column)
    return nodes, offsets, properties or ["population"], id_columns


def _scan_keys(manifest, entries):
    """The scan parameters, and the shortest unambiguous name of each."""
    scan_keys = list((manifest["campaign"]["scan_parameters"] or {}).keys())
    for entry in entries:
        scan_keys += [k for k in entry["scan_parameters"] if k not in scan_keys]
    return scan_keys, _short_keys(scan_keys)


def _simulation(entry, spikes, offsets, config, targets, reports, scan_keys, short, folder, spike_file):
    """One Simulation, from its spikes ({population: (node ids, times)}), its SONATA config, its
    reports, and targets(node_set): the neurons of a node set, as rows of the node table."""
    neurons = [ids + offsets[p] for p, (ids, _) in spikes.items()]
    times = [t for _, t in spikes.values()]
    neurons = np.concatenate(neurons) if neurons else np.array([], dtype=np.int64)
    times = np.concatenate(times) if times else np.array([])
    order = np.argsort(times, kind="stable")
    config.setdefault("run", {}).setdefault("tstop", float(np.ceil(times.max())) if len(times) else 1.0)

    def replayed(spec):
        """The spikes an input replays, read from its spike file when that was downloaded."""
        path = folder / spec["spike_file"] if spec.get("spike_file") else None
        return _read_spike_file(path) if path is not None and path.is_file() else {}

    inputs = [Input(
        name=name, module=spec.get("module", "?"), input_type=spec.get("input_type", "?"),
        node_set=spec.get("node_set", config.get("node_set", "")),
        delay=float(spec.get("delay", 0.0)), duration=float(spec.get("duration", config["run"]["tstop"])),
        params={k: v for k, v in spec.items() if k not in INPUT_FIELDS},
        targets=targets(spec.get("node_set", config.get("node_set", ""))), sources=replayed(spec),
    ) for name, spec in (config.get("inputs") or {}).items()]
    stimulated = [i.targets for i in inputs]
    params = {k: entry["scan_parameters"][k] for k in scan_keys if k in entry["scan_parameters"]}
    return Simulation(
        label=", ".join(f"{short[k]}={_fmt(v)}" for k, v in params.items()) or entry["name"],
        params=params, id=entry["id"], name=entry["name"], result_id=entry["result_id"], folder=folder,
        spike_file=spike_file, config=config, neurons=neurons[order], times=times[order], inputs=inputs,
        reports=reports,
        stimulated=np.unique(np.concatenate(stimulated)) if stimulated else np.array([], dtype=np.int64),
    )


def _scan(manifest, sims, missing, nodes, offsets, properties, id_columns, scan_keys, short, source_sizes=None):
    """The Scan, with its simulations in scan order and uniquely labelled."""
    sims.sort(key=lambda s: (tuple(_sortable(s.params.get(k)) for k in scan_keys), s.name))
    for sim in sims:  # labels must be unique: they name the simulations everywhere
        same = [s for s in sims if s.label == sim.label]
        if len(same) > 1:
            for s in same:
                s.label = f"{s.label} ({s.name})" if s.label != s.name else f"{s.name} ({s.id[:8]})"
    name = manifest["campaign"]["name"]
    if manifest["kind"] == "simulation":
        name += f" › {manifest['simulations'][0]['name']}"
    return Scan(
        name=name,
        description=manifest["campaign"]["description"], environment=manifest["environment"],
        entity_id=manifest["id"], campaign_id=manifest["campaign"]["id"], project=_context(manifest["project"]),
        circuit=manifest["circuit"], nodes=nodes, properties=properties, id_columns=id_columns, sims=sims,
        scan_keys=[k for k in scan_keys if any(k in s.params for s in sims)], short_keys=short,
        plots_dir=HOME / "plots" / manifest["id"], missing=missing, offsets=offsets,
        population_types=(manifest["circuit"] or {}).get("types", {}), source_sizes=source_sizes or {},
    )


def _build(manifest, folder):
    entries = [s for s in manifest["simulations"] if s["result_id"] and s["spike_file"]]
    missing = [s["name"] for s in manifest["simulations"] if not (s["result_id"] and s["spike_file"])]
    raw = [_read_spike_file(folder / entry["dir"] / entry["spike_file"]) for entry in entries]

    circuit = manifest["circuit"]
    tables, libsonata_populations = [], {}
    if circuit:
        for name, nodes_file in circuit["populations"].items():
            pop, table = _read_nodes(folder / circuit["dir"] / nodes_file, name)
            libsonata_populations[name] = pop
            tables.append((name, table))
    nodes, offsets, properties, id_columns = _node_table(_with_bare_populations(tables, raw))

    # Node sets: the circuit's, overridden by each simulation's own.
    circuit_sets = {}
    if circuit and circuit.get("node_sets"):
        circuit_sets = json.loads((folder / circuit["dir"] / circuit["node_sets"]).read_text())

    @lru_cache(maxsize=256)
    def resolve(sets_json, name):
        try:
            sets = libsonata.NodeSets(sets_json)
        except Exception:
            return np.array([], dtype=np.int64)
        rows = []
        for population, pop in libsonata_populations.items():
            try:
                rows.append(sets.materialize(name, pop).flatten().astype(np.int64) + offsets[population])
            except Exception:
                continue  # the node set does not apply to this population
        return np.unique(np.concatenate(rows)) if rows else np.array([], dtype=np.int64)

    scan_keys, short = _scan_keys(manifest, entries)
    sims = []
    for entry, spikes in zip(entries, raw):
        sim_dir = folder / entry["dir"]
        # Simulations registered from outside the platform may have no SONATA config.
        config_path = sim_dir / "simulation_config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        sets = dict(circuit_sets)
        if (sim_dir / "node_sets.json").exists():
            sets.update(json.loads((sim_dir / "node_sets.json").read_text()))
        reports = {}
        for name, spec in (config.get("reports") or {}).items():
            file_name = spec.get("file_name", name)
            file_name = file_name if file_name.endswith(".h5") else f"{file_name}.h5"
            reports[name] = Report(name=name, cells=spec.get("cells", config.get("node_set", "")),
                                   variable=spec.get("variable_name", "?"), unit=spec.get("unit", ""),
                                   file=sim_dir / file_name, size=entry["report_files"].get(file_name, 0))
        named = {r.file.name for r in reports.values()}
        for file_name, size in entry["report_files"].items():  # result files the config does not name
            if file_name not in named and file_name.endswith(".h5"):
                reports[Path(file_name).stem] = Report(name=Path(file_name).stem, cells="?", variable="?",
                                                       unit="", file=sim_dir / file_name, size=size)
        targets = partial(resolve, json.dumps(sets, sort_keys=True))
        sims.append(_simulation(entry, spikes, offsets, config, targets, reports, scan_keys, short,
                                sim_dir, sim_dir / entry["spike_file"]))
    source_sizes = {name: libsonata.NodeStorage(str(folder / circuit["dir"] / nodes_file)).open_population(name).size
                    for name, nodes_file in (circuit or {}).get("sources", {}).items()}
    return _scan(manifest, sims, missing, nodes, offsets, properties, id_columns, scan_keys, short, source_sizes)


def _context(project):
    if not project:
        return None
    from entitysdk import ProjectContext

    return ProjectContext(virtual_lab_id=project["virtual_lab_id"], project_id=project["project_id"])


def _print_loaded(scan):
    populations = scan.nodes["population"].value_counts()
    print(f"{scan.name}")
    print(f"  {_plural(len(scan.sims), 'simulation')}"
          + (f" (scanning {', '.join(scan.short_keys[k] for k in scan.scan_keys)})" if scan.scan_keys else "")
          + f", {_plural(sum(len(s.times) for s in scan.sims), 'spike')}")
    others = [p for p in scan.population_types if p not in scan.offsets]
    replayed = {p for s in scan.sims for i in s.inputs for p in i.sources}
    print(f"  circuit {scan.circuit['name'] if scan.circuit else '(no node properties)'}: "
          + ", ".join(f"{_plural(int(n), 'neuron')} in {p}" for p, n in populations.items())
          + ("; not simulated: " + ", ".join(
              f"{p} ({scan.population_types[p] or '?'}{', spikes replayed' if p in replayed else ''})" for p in others)
             if others else ""))
    if scan.missing:
        print(f"  {len(scan.missing)} of {len(scan.missing) + len(scan.sims)} simulations have no result yet: "
              + ", ".join(scan.missing[:5]) + (", ..." if len(scan.missing) > 5 else ""))


# --- Describing the scan -------------------------------------------------------------------------


def _table(frame):
    display(HTML(frame.to_html(escape=False, index=False, na_rep="")))


def describe(scan):
    """Tables of the simulations, their stimuli and their recordings, with links to the platform."""
    a = "<a href='{}' target='_blank'>{}</a>".format
    display(HTML(
        f"<b>{a(scan.link(scan.campaign_id), scan.name)}</b>"
        + (f"<br>{scan.description}" if scan.description else "")
        + (f"<br>Circuit: {a(scan.link(scan.circuit['id']), scan.circuit['name'])}" if scan.circuit else "")
    ))
    populations = scan.populations()
    if len(populations) > 1:
        display(HTML("<b>Node populations</b>"))
        _table(populations)
    rows = []
    for sim in scan.sims:
        rows.append({
            "simulation": a(scan.link(sim.id), sim.label),
            **{scan.short_keys[k]: _fmt(sim.params.get(k, "")) for k in scan.scan_keys},
            "length (ms)": _fmt(sim.tstop), "spikes": f"{len(sim.times):,}",
            "neurons that fire": f"{len(np.unique(sim.neurons)):,}",
        })
    display(HTML("<b>Simulations</b>"))
    _table(pd.DataFrame(rows))
    if scan.missing:
        display(HTML(f"No result yet for {_plural(len(scan.missing), 'simulation')}: {', '.join(scan.missing)}"))
    if not scan.sims:
        return
    sim = scan.sims[0]
    if sim.inputs:
        varying = [scan.short_keys[k] for k in scan.scan_keys if k.startswith(("stimuli.", "neuron_sets."))]
        display(HTML("<b>Stimuli</b>" + (f" (of the first simulation; {', '.join(varying)} "
                                         f"{'varies' if len(varying) == 1 else 'vary'})" if varying else "")))
        _table(pd.DataFrame([{
            "input": i.name, "type": i.input_type,
            "target": f"{i.node_set} ({_plural(len(i.targets), 'neuron') if scan.circuit else '? neurons'})",
            "from (ms)": _fmt(i.delay), "to (ms)": _fmt(i.end), "parameters": i.describe(),
        } for i in sim.inputs]))
    if sim.reports:
        display(HTML("<b>Recordings</b>"))
        _table(pd.DataFrame([{
            "report": r.name, "neurons": r.cells, "variable": f"{r.variable} ({r.unit})",
            "file": f"{r.file.name} ({r.size / 1e6:,.1f} MB)",
            "fetched": ("linked" if r.file.is_symlink() else "downloaded") if r.downloaded
                       else "no: load(..., reports=True)",
        } for r in sim.reports.values()]))


def neuron_table(scan, neurons, by=None, column="node_id"):
    """Spike counts and first-spike times of particular neurons in every simulation.

    neurons: values of the integer column `column` of scan.nodes; node ids may be written
    'population:node_id'. by: the property shown first.
    """
    rows, unknown = scan.find(neurons, column)
    if unknown:
        print("Not found:", ", ".join(unknown))
    if not len(rows):
        return
    names = pd.Index([str(scan.neuron_name(r)) for r in rows], name="neuron")
    by = by or scan.group_by
    shown = [*scan.id_columns, by, *[p for p in scan.properties if p != by][:3]]
    identity = scan.nodes.iloc[rows][list(dict.fromkeys(shown))].set_axis(names)
    spikes = pd.DataFrame({s.label: [int((s.neurons == r).sum()) for r in rows] for s in scan.sims}, index=names)
    first = pd.DataFrame({s.label: [s.times[s.neurons == r].min() if (s.neurons == r).any() else np.nan
                                    for r in rows] for s in scan.sims}, index=names)
    display(HTML("<b>The neurons</b>"))
    display(identity)
    display(HTML("<b>Spikes in each simulation</b>"))
    display(spikes)
    display(HTML("<b>Time of the first spike (ms)</b>"))
    display(first.round(2))


# --- The neurons and their connectivity -----------------------------------------------------------


def _ranked(values, top=len(PALETTE)):
    """The `top` most common values, most common first, and how many neurons carry each value."""
    counts = pd.Series(values).value_counts()
    return list(counts.index[:top]), counts


def plot_positions(nodes, by):
    """Where neurons are: the x, y, z positions of the rows of a node table (scan.nodes, or some of its
    rows) seen from two sides, coloured by the property `by`, the most common values each in its own colour."""
    if not {"x", "y", "z"} <= set(nodes.columns):
        print("The node table has no x, y and z positions.")
        return
    values = nodes[by].astype(str).to_numpy()
    top, counts = _ranked(values)
    groups = [(f"{v} ({counts[v]:,})", PALETTE[slot], values == v) for slot, v in enumerate(top)]
    rest = ~np.isin(values, top)
    if rest.any():
        groups.append((f"other ({len(set(values[rest]))} values, {int(rest.sum()):,})", OTHER, rest))
    size = float(np.clip(3000 / max(len(nodes), 1), 0.3, 12))  # marker area: smaller for more neurons
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    for ax, (a, b) in zip(axes, [("x", "y"), ("x", "z")]):
        for label, colour, members in groups[::-1]:  # the largest groups on top
            ax.scatter(nodes[a].to_numpy()[members], nodes[b].to_numpy()[members], s=size, color=colour,
                       lw=0, rasterized=True, label=label)
        ax.set(xlabel=a, ylabel=b, aspect="equal")
        ax.set_title(f"{b} against {a}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="outside right upper", title=f"{by} (neurons)", markerscale=max(1, 6 / size ** 0.5))
    fig.suptitle(f"{len(nodes):,} simulated neurons", x=0.01, ha="left", fontsize=9, color=INK_2)
    plt.show()


def _select(nodes, filters, rows, has_ids, match):
    """A mask over the neurons: those with the property values of filters [(property, values)],
    and those in rows when has_ids, combined with AND (match "all") or OR ("any"). Every neuron
    when there is no criterion."""
    criteria = [np.isin(nodes[prop].to_numpy(), list(values)) for prop, values in filters]
    if has_ids:
        listed = np.zeros(len(nodes), dtype=bool)
        listed[list(rows)] = True
        criteria.append(listed)
    if not criteria:
        return np.ones(len(nodes), dtype=bool)
    return np.logical_and.reduce(criteria) if match == "all" else np.logical_or.reduce(criteria)


def _selection_text(filters, n_listed, has_ids, match):
    """One line naming a selection; '' when it has no criterion."""
    parts = [f"{prop}: {', '.join(map(_fmt, values[:4]))}" + (f" (+{len(values) - 4})" if len(values) > 4 else "")
             for prop, values in filters]
    if has_ids:
        parts.append(f"{n_listed} listed neuron{'s' if n_listed != 1 else ''}")
    return (" AND " if match == "all" else " OR ").join(parts)


def property_menu(scan):
    """A menu of the node properties that can group neurons, with how many values each takes."""
    return widgets.Dropdown(options=[(f"{p} ({scan.nodes[p].nunique():,} values)", p) for p in scan.properties],
                            value=scan.group_by, description="Group by", style=STYLE)


def _id_widgets(scan):
    """The 'Select neurons by id' widgets: a menu of the integer columns of scan.nodes, node_id
    first, and a box for the values of the one chosen."""
    columns = [c for c in scan.nodes.columns if scan.nodes[c].dtype.kind in "iu"]
    column = widgets.Dropdown(options=columns, value="node_id", description="Id column", style=STYLE)
    ids = widgets.Textarea(continuous_update=False, layout=widgets.Layout(width="886px", height="56px"))

    def hint(_=None):
        names = (f"node ids{' or population:node_id' if len(scan.offsets) > 1 else ''}"
                 if column.value == "node_id" else f"values of {column.value}")
        ids.placeholder = f"{names}, separated by spaces, commas or new lines"

    column.observe(hint, names="value")
    hint()
    return column, ids


class NeuronSelector:
    """Widgets that select neurons of scan.nodes by property values and/or ids, in a panel that
    folds, with a table of the selected neurons below them. Their lists hold the values of every
    neuron, unlike RasterApp's, which hold those of the neurons that fire. `.nodes` is the selection
    as a DataFrame."""

    def __init__(self, scan, n_filters=3):
        self.scan = scan
        self._panels = []
        options = [("(choose a property)", "")] + [(p, p) for p in scan.properties]
        self.filters = [(widgets.Dropdown(options=options, layout=widgets.Layout(width="290px")),
                         widgets.SelectMultiple(options=[], rows=6, layout=widgets.Layout(width="290px")))
                        for _ in range(n_filters)]
        self.id_column, self.ids = _id_widgets(scan)
        self.match = widgets.ToggleButtons(options=[("all criteria (AND)", "all"), ("any criterion (OR)", "any")],
                                           description="Match", style=STYLE)
        self.clear = widgets.Button(description="Clear selection", icon="times")
        self.output = widgets.Output()
        for prop_w, values_w in self.filters:
            prop_w.observe(lambda change, values_w=values_w: self._fill(change["new"], values_w), names="value")
            values_w.observe(self._show, names="value")
        for w in (self.id_column, self.ids, self.match):
            w.observe(self._show, names="value")
        self.clear.on_click(self._clear)

    def _fill(self, prop, values_w):
        """List the values that `prop` takes, with how many neurons carry each."""
        counts = self.scan.nodes[prop].value_counts() if prop else pd.Series(dtype=int)
        values_w.options = [(f"{_fmt(value)} ({count:,})", value) for value, count in counts.items() if count]
        self._show()

    def _clear(self, _):
        for prop_w, _values in self.filters:
            prop_w.value = ""
        self.ids.value = ""

    def _mask(self):
        """The selection as a mask over the neurons, the ids not recognised, and a line naming it."""
        filters = [(p.value, tuple(v.value)) for p, v in self.filters if p.value and v.value]
        rows, unknown = self.scan.find(self.ids.value, self.id_column.value)
        has_ids = bool(self.ids.value.strip())
        name = _selection_text(filters, len(rows), has_ids, self.match.value) or "all neurons"
        return _select(self.scan.nodes, filters, rows, has_ids, self.match.value), unknown, name

    @property
    def nodes(self):
        """The selected neurons, as rows of scan.nodes."""
        return self.scan.nodes[self._mask()[0]]

    def _show(self, _=None):
        mask, unknown, name = self._mask()
        for accordion in self._panels:
            accordion.set_title(0, f"Select neurons: {name}")
        selected = self.scan.nodes[mask]
        n_fire = int(np.isin(np.flatnonzero(mask), self.scan.responders).sum())
        text = (f"<b>{len(selected):,} of {len(self.scan.nodes):,} neurons selected</b>, {n_fire:,} of which fire "
                "in at least one simulation")
        if unknown:
            text += f". <span style='color:#d03b3b'>Not recognised: {', '.join(unknown[:10])}</span>"
        with self.output:
            self.output.clear_output(wait=True)
            display(HTML(text))
            display(selected)

    def panel(self):
        """The widgets in a panel that folds, with the selected neurons below it."""
        heading = "<b>{}</b> <span style='color:#52514e'>{}</span>".format
        controls = widgets.VBox([
            widgets.HTML(heading("Select neurons by property", "pick a property in a box, then one or more of "
                                 "its values (ctrl-click, or ⌘-click on a Mac). The lists show every neuron's "
                                 "values, with counts.")),
            widgets.HBox([widgets.VBox([p, v]) for p, v in self.filters]),
            widgets.HTML(heading("Select neurons by id", "")),
            self.id_column,
            self.ids,
            widgets.HBox([self.match, self.clear]),
        ])
        self._panels.append(widgets.Accordion(children=[controls], selected_index=0))
        self._show()
        return widgets.VBox([self._panels[-1], self.output])


def download_circuit_connectivity(client, scan, base=None):
    """The connectivity matrices of the scan's circuit, {description: file}, fetched as in the "Load
    spiking output of a circuit simulation campaign" notebook into base/<circuit id>/ (by default
    data/circuits/<circuit id>/connectivity/). With a client that has a local store at /data, they
    are linked from the mounted file system rather than downloaded."""
    from entitysdk import models
    from entitysdk.exception import EntitySDKError
    from entitysdk.types import AssetLabel

    if not scan.circuit:
        print("The simulations have no circuit, so no connectivity matrix.")
        return {}
    folder = Path(base) / scan.circuit["id"] if base else HOME / "data" / "circuits" / scan.circuit["id"] / "connectivity"
    config = folder / "matrix_config.json"
    if not (config.exists() and all(path.exists() for path in folder.rglob("*"))):  # a dangling link: fetch again
        try:
            circuit = client.get_entity(entity_id=scan.circuit["id"], entity_type=models.Circuit,
                                        project_context=scan.project)
        except EntitySDKError:
            circuit = None
        asset = next((a for a in circuit.assets if a.label == AssetLabel.circuit_connectivity_matrices), None) if circuit else None
        if asset is None:
            print("No connectivity matrix available for the campaign circuit.")
            return {}
        shutil.rmtree(folder, ignore_errors=True)
        client.fetch_directory(entity_id=circuit.id, entity_type=models.Circuit, asset_id=asset, output_path=folder,
                               project_context=scan.project, ignore_directory_name=True)
    matrices = json.loads(config.read_text())
    return {spec["description"]: folder / spec["path"] for files in matrices.values() for spec in files.values()}


def load_connectivity(scan, path, by=None):
    """A connectivity matrix loaded with Connectome-Utilities, with the property that groups the
    neurons (by, scan.group_by by default) added from the node table if it lacks it."""
    from conntility import ConnectivityMatrix

    by = by or scan.group_by
    matrix = ConnectivityMatrix.from_h5(path)
    if by not in matrix.vertex_properties:
        population = Path(path).parent.parent.name.split("__")[0]  # the folder is named after the edge population
        if by == "population":
            values = np.full(len(matrix.vertices), population)
        else:  # neurons missing from the node table (no simulation has a result yet) become "nan"
            nodes = scan.nodes[scan.nodes["population"] == population].set_index("node_id")[by]
            values = nodes.reindex(matrix.vertices["node_ids"]).astype(str).to_numpy()
        matrix.add_vertex_property(by, values)
    return matrix


def _count_pairs(matrix, block, n, chunk=4_000_000):
    """Connections from each block of neurons (rows) to each (columns), given each neuron's block
    (0 to n - 1). In chunks of connections, to keep memory low."""
    edges, counts = matrix.matrix, np.zeros(n * n, dtype=np.int64)
    for start in range(0, edges.nnz, chunk):
        rows, cols = edges.row[start:start + chunk], edges.col[start:start + chunk]
        counts += np.bincount(block[rows] * n + block[cols], minlength=n * n)
    return counts.reshape(n, n)


def pathways(matrix, by):
    """Connections from each group of neurons (rows) to each group (columns): what matrix.condense(by)
    counts, without its memory, which for a large circuit is more than the sandbox can spare."""
    groups = pd.Categorical(matrix.vertices[by].astype(str))
    counts = _count_pairs(matrix, groups.codes.astype(np.int64), len(groups.categories))
    return pd.DataFrame(counts, index=pd.Index(groups.categories, name="from"),
                        columns=pd.Index(groups.categories, name="to"))


def plot_pathways(matrix, by):
    """The probability that a neuron of one group connects to a neuron of another: their connections
    over their pairs of neurons. The groups are in the order of the pathways table and the matrix image."""
    counts = pathways(matrix, by)
    if not counts.to_numpy().any():
        print("The matrix has no connections.")
        return
    sizes = matrix.vertices[by].astype(str).value_counts()
    order = list(counts.index)
    probability = counts / np.outer(sizes[order], sizes[order])
    side = float(np.clip(0.22 * len(order) + 3, 5, 12))
    fig, ax = plt.subplots(figsize=(side + 1.2, side), constrained_layout=True)
    image = ax.imshow(probability.where(probability > 0), cmap="magma_r", norm=LogNorm())
    ax.set_xticks(range(len(order)), order, rotation=90)
    ax.set_yticks(range(len(order)), order)
    ax.tick_params(length=0)
    ax.set(xlabel=f"Postsynaptic neurons, by {by}", ylabel=f"Presynaptic neurons, by {by}")
    ax.set_title("Connection probability between groups (white: no connection)")
    fig.colorbar(image, ax=ax, shrink=0.8, label="connections / pairs of neurons")
    plt.show()


def plot_connectivity(matrix, by, bins=600):
    """The connectivity matrix as an image, its neurons sorted by `by`, presynaptic ones as rows.
    With more neurons than bins, each pixel is a block of neurons, coloured by the fraction of its
    neuron pairs that connect."""
    groups = pd.Categorical(matrix.vertices[by].astype(str))
    n = len(groups)
    bins = min(bins, n)
    rank = np.empty(n, dtype=np.int64)
    rank[np.argsort(groups.codes, kind="stable")] = np.arange(n)
    block = rank * bins // n
    size = np.bincount(block, minlength=bins)
    density = np.ma.masked_equal(_count_pairs(matrix, block, bins) / np.outer(size, size), 0)
    fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    if bins < n:
        image = ax.imshow(density, cmap="magma_r", norm=LogNorm(), extent=(0, n, n, 0), interpolation="nearest")
        fig.colorbar(image, ax=ax, shrink=0.8, label="fraction of neuron pairs that connect")
    else:  # every neuron its own row and column: a dot per connection
        ax.imshow(density, cmap=ListedColormap([INK]), extent=(0, n, n, 0), interpolation="nearest")
    sizes = np.bincount(groups.codes, minlength=len(groups.categories))
    ends = np.cumsum(sizes)
    for end in ends[:-1]:
        ax.axhline(end, color=MUTED, lw=0.5)
        ax.axvline(end, color=MUTED, lw=0.5)
    shown = sizes >= 0.012 * n  # name the groups wide enough to read
    ax.set_xticks((ends - sizes / 2)[shown], np.asarray(groups.categories)[shown], rotation=90)
    ax.set_yticks((ends - sizes / 2)[shown], np.asarray(groups.categories)[shown])
    ax.tick_params(length=0)
    ax.set_xlabel(f"Postsynaptic neurons, by {by}")
    ax.set_ylabel(f"Presynaptic neurons, by {by}")
    ax.set_title(f"{n:,} neurons, {matrix.matrix.nnz:,} connections"
                 + (f"; a pixel is a block of about {n / bins:,.0f} × {n / bins:,.0f} neurons" if bins < n else ""))
    plt.show()


# --- Summary curves ------------------------------------------------------------------------------


def _layout(scan):
    """Where each simulation goes along the x axis of the summary curves."""
    params = pd.DataFrame([s.params for s in scan.sims], columns=scan.scan_keys)
    numeric = [k for k in scan.scan_keys
               if pd.to_numeric(params[k], errors="coerce").notna().all() and params[k].astype(str).nunique() > 1]
    if numeric:
        key = max(numeric, key=lambda k: params[k].astype(str).nunique())
        x = params[key].astype(float).to_numpy()
        others = [k for k in scan.scan_keys if k != key and params[k].astype(str).nunique() > 1]
        series = ([", ".join(f"{scan.short_keys[k]}={_fmt(s.params.get(k))}" for k in others) for s in scan.sims]
                  if others else None)
        values = np.unique(x)
        log = values.min() > 0 and values.max() / values.min() >= 8
        return {"x": x, "ticks": values, "labels": [_fmt(v) for v in values], "series": series,
                "xlabel": scan.short_keys[key] + (" (log scale)" if log else ""), "log": log}
    x = np.arange(len(scan.sims), dtype=float)
    return {"x": x, "ticks": x, "labels": [s.label for s in scan.sims], "series": None,
            "xlabel": "simulation", "log": False}


def _curves(scan, panels, path=None):
    """Side-by-side plots of one value per simulation: panels = [(values, title)]."""
    if len(scan.sims) < 2:
        return
    layout = _layout(scan)
    x = layout["x"]
    groups = (pd.Series(np.arange(len(x))).groupby(layout["series"], sort=False).indices
              if layout["series"] else {None: np.arange(len(x))})
    fig, axes = plt.subplots(1, len(panels), figsize=(11 if len(panels) > 1 else 7, 3.4), constrained_layout=True)
    for ax, (values, title) in zip(np.atleast_1d(axes), panels):
        values = np.asarray(values, dtype=float)
        for slot, (series, members) in enumerate(groups.items()):
            members = members[np.argsort(x[members], kind="stable")]
            ax.plot(x[members], values[members], color=PALETTE[slot % len(PALETTE)], lw=2, marker="o",
                    ms=7, mec=SURFACE, mew=1.5, clip_on=False, label=series)
        if layout["series"] is None and len(values):
            last = np.argmax(x)
            ax.annotate(_fmt(round(float(values[last]), 1)), (x[last], values[last]), xytext=(8, 0),
                        textcoords="offset points", va="center", color=INK)
        if layout["log"]:
            ax.set_xscale("log")
        ax.set_xticks(layout["ticks"], layout["labels"],
                      rotation=30 if layout["xlabel"] == "simulation" else 0,
                      ha="right" if layout["xlabel"] == "simulation" else "center")
        ax.minorticks_off()
        ax.set_xlabel(layout["xlabel"])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y")
        ax.set_title(title)
    if layout["series"]:
        fig.legend(*np.atleast_1d(axes)[0].get_legend_handles_labels(), loc="outside right upper")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    plt.show()


def summary(scan):
    """Per simulation: spikes, how many neurons fire, the stimulated neurons' rate, and any
    activity left after the stimuli end."""
    if not scan.sims:
        display(HTML("No simulation has a result yet."))
        return
    rows = []
    for sim in scan.sims:
        driven = np.isin(sim.neurons, sim.stimulated)
        during = scan._in_windows(sim, sim.times)
        n_firing = len(np.unique(sim.neurons))
        row = {"simulation": sim.label, "spikes": len(sim.times), "neurons that fire": n_firing,
               "mean rate of the neurons that fire (Hz)": len(sim.times) / max(n_firing, 1) / (sim.tstop / 1000)}
        if len(sim.stimulated):
            row["stimulated neurons"] = len(sim.stimulated)
            row["their rate while stimulated (Hz)"] = ((driven & during).sum() / len(sim.stimulated)
                                                      / (scan.stimulus_duration(sim) / 1000))
            row["other neurons that fire"] = len(np.unique(sim.neurons[~driven]))
        if sim.inputs and sim.stimulus_end < sim.tstop:
            row["spikes after the stimuli end"] = int((sim.times >= sim.stimulus_end).sum())
        row["last spike (ms)"] = sim.times.max() if len(sim.times) else np.nan
        rows.append(row)
    table = pd.DataFrame(rows).set_index("simulation")
    display(table.round(2))
    if scan.focal:
        panels = [(table["their rate while stimulated (Hz)"], "Stimulated neurons: mean rate while stimulated (Hz)"),
                  (table["other neurons that fire"], "Other neurons that fire at least once")]
    else:
        panels = [(table["neurons that fire"], "Neurons that fire at least once"),
                  (table["mean rate of the neurons that fire (Hz)"], "Mean rate of the neurons that fire (Hz)")]
    _curves(scan, panels, scan.plots_dir / "summary.png")


# --- Report traces -------------------------------------------------------------------------------


@lru_cache(maxsize=16)
def _report_traces(path, offsets_items, neurons, max_traces, max_points=20000):
    """Time (ms), traces [time, neuron] and neurons (rows of nodes) of a SONATA report: the first
    element (the soma) of each recorded neuron among `neurons` (every recorded one if None)."""
    offsets = dict(offsets_items)
    time, columns, rows = None, [], []
    with h5py.File(path) as f:
        for population, group in f["report"].items():
            if population not in offsets or "mapping" not in group:
                continue
            recorded = group["mapping/node_ids"][:].astype(np.int64) + offsets[population]
            first_element = group["mapping/index_pointers"][:-1]
            keep = np.isin(recorded, neurons) if neurons is not None else np.ones(len(recorded), dtype=bool)
            keep = np.flatnonzero(keep)[: max_traces - len(rows)]
            if not len(keep):
                continue
            start, _, dt = group["mapping/time"][:]
            n_frames = group["data"].shape[0]
            step = max(1, math.ceil(n_frames / max_points))
            columns.append(group["data"][::step, first_element[keep]])  # columns increase along the mapping
            rows.extend(recorded[keep])
            time = start + np.arange(0, n_frames, step) * dt
    if not rows:
        return None, None, np.array([], dtype=np.int64)
    return time, np.hstack(columns), np.asarray(rows)


def report_traces(scan, sim, name, neurons=None, max_traces=MAX_TRACES):
    """Time (ms), traces [time, neuron] and the neurons (rows of scan.nodes) of a report."""
    report = sim.reports[name]
    key = None if neurons is None else tuple(int(n) for n in np.sort(neurons))
    if report.traces:
        return report.traces(key, max_traces)
    return _report_traces(str(report.file), tuple(scan.offsets.items()), key, max_traces)


# --- Raster drawing ------------------------------------------------------------------------------


def _ranked_values(nodes, prop, members):
    counts = pd.Series(nodes[prop].to_numpy()[members], dtype=object).value_counts()
    return list(counts.sort_index().sort_values(ascending=False, kind="stable").index)


def _band_members(sim, members, selection, separate_stimulated):
    """Selected neurons that go into property bands, and the selected stimulated neurons."""
    members = members[selection[members]]
    driven = np.isin(members, sim.stimulated) if sim is not None else np.isin(members, [])
    if separate_stimulated:
        return members[~driven], members[driven]
    return members, members[:0]


def _scan_ranking(scan, prop, selection, separate_stimulated):
    """Values ranked over the selected neurons that fire in any simulation."""
    members = scan.responders[selection[scan.responders]]
    if separate_stimulated:
        members = members[~np.isin(members, scan.stimulated)]
    return _ranked_values(scan.nodes, prop, members)


def _bands(scan, sim, prop, selection, n_groups=len(PALETTE), rank_by="all simulations",
           order="first spike", separate_stimulated=True):
    """Raster bands, top to bottom: (label, colour, neurons in display order).

    With rank_by "all simulations" a value keeps its colour in every simulation; with "this
    simulation" values are ranked within the one simulation.
    """
    first_spike = pd.Series(sim.times).groupby(sim.neurons).min()
    spike_count = pd.Series(sim.neurons).value_counts()

    def arrange(members):
        if order == "first spike":
            return first_spike.loc[members].sort_values(kind="stable").index.to_numpy()
        if order == "spike count":
            return spike_count.loc[members].sort_values(ascending=False, kind="stable").index.to_numpy()
        return np.sort(members)

    others, driven = _band_members(sim, np.unique(sim.neurons), selection, separate_stimulated)
    bands = [("stimulated neurons", INK, arrange(driven))] if len(driven) else []
    ranking = (_ranked_values(scan.nodes, prop, others) if rank_by == "this simulation"
               else _scan_ranking(scan, prop, selection, separate_stimulated))
    top = ranking[:n_groups]
    values = scan.nodes[prop].to_numpy()[others]
    for slot, value in enumerate(top):
        members = others[values == value]
        if len(members):
            bands.append((_fmt(value), PALETTE[slot], arrange(members)))
    rest = ~np.isin(values, top)
    if rest.any():
        bands.append((f"other ({len(np.unique(values[rest]))} values)", OTHER, arrange(others[rest])))
    return bands


def _draw_raster(ax, scan, sim, bands, t_range, label_bands=True):
    for start, end in sim.shading():
        ax.axvspan(start, end, color=INK, alpha=0.05, lw=0)
    ax.set_xlim(*t_range)
    n_rows = sum(len(members) for _, _, members in bands)
    if n_rows == 0:
        ax.set_yticks([])
        ax.text(0.5, 0.5, "No spikes in this simulation" if len(sim.times) == 0
                else "None of the selected neurons fire in this simulation",
                transform=ax.transAxes, ha="center", va="center", color=MUTED)
        return
    row = np.full(len(scan.nodes), -1)
    band_of = np.full(len(scan.nodes), -1)
    row_pt = ax.get_position().height * ax.figure.get_figheight() * 72 / n_rows
    ticks, labels, top = [], [], 0
    for index, (label, _, members) in enumerate(bands):
        row[members] = np.arange(top, top + len(members))
        band_of[members] = index
        if top:
            ax.axhline(top - 0.5, color=GRID, lw=0.8)
        if label_bands and len(members) * row_pt >= 11:
            ticks.append(top + (len(members) - 1) / 2)
            labels.append(f"{label} ({len(members)})")
        top += len(members)
    keep = (row[sim.neurons] >= 0) & (sim.times >= t_range[0]) & (sim.times <= t_range[1])
    colours = np.array([colour for _, colour, _ in bands])[band_of[sim.neurons[keep]]]
    tick = float(np.clip(row_pt * 0.9, 1.5, 8))
    ax.scatter(sim.times[keep], row[sim.neurons[keep]], c=colours, marker="|", s=tick**2,
               linewidths=float(np.clip(tick / 4, 0.6, 1.2)), rasterized=True)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis="y", length=0)


def _lanes(sim):
    lanes = {}
    for i in sim.inputs:
        lanes.setdefault(i.lane, []).append(i)
    return lanes


def _draw_replayed(ax, inputs, base, lane_pt):
    """The spikes that inputs replay, one row per source neuron, coloured by population; lane_pt is
    the height of the lane in points."""
    sources = [(p, ids, times) for i in inputs for p, (ids, times) in i.sources.items()]
    populations = list(dict.fromkeys(p for p, _, _ in sources))
    rows, n_rows = {}, 0
    for p in populations:
        rows[p] = (np.unique(np.concatenate([ids for q, ids, _ in sources if q == p])), n_rows)
        n_rows += len(rows[p][0])
    tick = float(np.clip(0.76 * lane_pt / max(n_rows, 1), 1.5, 10))
    for p, ids, times in sources:
        unique, first = rows[p]
        y = base + 0.88 - 0.76 * (first + np.searchsorted(unique, ids) + 0.5) / max(n_rows, 1)
        ax.scatter(times, y, s=tick**2, marker="|", linewidths=float(np.clip(tick / 4, 0.6, 1.2)),
                   rasterized=True, color=PALETTE[populations.index(p) % len(PALETTE)])


def _draw_stimuli(ax, scan, sim):
    """One lane per stimulus: its time course where it has a simple shape, the spikes it replays
    for a spike replay, else a box for its window. Amplitudes are to scale within a lane, across all
    simulations."""
    lanes = _lanes(sim)
    t = np.linspace(0, sim.tstop, 20001)
    ticks, labels = [], []
    for index, ((name, module, node_set), inputs) in enumerate(lanes.items()):
        base = len(lanes) - 1 - index
        scale = max((i.amplitude() or 0 for s in scan.sims for i in s.inputs if i.lane == inputs[0].lane),
                    default=0) or 1
        shapes = [i.waveform(t) for i in inputs]
        if any(i.sources for i in inputs):
            _draw_replayed(ax, inputs, base, ax.get_position().height * ax.figure.get_figheight() * 72 / len(lanes))
        elif all(shape is not None for shape in shapes):
            ax.axhline(base + 0.5, color=GRID, lw=0.6)
            ax.plot(t, base + 0.5 + 0.42 * np.sum(shapes, axis=0) / scale, color=INK, lw=1.2)
        else:
            for i in inputs:
                ax.add_patch(Rectangle((i.delay, base + 0.18), i.duration, 0.64, color=PALETTE[0],
                                       alpha=0.3, lw=0))
        ticks.append(base + 0.5)
        size = f"{len(inputs[0].targets):,}" if scan.circuit else "?"  # no circuit, no node sets to resolve
        labels.append(f"{name} → {node_set} ({size})\n{inputs[0].describe()}")
    ax.set_ylim(0, max(len(lanes), 1))
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title("Stimuli (lane → target node set, neurons); shapes to scale within a lane"
                 + ("; replayed spikes one row per source neuron" if any(i.sources for i in sim.inputs) else ""))


def _bin_ms(tstop):
    """5 ms bins, wider for long runs so that there are at most about 400."""
    return next((b for b in (5, 10, 20, 50, 100, 200, 500) if tstop / b <= 400), 1000)


def _intervals(neurons, times):
    """Per neuron: the mean interval between its spikes (ms; of neurons with two spikes or more)
    and the coefficient of variation of those intervals (three spikes or more)."""
    order = np.lexsort((times, neurons))
    neurons, times = neurons[order], times[order]
    same = neurons[1:] == neurons[:-1]
    gaps = pd.Series(np.diff(times)[same]).groupby(neurons[1:][same])
    mean = gaps.mean()
    return mean, (gaps.std(ddof=0) / mean)[(gaps.size() >= 2) & (mean > 0)]


# --- Statistics of groups of neurons ---------------------------------------------------------------


def _while_on(scan):
    """', while stimulated' when some stimuli stop before the end of the run: rates are then over their windows."""
    return ", while stimulated" if any(s.windows and scan.stimulus_duration(s) < s.tstop for s in scan.sims) else ""


def _statistic_options(scan):
    """The statistics of a group of neurons in each simulation, as (label, key)."""
    while_on = _while_on(scan)
    return [("Neurons that fire", "firing"), ("Fraction of the selection that fires (%)", "fraction"),
            ("Spikes", "spikes"), (f"Mean rate of the neurons that fire{while_on} (Hz)", "rate"),
            (f"Mean rate of all selected neurons{while_on} (Hz)", "rate_all"),
            ("Peak population rate of the neurons that fire (Hz)", "peak"),
            ("Time of the peak population rate (ms)", "peak_time"),
            ("Median time of the first spike (ms)", "first"),
            ("Mean interval between spikes (ms)", "isi"),
            ("Irregularity: CV of the intervals between spikes", "cv"),
            ("Synchrony: Fano factor of the population spike count", "fano")]


def _statistics_by(scan, codes, n):
    """Every statistic of _statistic_options for n groups of neurons, given the group of each row of
    scan.nodes (codes; -1 for none): {key: array of groups x simulations}."""
    sizes = np.bincount(codes[codes >= 0], minlength=n)
    values = {key: np.full((n, len(scan.sims)), np.nan) for _, key in _statistic_options(scan)}
    for j, sim in enumerate(scan.sims):
        groups = codes[sim.neurons]
        keep = groups >= 0
        neurons, times, groups = sim.neurons[keep], sim.times[keep], groups[keep]
        firing = np.bincount(codes[np.unique(neurons)], minlength=n)
        during = np.bincount(groups[scan._in_windows(sim, times)], minlength=n)
        seconds = scan.stimulus_duration(sim) / 1000
        values["firing"][:, j] = firing
        values["fraction"][:, j] = 100 * firing / np.maximum(sizes, 1)
        values["spikes"][:, j] = np.bincount(groups, minlength=n)
        values["rate"][:, j] = during / np.maximum(firing, 1) / seconds
        values["rate_all"][:, j] = during / np.maximum(sizes, 1) / seconds
        mean, cv = _intervals(neurons, times)
        first = pd.Series(times).groupby(neurons).min()
        for key, per_neuron, how in (("first", first, "median"), ("isi", mean, "mean"), ("cv", cv, "mean")):
            group_of = codes[per_neuron.index.to_numpy(dtype=np.int64)]
            values[key][:, j] = per_neuron.groupby(group_of).agg(how).reindex(range(n)).to_numpy()
        # Each group's spike count in the time bins of the explorer's population rate
        width = _bin_ms(sim.tstop)
        n_bins = len(np.arange(0, sim.tstop + width, width)) - 1
        inside = (times >= 0) & (times <= n_bins * width)
        active, row = np.unique(groups[inside], return_inverse=True)
        column = np.minimum(times[inside] // width, n_bins - 1).astype(np.int64)
        counts = np.bincount(row * n_bins + column, minlength=len(active) * n_bins).reshape(len(active), n_bins)
        values["peak"][:, j] = 0.0
        values["peak"][active, j] = counts.max(axis=1) / firing[active] / (width / 1000)
        values["peak_time"][active, j] = (counts.argmax(axis=1) + 0.5) * width
        values["fano"][active, j] = counts.var(axis=1) / counts.mean(axis=1)
    return values


def _per_neuron(scan, key, counts):
    """What a table of neurons shows for the statistic `key`, and its value for each neuron of counts
    (their spike counts, neurons x simulations), one column per simulation. A statistic of a whole
    group has no value per neuron, so the table shows the spike counts."""
    rows = counts.index.to_numpy()
    if key in ("rate", "rate_all"):
        seconds = np.array([scan.stimulus_duration(s) for s in scan.sims]) / 1000
        return f"Firing rate{_while_on(scan)} (Hz)", (scan.stimulus_counts.loc[rows] / seconds).round(2)
    names = {"first": "Time of the first spike (ms)", "isi": "Mean interval between spikes (ms)",
             "cv": "CV of the intervals between spikes"}
    if key not in names:
        return "Spike counts", counts
    columns = {}
    for sim in scan.sims:
        keep = np.isin(sim.neurons, rows)
        neurons, times = sim.neurons[keep], sim.times[keep]
        if key == "first":
            values = pd.Series(times).groupby(neurons).min()
        else:
            mean, cv = _intervals(neurons, times)
            values = mean if key == "isi" else cv
        columns[sim.label] = values.reindex(rows).to_numpy()
    return names[key], pd.DataFrame(columns, index=counts.index).round(2)


def statistic_menu(scan):
    """A menu of the statistics of RasterApp.statistics, for group_statistics and neuron_statistics.
    It starts at the spikes."""
    return widgets.Dropdown(options=[label for label, _ in _statistic_options(scan)], value="Spikes",
                            description="Statistic", style=STYLE, layout=widgets.Layout(width="520px"))


def group_statistics(scan, by):
    """Every statistic of RasterApp.statistics for the neurons of each value of the property `by`, in
    each simulation: one row per group and simulation, with the group's number of neurons."""
    groups = pd.Categorical(scan.nodes[by])
    values = _statistics_by(scan, groups.codes.astype(np.int64), len(groups.categories))
    labels = [s.label for s in scan.sims]
    sizes = np.bincount(groups.codes[groups.codes >= 0], minlength=len(groups.categories))
    table = pd.DataFrame({
        by: pd.Categorical(np.repeat(groups.categories, len(labels)), categories=groups.categories),
        "simulation": pd.Categorical(np.tile(labels, len(groups.categories)), categories=labels),
        "neurons": np.repeat(sizes, len(labels)),
    })
    for label, key in _statistic_options(scan):
        table[label] = values[key].ravel().astype(np.int64 if key in ("firing", "spikes") else float)
    return table[table["neurons"] > 0].reset_index(drop=True)


def neuron_statistics(scan, statistic, simulation, properties=()):
    """A statistic of statistic_menu for each neuron that fires in one simulation, with its population,
    node id and the node properties named. Rates, the time of the first spike and the intervals
    between spikes have a value per neuron; for a statistic of a whole group, the last column holds
    the neurons' spike counts instead. It is named after what it holds."""
    key = dict(_statistic_options(scan)).get(statistic, statistic)
    sim = scan.simulation(simulation)
    rows = np.unique(sim.neurons)
    what, values = _per_neuron(scan, key, scan.spike_counts.loc[rows])
    table = pd.DataFrame({"population": scan.nodes["population"].to_numpy()[rows],
                          "node_id": scan.nodes["node_id"].to_numpy()[rows]}, index=rows)
    for name in properties:
        table[name] = scan.nodes[name].iloc[rows].to_numpy()
    table[what] = values[sim.label].to_numpy()
    return table.dropna(subset=[what])


def _grid(scan, sims):
    """For a scan over two parameters, (key down, key across, {key: value labels in order}), so
    that the overview can be a table of rasters; None for any other scan."""
    keys = [k for k in scan.scan_keys if len({_fmt(s.params.get(k)) for s in sims}) > 1]
    if len(keys) != 2 or len({tuple(_fmt(s.params.get(k)) for k in keys) for s in sims}) < len(sims):
        return None
    values = {k: list(dict.fromkeys(_fmt(v) for v in sorted((s.params.get(k) for s in sims), key=_sortable)))
              for k in keys}
    across, down = sorted(keys, key=lambda k: -len(values[k]))  # stable: on a tie, the first key goes across
    if len(values[across]) > 6:
        across, down = down, across
    return (down, across, values) if len(values[across]) <= 6 else None


def plot_overview(scan, prop, selection, caption, description):
    """One raster per simulation; a table of rasters for a scan over two parameters."""
    grid = _grid(scan, scan.sims) if len(scan.sims) <= MAX_GRID else None
    sims = scan.sims if grid else scan.sims[:MAX_OVERVIEW]
    if grid:
        down, across, values = grid
        n_rows, n_cols = len(values[down]), len(values[across])
    else:
        n_cols = 2 if len(sims) <= 8 else 3 if len(sims) <= 18 else 4
        n_rows = math.ceil(len(sims) / n_cols)
    same_length = len({s.tstop for s in sims}) == 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, (2.3 if n_cols == 2 else 1.8) * n_rows + 1.2),
                             sharex=same_length, constrained_layout=True, squeeze=False)
    if grid:
        place = {(values[down].index(_fmt(s.params.get(down))), values[across].index(_fmt(s.params.get(across)))): s
                 for s in sims}
        cells = [(axes[r, c], place.get((r, c))) for r in range(n_rows) for c in range(n_cols)]
    else:  # two columns read top to bottom
        flat = list(axes.T.flat if n_cols == 2 else axes.flat)
        cells = [(ax, sims[i] if i < len(sims) else None) for i, ax in enumerate(flat)]
    size = 9 if n_cols == 2 else 8
    for ax, sim in cells:
        if sim is None:
            ax.set_visible(False)
            continue
        _draw_raster(ax, scan, sim, _bands(scan, sim, prop, selection, separate_stimulated=scan.focal),
                     (0, sim.tstop), label_bands=False)
        selected = selection[sim.neurons]
        counts = f"{len(np.unique(sim.neurons[selected])):,} neurons, {int(selected.sum()):,} spikes"
        label = "" if grid else "\n".join(textwrap.wrap(sim.label, 150 // n_cols, break_long_words=False)) + ": "
        ax.set_title(label + counts, fontsize=size)
    if grid:  # the parameter values head the columns and the rows, on their first panel
        for c, value in enumerate(values[across]):
            ax = next((a for a in axes[:, c] if a.get_visible()), None)
            if ax is not None:
                header = textwrap.fill(f"{scan.short_keys[across]} = {value}", 150 // n_cols, break_long_words=False)
                ax.set_title(f"{header}\n{ax.get_title(loc='left')}", loc="left", fontsize=size)
        fig.supylabel(scan.short_keys[down], fontsize=size + 1, color=INK_2)
        for r, value in enumerate(values[down]):
            ax = next((a for a in axes[r] if a.get_visible()), None)
            if ax is not None:
                ax.set_ylabel(value, rotation=0, ha="right", va="center", fontsize=size + 1, color=INK)
    for ax in axes[-1]:
        ax.set_xlabel("Time (ms)")
    # The legend follows the colour ranking that _bands uses.
    ranking = _scan_ranking(scan, prop, selection, scan.focal)
    entries = [(_fmt(value), PALETTE[slot]) for slot, value in enumerate(ranking[:len(PALETTE)])]
    if scan.focal and selection[scan.stimulated].any():
        entries.insert(0, ("stimulated neurons", INK))
    if len(ranking) > len(PALETTE):
        entries.append(("other", OTHER))
    if entries:
        handles = [Line2D([], [], color=colour, marker="s", ls="", ms=7) for _, colour in entries]
        fig.legend(handles, [label for label, _ in entries], loc="outside upper center",
                   ncol=min(len(handles), 6), title=f"{caption}, grouped by {prop}")
    titles = [description] if description else []
    if len(scan.sims) > len(sims):
        titles.append(f"The first {len(sims)} of {len(scan.sims)} simulations")
    if titles:
        fig.suptitle("; ".join(titles), x=0.01, ha="left", fontsize=9, color=INK_2)
    return fig


def plot_explorer(scan, sim, prop, selection, caption, description, n_groups, rank_by, order, t_range,
                  separate_stimulated, report, show_traces):
    """Stimuli, recorded traces, raster and population rate of one simulation."""
    t_range = (t_range[0], min(t_range[1], sim.tstop))
    traces = (None, None, [])
    if show_traces and report in sim.reports and sim.reports[report].downloaded:
        selected = np.flatnonzero(selection)
        traces = report_traces(scan, sim, report, None if len(selected) == len(selection) else selected)
    panels = [*(["stimuli"] if sim.inputs else []), *(["traces"] if len(traces[2]) else []), "raster", "rate"]
    heights = {"stimuli": 0.55 * len(_lanes(sim)) + 0.5, "traces": 1.6, "raster": 6.0, "rate": 1.3}
    fig, axes = plt.subplots(len(panels), 1, sharex=True, constrained_layout=True,
                             figsize=(12, sum(heights[p] for p in panels) + 1.4),
                             gridspec_kw={"height_ratios": [heights[p] for p in panels]})
    ax = dict(zip(panels, np.atleast_1d(axes)))

    if "stimuli" in ax:
        _draw_stimuli(ax["stimuli"], scan, sim)

    if "traces" in ax:
        time, data, recorded = traces
        unit = sim.reports[report].unit
        ax["traces"].plot(time, data, color=INK, lw=0.6, alpha=0.3 if data.shape[1] > 1 else 0.9)
        ax["traces"].set_ylabel(unit)
        n = len(recorded)
        ax["traces"].set_title(f"{report}: {sim.reports[report].variable} of {_plural(n, 'recorded neuron')} "
                               f"in the selection" + (f" (the first {MAX_TRACES})" if n >= MAX_TRACES else ""))

    bands = _bands(scan, sim, prop, selection, n_groups, rank_by, order, separate_stimulated)
    _draw_raster(ax["raster"], scan, sim, bands, t_range)
    selected = selection[sim.neurons]
    ax["raster"].set_title(f"{caption}: {len(np.unique(sim.neurons[selected])):,} fire "
                           f"({int(selected.sum()):,} spikes), grouped by {prop}")
    if bands:
        handles = [Line2D([], [], color=colour, marker="s", ls="", ms=7) for _, colour, _ in bands]
        ax["raster"].legend(handles, [f"{label} ({len(members)})" for label, _, members in bands],
                            title=f"{prop} (neurons)", loc="upper left", bbox_to_anchor=(1.01, 1),
                            alignment="left")

    # The rate is of the selected neurons that are not stimulated; of the stimulated ones if no others fire.
    rated, who = selected, "selected neurons"
    if scan.focal:
        driven = np.isin(sim.neurons, sim.stimulated)
        rated, who = selected & ~driven, "selected non-stimulated neurons"
        if not rated.any() and (selected & driven).any():
            rated, who = selected & driven, "selected stimulated neurons"
    n_rated = len(np.unique(sim.neurons[rated]))
    width = _bin_ms(sim.tstop)
    edges = np.arange(0, sim.tstop + width, width)
    rate = np.histogram(sim.times[rated], edges)[0] / max(n_rated, 1) / (width / 1000)
    for start, end in sim.shading():
        ax["rate"].axvspan(start, end, color=INK, alpha=0.05, lw=0)
    ax["rate"].stairs(rate, edges, color=INK_2, lw=1.5)
    ax["rate"].set_ylim(0, max(1.0, 1.1 * rate.max()) if len(rate) else 1.0)
    ax["rate"].set_ylabel("Hz")
    ax["rate"].grid(axis="y")
    ax["rate"].set_title(f"Mean rate of the {n_rated:,} {who} that fire ({width:g} ms bins)")
    ax["rate"].set_xlabel("Time (ms)")
    for a in ax.values():
        a.set_xlim(*t_range)
    fig.suptitle("; ".join(x for x in [sim.label, description] if x), x=0.01, ha="left", fontsize=9,
                 color=INK_2)
    return fig


# --- The app: one neuron selection shared by every plot ------------------------------------------


def _slug(text):
    return re.sub(r"[^A-Za-z0-9.=-]+", "_", text).strip("_")[:80]


class RasterApp:
    """Widgets and plots that all follow one selection of neurons, by property and/or by id."""

    N_FILTERS = 3

    def __init__(self, scan, group_by=None):
        self.scan = scan
        self._outputs = {}  # name -> (Output, render, widgets it depends on)
        self._panels = []
        self._figures = {}
        self._paused = False
        self._first_view = None  # the view whose copy of the selection controls is open

        # Shared selection: property filters, ids, how to combine them, and the grouping.
        property_options = [("(choose a property)", "")] + [(p, p) for p in scan.properties]
        self.filters = [
            (widgets.Dropdown(options=property_options, layout=widgets.Layout(width="290px")),
             widgets.SelectMultiple(options=[], rows=6, layout=widgets.Layout(width="290px")))
            for _ in range(self.N_FILTERS)
        ]
        self.id_column, self.ids = _id_widgets(scan)
        self.match = widgets.ToggleButtons(
            options=[("all criteria (AND)", "all"), ("any criterion (OR)", "any")],
            description="Match", style=STYLE,
        )
        self.clear = widgets.Button(description="Clear selection", icon="times")
        self.group_by = widgets.Dropdown(options=scan.properties, value=group_by or scan.group_by,
                                         description="Group by", style=STYLE)
        self.selection_widgets = {*(w for pair in self.filters for w in pair), self.ids, self.match}

        # Explorer controls.
        # Opens on the simulation where most neurons fire; on a tie, the one with most spikes.
        busiest = max(range(len(scan.sims)), key=lambda i: (len(np.unique(scan.sims[i].neurons)),
                                                            len(scan.sims[i].times)), default=0)
        self.simulation = widgets.SelectionSlider(
            options=[(s.label, i) for i, s in enumerate(scan.sims)] or [("none", 0)], value=busiest,
            description="Simulation", continuous_update=False, style=STYLE,
            layout=widgets.Layout(width="560px"),
        )
        self.groups = widgets.IntSlider(value=len(PALETTE), min=1, max=len(PALETTE),
                                        description="Groups", continuous_update=False, style=STYLE)
        self.rank = widgets.ToggleButtons(options=["all simulations", "this simulation"],
                                          description="Rank values by", style=STYLE)
        self.order = widgets.ToggleButtons(options=["first spike", "spike count", "node id"],
                                           description="Sort", style=STYLE)
        tstop = max((s.tstop for s in scan.sims), default=1000.0)
        self.time = widgets.FloatRangeSlider(
            value=(0.0, tstop), min=0.0, max=tstop, step=10 ** math.floor(math.log10(max(tstop / 100, 0.01))),
            description="Time (ms)",
            continuous_update=False, readout_format=".0f", style=STYLE,
            layout=widgets.Layout(width="440px"),
        )
        self.separate = widgets.Checkbox(value=scan.focal, description="Stimulated neurons on top",
                                         style=STYLE)
        report_names = list(dict.fromkeys(n for s in scan.sims for n in s.reports))
        # Shown at first: the first report that was downloaded (large ones are only with reports=True).
        downloaded = [n for n in report_names if any(n in s.reports and s.reports[n].downloaded for s in scan.sims)]
        self.report = widgets.Dropdown(options=report_names or ["(none)"], description="Report", style=STYLE,
                                       value=(downloaded or report_names or ["(none)"])[0],
                                       layout=widgets.Layout(display="flex" if report_names else "none"))
        self.traces = widgets.Checkbox(value=bool(downloaded), description="Recorded traces", style=STYLE,
                                       layout=widgets.Layout(display="flex" if report_names else "none"))
        self.explorer_widgets = {self.simulation, self.groups, self.rank, self.order, self.time,
                                 self.separate, self.report, self.traces}
        self.measure = widgets.ToggleButtons(options=["neurons that fire", "spikes"], description="Count",
                                             style=STYLE)
        self.statistic = widgets.Dropdown(options=_statistic_options(scan), description="Statistic", style=STYLE,
                                          layout=widgets.Layout(width="520px"))

        for prop_w, values_w in self.filters:
            prop_w.observe(lambda change, values_w=values_w: self._fill_values(change["new"], values_w),
                           names="value")
        # A property box redraws through _fill_values once its value list is refilled.
        property_boxes = {prop_w for prop_w, _ in self.filters}
        for w in [*(self.selection_widgets - property_boxes), self.group_by, *self.explorer_widgets,
                  self.measure, self.statistic]:
            w.observe(self._on_change, names="value")
        self.id_column.observe(self._on_id_column, names="value")
        self.clear.on_click(self._clear)

    # Selection ---------------------------------------------------------------------------------

    def _fill_values(self, prop, values_w):
        """List the values `prop` takes among neurons that fire, with how many of them carry each."""
        was_paused, self._paused = self._paused, True
        if prop:
            counts = self.scan.nodes[prop].iloc[self.scan.responders].value_counts()
            values_w.options = [(f"{_fmt(value)} ({count})", value) for value, count in counts.items() if count]
        else:
            values_w.options = []
        self._paused = was_paused
        if not was_paused:
            self._redraw_selection()

    def _clear(self, _):
        self._paused = True
        for prop_w, values_w in self.filters:
            prop_w.value = ""
            values_w.options = []
        self.ids.value = ""
        self._paused = False
        self._redraw_selection()

    def _on_id_column(self, _):
        if self.ids.value.strip():  # the column changes the selection only when ids are listed
            self._redraw_selection()

    def _spec(self):
        filters = tuple((p.value, tuple(v.value)) for p, v in self.filters if p.value and v.value)
        rows = self.scan.find(self.ids.value, self.id_column.value)[0]
        return filters, tuple(rows), bool(self.ids.value.strip()), self.match.value

    def selection(self):
        """Boolean mask over all neurons; with no criteria, every neuron is selected."""
        return self._mask(self._spec())

    @lru_cache(maxsize=32)
    def _mask(self, spec):
        return _select(self.scan.nodes, *spec)

    def description(self):
        """One line naming the selection; '' when every neuron is selected."""
        filters, listed_ids, has_ids, match = self._spec()
        return _selection_text(filters, len(listed_ids), has_ids, match)

    def caption(self):
        n = int(self.selection().sum())
        return "All neurons" if n == len(self.scan.nodes) else f"{n:,} selected neurons"

    def _tag(self):
        """'' when every neuron is selected, else a short tag naming the selection in file names."""
        description = self.description()
        return "_sel-" + hashlib.md5(description.encode()).hexdigest()[:6] if description else ""

    # Redrawing ---------------------------------------------------------------------------------

    def _register(self, name, render, depends_on):
        self._outputs[name] = (widgets.Output(), render, set(depends_on))
        self._draw(name)
        return self._outputs[name][0]

    def _draw(self, name):
        out, render, _ = self._outputs[name]
        with out:
            out.clear_output(wait=True)
            render()

    def _on_change(self, change):
        if self._paused:
            return
        if change["owner"] in self.selection_widgets or change["owner"] is self.group_by:
            self._refresh_titles()
        for name, (_, _, depends_on) in self._outputs.items():
            if change["owner"] in depends_on:
                self._draw(name)

    def _redraw_selection(self):
        self._refresh_titles()
        for name, (_, _, depends_on) in self._outputs.items():
            if depends_on & self.selection_widgets:
                self._draw(name)

    # Panels ------------------------------------------------------------------------------------

    def _controls(self):
        """A view of the shared selection widgets; every view of them stays in sync."""
        heading = "<b>{}</b> <span style='color:#52514e'>{}</span>".format
        return widgets.VBox([
            widgets.HTML(heading("Select neurons by property", "pick a property in a box, then "
                                 "one or more of its values (ctrl-click, or ⌘-click on a Mac). "
                                 "The lists show the values of neurons that fire, with counts.")),
            widgets.HBox([widgets.VBox([p, v]) for p, v in self.filters]),
            widgets.HTML(heading("Select neurons by id", "")),
            self.id_column,
            self.ids,
            widgets.HBox([self.match, self.clear]),
            self.group_by,
        ])

    def _panel_title(self):
        description = self.description() or "all neurons"
        return f"Select neurons: {description}, grouped by {self.group_by.value}"

    def _accordion(self, view):
        """A copy of the selection controls: open in the first view shown, folded in the others."""
        self._first_view = self._first_view or view
        accordion = widgets.Accordion(children=[self._controls()],
                                      selected_index=0 if view == self._first_view else None)
        accordion.set_title(0, self._panel_title())
        self._panels.append(accordion)
        return accordion

    def _refresh_titles(self):
        for accordion in self._panels:
            accordion.set_title(0, self._panel_title())

    def _save_button(self, kind):
        button = widgets.Button(description="Save figure", icon="save")
        status = widgets.Label()

        def save(_):
            fig, name = self._figures[kind]
            self.scan.plots_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.scan.plots_dir / f"{name}.png", dpi=150)
            filters, listed_ids, has_ids, match = self._spec()
            if filters or has_ids:
                (self.scan.plots_dir / f"{name}.json").write_text(json.dumps({
                    "selection": self.description(), "match": match,
                    "filters": {prop: [_fmt(v) for v in values] for prop, values in filters},
                    "neurons": [str(self.scan.neuron_name(r)) for r in listed_ids],
                }, indent=2))
            status.value = f"saved {self.scan.plots_dir / name}.png"

        button.on_click(save)
        return widgets.HBox([button, status])

    def selection_panel(self):
        """The selection controls, in a panel that folds, with the statistics of the selected neurons."""
        return widgets.VBox([self._accordion("selection"), self.statistic, self._register(
            "selection", self._show_selection, self.selection_widgets | {self.statistic})])

    def statistics(self):
        """The statistics of the selected neurons, with the menu that picks the one plotted, under a copy
        of the selection controls. Use it or selection_panel, not both."""
        return widgets.VBox([self._accordion("statistics"), self.statistic, self._register(
            "selection", self._show_selection, self.selection_widgets | {self.statistic})])

    def overview(self):
        """One raster per simulation, with the selection controls above it."""
        return widgets.VBox([self._accordion("overview"), self._save_button("overview"),
                             self._register("overview", self._show_overview,
                                            self.selection_widgets | {self.group_by})])

    def explorer(self):
        """One simulation in detail, with the selection controls and the explorer's own controls."""
        controls = widgets.VBox([
            widgets.HBox([self.simulation, self.time]),
            widgets.HBox([self.groups, self.rank]),
            widgets.HBox([self.order, self.separate]),
            widgets.HBox([self.traces, self.report]),
            self._save_button("explorer"),
        ])
        return widgets.VBox([self._accordion("explorer"), controls,
                             self._register("explorer", self._show_explorer,
                                            self.selection_widgets | self.explorer_widgets | {self.group_by})])

    def group_table(self):
        """How many selected neurons of each group fire (or how many spikes they fire) in each simulation."""
        return widgets.VBox([self.measure, self._register(
            "table", self._show_group_table, self.selection_widgets | {self.group_by, self.measure})])

    # Renderers ---------------------------------------------------------------------------------

    def _show_selection(self):
        scan, selection = self.scan, self.selection()
        selected = np.flatnonzero(selection)
        firing = scan.spike_counts.loc[np.intersect1d(scan.responders, selected)]
        text = (f"<b>{self.description() or 'All neurons'}</b>: {len(selected):,} neurons selected, "
                f"{len(firing):,} of which fire in at least one simulation")
        n_driven = int(np.isin(selected, scan.stimulated).sum())
        if n_driven and scan.focal:
            text += f", including {n_driven:,} stimulated neurons"
        unknown = scan.find(self.ids.value, self.id_column.value)[1]
        if unknown:
            text += f". <span style='color:#d03b3b'>Not recognised: {', '.join(unknown[:10])}</span>"
        display(HTML(text))
        if firing.empty:
            return
        statistics = self._statistics(selection)
        if len(scan.sims) > 1:
            _curves(scan, [(statistics[self.statistic.label].to_numpy(), self.statistic.label)])
        else:
            display(statistics.T.round(2))
        # The neurons in the order they start firing across the scan, then by their spikes.
        first_in = pd.Series(np.argmax(firing.to_numpy() > 0, axis=1), index=firing.index)
        totals = firing.sum(axis=1)
        ranked = pd.DataFrame({"first": first_in, "fewer": -totals}).sort_values(["first", "fewer"], kind="stable").index
        shown = ranked[:50]
        what, values = _per_neuron(scan, self.statistic.value, firing.loc[shown])
        table = scan.nodes.loc[shown, ["population", "node_id", *scan.id_columns, *scan.properties]]
        table = table.loc[:, ~table.columns.duplicated()]
        if len(scan.offsets) == 1:
            table = table.drop(columns="population")
        table.insert(0, "first fires in", [scan.sims[i].label for i in first_in.loc[shown]])
        table = table.join(values)
        if what == "Spike counts":
            table = table.assign(total=totals.loc[shown])
        table = table.set_index("node_id" if len(scan.offsets) == 1 else ["population", "node_id"])
        display(HTML(f"{what} of {f'the first {len(shown)} of ' if len(ranked) > len(shown) else 'the '}"
                     f"{len(ranked):,} neurons that fire, in the order they start firing:"))
        display(table)

    def _statistics(self, selection):
        """Every statistic of the menu for the selected neurons, one row per simulation."""
        values = _statistics_by(self.scan, np.where(selection, 0, -1), 1)
        return pd.DataFrame({label: values[key][0] for label, key in self.statistic.options},
                            index=pd.Index([s.label for s in self.scan.sims], name="simulation"))

    def _show_overview(self):
        if not self.scan.sims:
            display(HTML("No simulation has a result yet."))
            return
        fig = plot_overview(self.scan, self.group_by.value, self.selection(), self.caption(), self.description())
        self._figures["overview"] = (fig, f"rasters_by_{self.group_by.value}{self._tag()}")
        plt.show()

    def _show_explorer(self):
        if not self.scan.sims:
            display(HTML("No simulation has a result yet."))
            return
        sim = self.scan.sims[self.simulation.value]
        fig = plot_explorer(self.scan, sim, self.group_by.value, self.selection(), self.caption(),
                            self.description(), self.groups.value, self.rank.value, self.order.value,
                            self.time.value, self.separate.value, self.report.value, self.traces.value)
        self._figures["explorer"] = (fig, f"raster_{_slug(sim.label)}_by_{self.group_by.value}{self._tag()}")
        plt.show()

    def _show_group_table(self):
        scan, prop, selection = self.scan, self.group_by.value, self.selection()
        spikes = self.measure.value == "spikes"
        columns, driven_row = {}, {}
        for sim in scan.sims:
            members, driven = _band_members(sim, np.unique(sim.neurons), selection, scan.focal)
            if spikes:
                counts = pd.Series(sim.neurons).value_counts()
                columns[sim.label] = pd.Series(counts.loc[members].to_numpy(),
                                               index=scan.nodes[prop].iloc[members].to_numpy()).groupby(level=0).sum()
                driven_row[sim.label] = int(counts.loc[driven].sum())
            else:
                columns[sim.label] = pd.Series(scan.nodes[prop].iloc[members].to_numpy()).value_counts()
                driven_row[sim.label] = len(driven)
        table = pd.DataFrame(columns).fillna(0).astype(int)
        table = table.loc[[v for v in _scan_ranking(scan, prop, selection, scan.focal) if v in table.index]]
        if any(driven_row.values()):
            table = pd.concat([pd.DataFrame(driven_row, index=["stimulated neurons"]), table])
        table.index.name = prop
        if table.empty:
            display(HTML("None of the selected neurons fire in any simulation."))
        else:
            display(table)

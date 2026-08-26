"""Reusable helpers for load-only Na/K trace analysis.

The module contains no simulator setup or execution. It parses precomputed
result assets, makes trace metadata inspectable, selects exact traces, aligns
sample sequences, and provides the two plots used by the analysis notebook.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


DATA_KEYS = ("x", "y")
NONE_VALUE = "_NONE"
TRACE_NAMES = ("voltage", "sodium", "potassium")


__all__ = [
    "DATA_KEYS",
    "NONE_VALUE",
    "entries_to_df",
    "load_simulation_results",
    "metadata_units",
    "normalize_trace_bundle",
    "plot_normalized_voltage_and_currents",
    "plot_voltage_and_currents",
    "plot_zoomed_ap",
    "prepare_trace_bundle",
    "read_list_of_entries",
    "read_sim_config_data",
    "result_summary",
    "select_trace",
    "trace_catalog",
]


def _metadata_value(value: Any) -> Any:
    """Convert JSON metadata into a stable, indexable scalar."""
    if value is None:
        return NONE_VALUE
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, sort_keys=True, default=str)
    try:
        if bool(pd.isna(value)):
            return NONE_VALUE
    except (TypeError, ValueError):
        pass
    return value


def entries_to_df(
    entries: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    recording_group: Any = None,
) -> pd.DataFrame:
    """Convert x/y entries into a metadata-indexed DataFrame.

    Each input trace becomes one group of rows. The x/y samples remain in
    columns, while every other JSON field becomes repeated trace metadata and
    is included in the resulting index.
    """
    if isinstance(entries, dict):
        if all(key in entries for key in DATA_KEYS):
            entries = [entries]
        else:
            entries = list(entries.values())

    if not isinstance(entries, (list, tuple)):
        raise TypeError("Trace entries must be a mapping or a sequence of mappings.")

    frames = []
    for trace_index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            raise TypeError("Every simulation entry must be a JSON object.")

        entry = dict(raw_entry)
        missing = [key for key in DATA_KEYS if key not in entry]
        if missing:
            raise ValueError(f"Trace entry is missing required keys: {missing}")

        try:
            x = np.asarray(entry.pop("x"), dtype=float)
            y = np.asarray(entry.pop("y"), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Trace x/y values must be numeric arrays.") from exc

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("Trace x/y values must be one-dimensional arrays.")
        if len(x) == 0 or len(x) != len(y):
            raise ValueError("Trace x and y arrays must be non-empty and have equal lengths.")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("Trace x/y arrays must contain only finite values.")

        metadata = {key: _metadata_value(value) for key, value in entry.items()}
        if recording_group is not None:
            metadata["recording_group"] = _metadata_value(recording_group)
        else:
            metadata.setdefault("recording_group", NONE_VALUE)
        metadata["trace_index"] = trace_index

        frame = pd.DataFrame({"x": x, "y": y})
        for key, value in metadata.items():
            frame[key] = value
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=list(DATA_KEYS))

    result = pd.concat(frames, axis=0, ignore_index=True)
    metadata_columns = [column for column in result.columns if column not in DATA_KEYS]
    return result.set_index(metadata_columns)


def read_list_of_entries(
    groups: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Read either a group-to-entries mapping or one entries sequence."""
    if isinstance(groups, dict):
        grouped_entries = groups.items()
    elif isinstance(groups, list):
        grouped_entries = [(NONE_VALUE, groups)]
    else:
        raise TypeError("The simulation section must be a mapping or list.")

    frames = [
        entries_to_df(entries, recording_group=group_name)
        for group_name, entries in grouped_entries
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise ValueError("The simulation result does not contain any traces.")
    return pd.concat(frames, axis=0)


def read_sim_config_data(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read and validate one precomputed simulation result JSON file."""
    path = Path(path)
    with path.open(encoding="utf-8") as file:
        config = json.load(file)

    for key in ("simulation", "stimulus"):
        if key not in config:
            raise ValueError(f"Result {path} is missing the '{key}' section.")

    data = read_list_of_entries(config["simulation"])
    stimulus = entries_to_df(config["stimulus"], recording_group="stimulus")
    return data, stimulus


def load_simulation_results(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Parse a collection of local result files into notebook-ready records."""
    results = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Simulation result not found: {path}")
        data, stimulus = read_sim_config_data(path)
        results.append({"path": path, "data": data, "stimulus": stimulus})
    if not results:
        raise ValueError("No simulation result files were supplied.")
    return results


def result_summary(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Return one row per loaded result with trace and stimulus counts."""
    return pd.DataFrame(
        {
            "file": [Path(result["path"]).name for result in results],
            "trace_rows": [len(result["data"]) for result in results],
            "stimulus_rows": [len(result["stimulus"]) for result in results],
        }
    )


def trace_catalog(data: pd.DataFrame) -> pd.DataFrame:
    """Return one row per unique recorded trace and its metadata."""
    if data.empty:
        return pd.DataFrame(columns=list(data.index.names))
    return data.index.to_frame(index=False).drop_duplicates().reset_index(drop=True)


def _normalise_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _row_contains_alias(row: pd.Series, aliases: Sequence[str]) -> bool:
    tokens = set()
    for value in row.tolist():
        normalised = _normalise_text(value)
        tokens.add(normalised)
        tokens.update(normalised.split("_"))
    return any(_normalise_text(alias) in tokens for alias in aliases)


def select_trace(
    data: pd.DataFrame,
    aliases: Sequence[str],
    selector: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select exactly one trace by metadata or common variable aliases.

    ``selector`` should contain exact metadata fields from ``trace_catalog``.
    Alias matching is retained as a convenience, but ambiguous matches fail
    loudly so the notebook cannot silently plot the wrong recording.
    """
    available = trace_catalog(data)
    if available.empty:
        raise ValueError("The simulation result does not contain any trace metadata.")

    if selector is not None:
        unknown = set(selector).difference(available.columns)
        if unknown:
            raise KeyError(f"Unknown trace metadata fields: {sorted(unknown)}")
        matches = available.copy()
        for key, value in selector.items():
            matches = matches[
                matches[key].map(_normalise_text).eq(_normalise_text(value))
            ]
    else:
        matches = available[
            available.apply(_row_contains_alias, axis=1, aliases=aliases)
        ]

    if matches.empty:
        raise ValueError(
            f"Could not find a trace matching {aliases}. Inspect catalog and set an exact filter.\n"
            + available.to_string(index=False)
        )
    if len(matches) > 1:
        raise ValueError(
            f"More than one trace matches {aliases}. Set an exact filter.\n"
            + matches.to_string(index=False)
        )

    identity = matches.iloc[0].to_dict()
    reset_data = data.reset_index()
    row_mask = np.ones(len(reset_data), dtype=bool)
    for key, value in identity.items():
        row_mask &= (
            reset_data[key].map(_normalise_text).eq(_normalise_text(value)).to_numpy()
        )

    trace = reset_data.loc[row_mask, list(DATA_KEYS)]
    if trace.empty:
        raise ValueError("The selected trace contains no samples.")
    return {
        "time": trace["x"].to_numpy(dtype=float),
        "values": trace["y"].to_numpy(dtype=float),
        "metadata": identity,
    }


def _clean_trace(trace: Mapping[str, Any], name: str) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(trace["time"], dtype=float)
    values = np.asarray(trace["values"], dtype=float)
    if len(time) != len(values) or len(time) < 2:
        raise ValueError(f"{name} must contain at least two aligned samples.")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values.")

    order = np.argsort(time)
    time, values = time[order], values[order]
    if np.any(np.diff(time) <= 0):
        raise ValueError(f"{name} time values must be strictly increasing.")
    return time, values


def prepare_trace_bundle(traces: Mapping[str, Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Validate and collect voltage, Na, and K traces on one sample sequence."""
    missing = set(TRACE_NAMES).difference(traces)
    if missing:
        raise KeyError(f"Missing required traces: {sorted(missing)}")

    cleaned = {
        name: _clean_trace(traces[name], name)
        for name in TRACE_NAMES
    }
    base_time = cleaned["voltage"][0]
    bundle = {"t": base_time, "v": cleaned["voltage"][1]}

    for name, output_name in (("sodium", "ina"), ("potassium", "ik")):
        time, values = cleaned[name]
        same_grid = (
            time.shape == base_time.shape
            and np.allclose(time, base_time, rtol=1e-7, atol=1e-12)
        )
        if not same_grid:
            raise ValueError(f"{name} does not use the voltage sample sequence.")
        bundle[output_name] = values
    return bundle


def metadata_units(trace: Mapping[str, Any], default: str) -> str:
    """Read a trace unit field, falling back when metadata has no unit."""
    metadata = trace.get("metadata", {})
    for key in ("units", "unit"):
        value = metadata.get(key)
        if value not in (None, NONE_VALUE, ""):
            return str(value)
    return default


def _get_pyplot():
    import matplotlib.pyplot as plt

    return plt


def _plot_arrays(
    t: Sequence[float],
    v: Sequence[float],
    ina: Sequence[float],
    ik: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(np.asarray(values, dtype=float) for values in (t, v, ina, ik))
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("Plot inputs must be one-dimensional arrays.")
    if len({len(array) for array in arrays}) != 1:
        raise ValueError("Plot inputs must have equal lengths.")
    if len(arrays[0]) < 2:
        raise ValueError("Plot inputs must contain at least two samples.")
    return arrays


def normalize_trace_bundle(
    t: Sequence[float],
    v: Sequence[float],
    ina: Sequence[float],
    ik: Sequence[float],
) -> dict[str, np.ndarray]:
    """Normalize voltage and currents for dimensionless shape comparison.

    Voltage is scaled from its initial value to its maximum value. Each
    current is divided by its maximum absolute magnitude, preserving polarity.
    The input arrays are not modified.
    """
    t, v, ina, ik = _plot_arrays(t, v, ina, ik)
    if any(not np.all(np.isfinite(array)) for array in (t, v, ina, ik)):
        raise ValueError("Normalization inputs must contain only finite values.")

    voltage_baseline = float(v[0])
    voltage_scale = float(np.max(v) - voltage_baseline)
    if voltage_scale <= 0:
        raise ValueError("Voltage must rise above its initial baseline to be normalized.")

    current_scales = {
        "ina": float(np.max(np.abs(ina))),
        "ik": float(np.max(np.abs(ik))),
    }
    for name, scale in current_scales.items():
        if scale <= 0:
            raise ValueError(f"{name} must contain a non-zero value to be normalized.")

    return {
        "t": t,
        "v": (v - voltage_baseline) / voltage_scale,
        "ina": ina / current_scales["ina"],
        "ik": ik / current_scales["ik"],
    }


def plot_normalized_voltage_and_currents(
    t: Sequence[float],
    v: Sequence[float],
    ina: Sequence[float],
    ik: Sequence[float],
):
    """Plot normalized voltage, Na current, and K current for shape comparison."""
    plt = _get_pyplot()
    normalized = normalize_trace_bundle(t, v, ina, ik)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(normalized["t"], normalized["v"], color="black", label="Voltage")
    ax.plot(normalized["t"], normalized["ina"], color="blue", label="iNa")
    ax.plot(normalized["t"], normalized["ik"], color="red", alpha=0.7, label="iK")
    ax.axhline(0, color="0.7", linewidth=0.8)
    ax.set_ylabel("Normalized amplitude")
    ax.set_xlabel("Time (ms)")
    ax.set_title("Normalized voltage and Na⁺ / K⁺ waveform shapes")
    ax.legend()
    fig.tight_layout()
    plt.show()
    return fig, ax


def plot_voltage_and_currents(
    t: Sequence[float],
    v: Sequence[float],
    ina: Sequence[float],
    ik: Sequence[float],
    voltage_units: str = "mV",
    current_units: str = "mA/cm²",
):
    """Plot voltage, Na current, and K current in three aligned panels."""
    plt = _get_pyplot()
    t, v, ina, ik = _plot_arrays(t, v, ina, ik)

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t, v, color="black")
    axes[0].set_ylabel(f"V ({voltage_units})")
    axes[0].set_title("Action potential with Na⁺ / K⁺ current densities")

    axes[1].plot(t, ina, color="blue")
    axes[1].set_ylabel(f"iNa ({current_units})")

    axes[2].plot(t, ik, color="red", alpha=0.7)
    axes[2].set_ylabel(f"iK ({current_units})")
    axes[2].set_xlabel("Time (ms)")
    fig.tight_layout()
    plt.show()
    return fig, axes


def _automatic_zoom_window(
    t: np.ndarray,
    v: np.ndarray,
    width_ms: float = 10,
) -> tuple[float, float]:
    if width_ms <= 0:
        raise ValueError("The zoom-window width must be positive.")
    if not np.any(np.isfinite(v)):
        raise ValueError("Voltage contains no finite values for automatic zooming.")
    peak_time = float(t[int(np.nanargmax(v))])
    return peak_time - 0.4 * width_ms, peak_time + 0.6 * width_ms


def plot_zoomed_ap(
    t: Sequence[float],
    v: Sequence[float],
    ina: Sequence[float],
    ik: Sequence[float],
    t_start: float | None = None,
    t_end: float | None = None,
    voltage_units: str = "mV",
    current_units: str = "mA/cm²",
):
    """Plot one action-potential window with voltage/current y-axes."""
    plt = _get_pyplot()
    t, v, ina, ik = _plot_arrays(t, v, ina, ik)

    if (t_start is None) != (t_end is None):
        raise ValueError("Provide both zoom-window bounds or neither.")
    if t_start is None and t_end is None:
        t_start, t_end = _automatic_zoom_window(t, v)
    else:
        t_start, t_end = float(t_start), float(t_end)
    if t_end <= t_start:
        raise ValueError("The zoom-window end must be greater than its start.")

    mask = (t >= t_start) & (t <= t_end)
    if mask.sum() < 2:
        raise ValueError("The selected zoom window contains fewer than two samples.")

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(t[mask], v[mask], color="black", label="Voltage (Vm)")
    ax1.set_ylabel(f"Vm ({voltage_units})")
    ax1.set_xlabel("Time (ms)")

    ax2 = ax1.twinx()
    ax2.plot(t[mask], ina[mask], color="blue", label=f"iNa ({current_units})")
    ax2.plot(t[mask], ik[mask], color="red", alpha=0.7, label=f"iK ({current_units})")
    ax2.set_ylabel(f"Ionic current ({current_units})")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title(f"Na⁺ and K⁺ currents around the AP ({t_start:.2f}–{t_end:.2f} ms)")
    fig.tight_layout()
    plt.show()
    return fig, ax1, ax2

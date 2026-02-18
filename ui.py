"""
EV Charge Manager — Interactive Terminal Launcher
==================================================
Run with:   python ui.py
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_GREEN  = "\033[32m"
_CYAN   = "\033[36m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_BLUE   = "\033[34m"
_MAGENTA= "\033[35m"
_WHITE  = "\033[37m"
_BG_BLUE = "\033[44m"

def _c(text: str, *codes: str) -> str:
    """Wrap text in ANSI codes (auto-stripped on non-TTY)."""
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + _RESET

def _header(title: str) -> None:
    width = 64
    print()
    print(_c("╔" + "═" * (width - 2) + "╗", _BOLD, _CYAN))
    label = f"  {title}  "
    pad   = width - 2 - len(label)
    left  = pad // 2
    right = pad - left
    print(_c("║" + " " * left + label + " " * right + "║", _BOLD, _CYAN))
    print(_c("╚" + "═" * (width - 2) + "╝", _BOLD, _CYAN))

def _section(title: str) -> None:
    print()
    print(_c(f"  ── {title} " + "─" * max(0, 50 - len(title)), _BOLD, _BLUE))

def _info(msg: str) -> None:
    print(_c(f"  ℹ  {msg}", _DIM))

def _ok(msg: str) -> None:
    print(_c(f"  ✔  {msg}", _GREEN))

def _warn(msg: str) -> None:
    print(_c(f"  ⚠  {msg}", _YELLOW))

def _err(msg: str) -> None:
    print(_c(f"  ✖  {msg}", _RED))

def _separator() -> None:
    print(_c("  " + "─" * 62, _DIM))

# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _ask(prompt: str, default: Any = None, cast=str,
         choices: Optional[List[str]] = None) -> Any:
    """Interactive prompt with default value and optional choice validation."""
    hint = ""
    if choices:
        hint = f" [{'/'.join(choices)}]"
    elif default is not None:
        hint = f" (default: {_c(str(default), _YELLOW)})"

    while True:
        raw = input(f"    {prompt}{hint}: ").strip()
        if raw == "" and default is not None:
            return default
        if raw == "":
            _warn("A value is required.")
            continue
        if choices and raw.upper() not in [c.upper() for c in choices]:
            _warn(f"Choose one of: {', '.join(choices)}")
            continue
        try:
            val = cast(raw)
            return val
        except (ValueError, TypeError):
            _warn(f"Expected {cast.__name__}, got: {raw!r}")


def _ask_bool(prompt: str, default: bool = False) -> bool:
    """Yes/No prompt."""
    hint = "Y/n" if default else "y/N"
    raw = input(f"    {prompt} [{hint}]: ").strip().lower()
    if raw == "":
        return default
    return raw in ("y", "yes", "1", "true")


def _ask_optional(prompt: str, default_display: str = "disabled", cast=float) -> Optional[Any]:
    """Prompt for an optional numeric value (Enter = disabled/None)."""
    raw = input(f"    {prompt} (Enter = {default_display}): ").strip()
    if raw == "":
        return None
    try:
        return cast(raw)
    except (ValueError, TypeError):
        _warn(f"Invalid value, using {default_display}.")
        return None


def _menu(title: str, options: List[Tuple[str, str]], prompt: str = "Choice") -> str:
    """Display a numbered menu and return the chosen key."""
    _header(title)
    print()
    for i, (key, label) in enumerate(options, 1):
        print(f"    {_c(str(i), _BOLD, _CYAN)}.  {label}")
    print()
    while True:
        raw = input(f"  {prompt} [1–{len(options)}]: ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1][0]
        _warn(f"Enter a number between 1 and {len(options)}.")


def _confirm_run(cmd: List[str]) -> bool:
    """Show the command that will be executed and ask for confirmation."""
    _section("Command to run")
    cmd_str = " ".join(cmd)
    print(f"    {_c(cmd_str, _CYAN)}")
    print()
    return _ask_bool("Proceed?", default=True)


# ---------------------------------------------------------------------------
# Highway Selection helper
# ---------------------------------------------------------------------------

def _run_highway_selector() -> Optional[object]:
    """
    Interactive wizard that fetches real highway service area data from OSM.

    Presents the registry of pre-registered highways, asks for a segment
    km range, calls HighwaySelector.select() in-process, prints the result,
    and returns a HighwaySelectionResult (or None if the user cancels).
    """
    try:
        from ev_charge_manager.highway_selection import HighwaySelector, HIGHWAY_REGISTRY
    except ImportError as exc:
        _err(f"Could not import highway_selection module: {exc}")
        return None

    _header("Real Highway Selection  (OpenStreetMap)")
    _info("Service area positions will be fetched from OpenStreetMap (no API key needed).")
    _info("Results are cached locally — subsequent runs are instant.")
    print()

    # ── Show the registry as a numbered list ──────────────────────────────
    hws = HighwaySelector.list_available_highways()
    print(_c("  Registered highways:", _BOLD))
    print()
    for i, hw in enumerate(hws, 1):
        line = (
            f"    {_c(str(i).rjust(2), _CYAN, _BOLD)}.  "
            f"{_c(hw['ref'], _BOLD, _YELLOW):<10}"
            f"  {hw['country']}   ~{hw['approx_km']:>5} km"
        )
        print(line)
    print()
    _info("You can also type a custom highway ref (e.g. A96) after choosing a bbox below.")
    print()

    # ── Highway selection ─────────────────────────────────────────────────
    raw = input(
        f"    Highway number [1–{len(hws)}] or highway ref (e.g. A5): "
    ).strip()

    custom_bbox: Optional[Tuple] = None
    if raw.isdigit():
        idx = int(raw)
        if not (1 <= idx <= len(hws)):
            _err("Number out of range.")
            return None
        hw_info = hws[idx - 1]
        highway_ref = hw_info["ref"]
        approx_km   = hw_info["approx_km"]
    else:
        highway_ref = raw.upper()
        if highway_ref not in HIGHWAY_REGISTRY:
            _warn(f"'{highway_ref}' is not in the built-in registry.")
            _info("You can still use it by providing a custom bounding box.")
            use_custom = _ask_bool("Provide a custom bounding box?", default=True)
            if not use_custom:
                return None
            _info("Bounding box format: south,west,north,east  (decimal degrees)")
            _info("Example for A96 Germany: 47.5,8.8,48.5,11.5")
            raw_bbox = _ask("Bounding box", default="47.5,8.8,48.5,11.5")
            try:
                parts = [float(x.strip()) for x in raw_bbox.split(",")]
                if len(parts) != 4:
                    raise ValueError
                custom_bbox = tuple(parts)
            except ValueError:
                _err("Invalid bounding box. Expected four comma-separated numbers.")
                return None
            approx_km = 300
        else:
            approx_km = HIGHWAY_REGISTRY[highway_ref]["approx_km"]

    # ── Segment km range ─────────────────────────────────────────────────
    _section("Segment Selection")
    _info(f"Highway approximate total length: ~{approx_km} km")
    start_km = _ask("Segment start (km)", default=0.0, cast=float)
    end_km   = _ask("Segment end   (km)", default=float(min(approx_km, 400)), cast=float)
    if start_km >= end_km:
        _err(f"start_km ({start_km}) must be less than end_km ({end_km}).")
        return None

    # ── Map option ───────────────────────────────────────────────────────
    show_map = _ask_bool("Show interactive map in browser?", default=False)

    # ── Run the selector ─────────────────────────────────────────────────
    _section("Fetching Data")
    print(
        f"  Fetching service area data for "
        f"{_c(highway_ref, _BOLD, _YELLOW)} "
        f"km {start_km:.0f}–{end_km:.0f} …"
    )
    print(_c("  (This may take a few seconds on the first run; results are cached.)", _DIM))
    print()

    try:
        selector = HighwaySelector()
        if custom_bbox is not None:
            result = selector.select_custom(
                highway_ref=highway_ref,
                bounding_box=custom_bbox,
                start_km=start_km,
                end_km=end_km,
                show_map=show_map,
            )
        else:
            result = selector.select(
                highway_ref=highway_ref,
                start_km=start_km,
                end_km=end_km,
                show_map=show_map,
            )
    except Exception as exc:
        _err(f"Highway selection failed: {exc}")
        _info("Falling back to manual configuration.")
        return None

    # ── Show result ───────────────────────────────────────────────────────
    _section("Service Areas Found")
    src_label = {
        "osm": _c("OpenStreetMap (live data)", _GREEN),
        "osm_expanded": _c("OpenStreetMap (expanded search)", _GREEN),
        "synthetic": _c("Synthetic (no OSM data found — evenly spaced)", _YELLOW),
    }.get(result.data_source, result.data_source)

    print(f"  Source : {src_label}")
    print(
        f"  Segment: {result.segment_start_km:.0f} – {result.segment_end_km:.0f} km  "
        f"({_c(f'{result.segment_length_km:.1f} km', _BOLD)})"
    )
    print(
        f"  Areas  : {_c(str(result.areas_in_segment), _BOLD, _GREEN)} stop areas"
    )
    print()

    for area in result.stop_areas:
        tag = _c(" [synthetic]", _YELLOW) if area.is_synthetic else ""
        print(
            f"    {_c(f'km {area.km_position:6.1f}', _CYAN)}  "
            f"{area.name}{tag}"
        )
    print()

    if not _ask_bool("Accept this configuration and continue?", default=True):
        return None

    return result


# ---------------------------------------------------------------------------
# Parameter collection — Simulation
# ---------------------------------------------------------------------------

def _collect_simulation() -> Optional[List[str]]:
    _header("Custom Simulation — Configuration")

    # ── Highway Source ────────────────────────────────────────────────────
    _section("Highway Source")
    _info("You can use real highway data from OpenStreetMap (service area positions")
    _info("are fetched automatically) or configure the highway manually.")
    print()
    use_real = _ask_bool("Use real highway from OpenStreetMap?", default=False)

    # Variables that differ between real and manual paths
    highway_length:      float             = 300.0
    num_stations:        int               = 5
    area_positions_str:  Optional[str]     = None   # pipe-joined km floats
    area_names_str:      Optional[str]     = None   # pipe-joined names

    if use_real:
        hw_result = _run_highway_selector()
        if hw_result is not None:
            highway_length     = hw_result.segment_length_km
            num_stations       = hw_result.areas_in_segment
            area_positions_str = "|".join(
                str(a.km_position) for a in hw_result.stop_areas
            )
            area_names_str = "|".join(a.name for a in hw_result.stop_areas)
            _ok(
                f"Locked: {num_stations} stations on "
                f"{hw_result.highway_ref}  "
                f"({highway_length:.1f} km)"
            )
        else:
            _warn("Highway selection cancelled — falling back to manual configuration.")
            use_real = False

    if not use_real:
        # ── Manual Highway ────────────────────────────────────────────────
        _section("Highway")
        highway_length    = _ask("Highway length (km)",         default=3000.0, cast=float)
        num_stations      = _ask("Number of charging stations", default=5,      cast=int)

    # ── Charging Station Config (always user-tunable) ─────────────────────
    _section("Charging Stations")
    chargers_per_station = _ask("Chargers per station",        default=4,      cast=int)
    waiting_spots        = _ask("Waiting spots per station",   default=6,      cast=int)

    # ── Traffic ───────────────────────────────────────────────────────────────
    _section("Traffic")
    _info("Available patterns: UNIFORM  RANDOM_POISSON  RUSH_HOUR  SINGLE_PEAK  PERIODIC  BURSTY")
    traffic_pattern  = _ask("Traffic pattern", default="RANDOM_POISSON",
                             choices=["UNIFORM", "RANDOM_POISSON", "RUSH_HOUR",
                                      "SINGLE_PEAK", "PERIODIC", "BURSTY"])
    vehicles_per_hour = _ask("Vehicles per hour",              default=60.0,   cast=float)
    duration          = _ask("Simulation duration (hours)",    default=4.0,    cast=float)
    seed_val          = _ask_optional("Random seed", default_display="random", cast=int)

    # ── Energy ────────────────────────────────────────────────────────────────
    _section("Energy Management  (optional — leave blank to disable)")
    grid_kw    = _ask_optional("Grid max power per station (kW)")
    solar_kw   = _ask_optional("Solar peak power per station (kW)")
    wind_kw    = _ask_optional("Wind base power per station (kW)")
    battery_kwh = _ask_optional("Battery capacity per station (kWh)")

    # ── Trackers ─────────────────────────────────────────────────────────────
    _section("Trackers / Verbosity")
    enable_station = _ask_bool("Enable station tracker?", default=False)
    enable_vehicle = _ask_bool("Enable vehicle tracker?",  default=False)
    no_viz         = _ask_bool("Skip visualizations?",     default=False)
    quiet          = _ask_bool("Quiet mode?",              default=False)

    # ── Output ────────────────────────────────────────────────────────────────
    _section("Output")
    output_dir = _ask("Output directory", default="./simulation_output")

    # ── Build command ─────────────────────────────────────────────────────────
    cmd = [
        sys.executable, "main.py",
        "--highway-length",        str(highway_length),
        "--num-stations",          str(num_stations),
        "--chargers-per-station",  str(chargers_per_station),
        "--waiting-spots",         str(waiting_spots),
        "--traffic-pattern",       traffic_pattern.upper(),
        "--vehicles-per-hour",     str(vehicles_per_hour),
        "--duration",              str(duration),
        "--output-dir",            output_dir,
    ]
    # Real highway: pass locked positions and names from OSM
    if area_positions_str is not None:
        cmd += ["--charging-area-positions", area_positions_str]
    if area_names_str is not None:
        cmd += ["--charging-area-names", area_names_str]
    if seed_val is not None:
        cmd += ["--seed", str(seed_val)]
    if grid_kw is not None:
        cmd += ["--energy-grid-max-kw", str(grid_kw)]
    if solar_kw is not None:
        cmd += ["--energy-solar-peak-kw", str(solar_kw)]
    if wind_kw is not None:
        cmd += ["--energy-wind-base-kw", str(wind_kw)]
    if battery_kwh is not None:
        cmd += ["--energy-battery-kwh", str(battery_kwh)]
    if enable_station:
        cmd.append("--enable-station-tracker")
    if enable_vehicle:
        cmd.append("--enable-vehicle-tracker")
    if no_viz:
        cmd.append("--no-visualize")
    if quiet:
        cmd.append("--quiet")

    return cmd


# ---------------------------------------------------------------------------
# Parameter collection — Hierarchical RL Training
# ---------------------------------------------------------------------------

def _collect_training() -> Optional[List[str]]:
    _header("Hierarchical RL Training — Configuration")

    # ── Mode ─────────────────────────────────────────────────────────────────
    _section("Training Mode")
    _info("SEQUENTIAL is the most stable (recommended for first runs).")
    mode = _ask(
        "Mode",
        default="sequential",
        choices=["sequential", "simultaneous", "curriculum", "frozen_micro"],
    )

    # ── Environment ───────────────────────────────────────────────────────────
    _section("Environment")
    n_agents       = _ask("Number of charging stations / agents", default=3,     cast=int)
    highway_length = _ask("Highway length (km)",                  default=300.0, cast=float)
    traffic_min    = _ask("Min traffic flow (veh/hr)",            default=30.0,  cast=float)
    traffic_max    = _ask("Max traffic flow (veh/hr)",            default=80.0,  cast=float)
    sim_duration   = _ask("Macro simulation duration (hours)",    default=24.0,  cast=float)
    robust         = _ask_bool("Use robust preset (more episodes)?", default=False)

    # ── Energy assets ────────────────────────────────────────────────────────
    _section("Station Energy Assets")
    battery_kwh = _ask("Battery capacity per station (kWh)",           default=500.0, cast=float)
    battery_kw  = _ask("Battery max charge/discharge power (kW)",      default=100.0, cast=float)
    solar_kw    = _ask("Solar installed capacity per station (kW)",     default=300.0, cast=float)
    wind_kw     = _ask("Wind installed capacity per station (kW)",      default=150.0, cast=float)

    # ── Cost / penalties ──────────────────────────────────────────────────────
    _section("Cost Parameters")
    stranded_pen   = _ask("Penalty per stranded vehicle ($)",   default=10000.0, cast=float)
    blackout_pen   = _ask("Penalty per blackout event ($)",     default=50000.0, cast=float)
    peak_price     = _ask("Grid peak price ($/kWh)",            default=0.25,    cast=float)
    shoulder_price = _ask("Grid shoulder price ($/kWh)",        default=0.15,    cast=float)
    offpeak_price  = _ask("Grid off-peak price ($/kWh)",        default=0.08,    cast=float)

    # ── Episodes ──────────────────────────────────────────────────────────────
    _section("Training Episodes  (ignored when --robust is set)")
    micro_episodes = _ask("Micro-RL episodes", default=1000, cast=int)
    macro_episodes = _ask("Macro-RL episodes", default=500,  cast=int)

    # ── Frozen-micro extras ───────────────────────────────────────────────────
    micro_load: Optional[str] = None
    if mode == "frozen_micro":
        _section("Frozen-Micro: pre-trained micro-agent path")
        micro_load = _ask("Path to micro-agent directory",
                          default="./models/hierarchical/micro_pretrained")

    # ── System ────────────────────────────────────────────────────────────────
    _section("System")
    save_dir   = _ask("Save directory", default="./models/hierarchical")
    no_viz     = _ask_bool("Skip training plots?", default=False)
    seed_val   = _ask_optional("Random seed", default_display="random", cast=int)

    # ── Build command ─────────────────────────────────────────────────────────
    cmd = [
        sys.executable, "hierarchical.py",
        "--mode",              mode,
        "--n-agents",          str(n_agents),
        "--highway-length",    str(highway_length),
        "--traffic-min",       str(traffic_min),
        "--traffic-max",       str(traffic_max),
        "--sim-duration",      str(sim_duration),
        "--battery-kwh",       str(battery_kwh),
        "--battery-kw",        str(battery_kw),
        "--solar-kw",          str(solar_kw),
        "--wind-kw",           str(wind_kw),
        "--stranded-penalty",  str(stranded_pen),
        "--blackout-penalty",  str(blackout_pen),
        "--grid-peak-price",   str(peak_price),
        "--grid-shoulder-price", str(shoulder_price),
        "--grid-offpeak-price",  str(offpeak_price),
        "--micro-episodes",    str(micro_episodes),
        "--macro-episodes",    str(macro_episodes),
        "--save-dir",          save_dir,
    ]
    if robust:
        cmd.append("--robust")
    if micro_load:
        cmd += ["--micro-load", micro_load]
    if no_viz:
        cmd.append("--no-visualize")
    if seed_val is not None:
        cmd += ["--seed", str(seed_val)]

    return cmd


# ---------------------------------------------------------------------------
# Parameter collection — Visualize
# ---------------------------------------------------------------------------

def _collect_visualize() -> Optional[List[str]]:
    options = [
        ("training", "Training plots  — from a hierarchical training run"),
        ("compare",  "Compare runs    — overlay multiple training runs"),
        ("curves",   "Demo curves     — sample training curve charts"),
        ("layout",   "Demo layout     — example highway layout chart"),
        ("back",     "← Back to main menu"),
    ]
    choice = _menu("Visualize", options)
    if choice == "back":
        return None

    cmd = [sys.executable, "visualize.py", choice]

    if choice == "training":
        _section("Training Plots")
        save_dir       = _ask("Training save directory", default="./models/hierarchical")
        highway_length = _ask("Highway length (km)",     default=300.0, cast=float)
        show           = _ask_bool("Show plots interactively?", default=False)
        cmd += ["--save-dir", save_dir, "--highway-length", str(highway_length)]
        if show:
            cmd.append("--show")

    elif choice == "compare":
        _section("Compare Runs")
        _info("Enter the directories to compare, separated by spaces.")
        dirs_raw = _ask("Directories", default="./models/sequential ./models/curriculum")
        dirs     = dirs_raw.split()
        output_dir     = _ask("Output directory (Enter = first dir)", default="")
        smoothing      = _ask("Smoothing window", default=10, cast=int)
        show           = _ask_bool("Show plot interactively?", default=False)
        cmd += ["--dirs"] + dirs
        if output_dir:
            cmd += ["--output-dir", output_dir]
        cmd += ["--smoothing-window", str(smoothing)]
        if show:
            cmd.append("--show")

    return cmd


# ---------------------------------------------------------------------------
# Parameter collection — Analysis tools
# ---------------------------------------------------------------------------

def _collect_analysis() -> Optional[List[str]]:
    options = [
        ("stranding",  "Traffic vs Stranding analysis"),
        ("optimizer",  "Infrastructure Optimizer (Pareto-front grid search)"),
        ("back",       "← Back to main menu"),
    ]
    choice = _menu("Analysis Tools", options)
    if choice == "back":
        return None

    if choice == "stranding":
        cmd = [sys.executable, "tools/traffic_stranding.py"]
    else:
        cmd = [sys.executable, "tools/infrastructure_optimizer.py"]

    return cmd


# ---------------------------------------------------------------------------
# Parameter collection — SUMO Validation
# ---------------------------------------------------------------------------

def _collect_sumo_validation() -> Optional[List[str]]:
    _header("SUMO Validation — Configuration")
    _info("Validates the trained hierarchical RL model using SUMO traffic simulation.")
    _info("The macro agent's best config builds the highway; the micro agent manages energy.")

    # ── Model ────────────────────────────────────────────────────────────────
    _section("Trained Model")
    model_dir = _ask("Model directory", default="./models/hierarchical")

    # ── SUMO Traffic ─────────────────────────────────────────────────────────
    _section("SUMO Traffic Parameters")
    days           = _ask("Simulation days",                      default=1,       cast=int)
    base_aadt      = _ask("Annual Average Daily Traffic (AADT)",  default=50000.0, cast=float)
    ev_penetration = _ask("EV penetration fraction (0-1)",        default=0.15,    cast=float)

    # ── Options ──────────────────────────────────────────────────────────────
    _section("Options")
    _info("Use sumo-gui to visually observe the traffic simulation.")
    sumo_binary     = _ask("SUMO binary", default="sumo", choices=["sumo", "sumo-gui"])
    compare_internal = _ask_bool("Also run internal sim for comparison?", default=False)
    no_plots        = _ask_bool("Skip plot generation?", default=False)

    # ── Output ───────────────────────────────────────────────────────────────
    _section("Output")
    output_dir = _ask("Output directory", default="./validation_output/sumo")
    seed_val   = _ask("Random seed", default=42, cast=int)

    # ── Build command ────────────────────────────────────────────────────────
    cmd = [
        sys.executable, "scripts/validate_sumo.py",
        "--model-dir",      model_dir,
        "--days",           str(days),
        "--base-aadt",      str(base_aadt),
        "--ev-penetration", str(ev_penetration),
        "--sumo-binary",    sumo_binary,
        "--output-dir",     output_dir,
        "--seed",           str(seed_val),
    ]
    if compare_internal:
        cmd.append("--compare-internal")
    if no_plots:
        cmd.append("--no-plots")

    return cmd


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _print_splash() -> None:
    width = 64
    lines = [
        "╔" + "═" * (width - 2) + "╗",
        "║" + " " * (width - 2) + "║",
        "║" + "  EV CHARGE MANAGER  —  Interactive Launcher".center(width - 2) + "║",
        "║" + " " * (width - 2) + "║",
        "║" + "  Highway EV charging simulation & RL training".center(width - 2) + "║",
        "║" + " " * (width - 2) + "║",
        "╚" + "═" * (width - 2) + "╝",
    ]
    print()
    for line in lines:
        print(_c(line, _BOLD, _CYAN))
    print()


MAIN_MENU_OPTIONS = [
    ("demo",      "Quick Demo          — 2-hour rush-hour simulation with defaults"),
    ("simulate",  "Custom Simulation   — configure highway, traffic, energy"),
    ("train",     "Hierarchical RL     — train micro + macro agents"),
    ("visualize", "Visualize Results   — training plots, comparison charts"),
    ("analysis",  "Analysis Tools      — stranding analysis, infra optimizer"),
    ("validate",  "SUMO Validation     — validate RL model with SUMO traffic sim"),
    ("exit",      "Exit"),
]


def main() -> None:
    os.chdir(Path(__file__).parent)   # always run from project root

    _print_splash()

    while True:
        choice = _menu("Main Menu", MAIN_MENU_OPTIONS)

        # ── Quick demo ───────────────────────────────────────────────────────
        if choice == "demo":
            cmd = [sys.executable, "main.py", "demo"]
            _section("Quick Demo")
            _info("2-hour RUSH_HOUR simulation · 3 stations · 80 veh/hr · seed 42")
            if _confirm_run(cmd):
                print()
                subprocess.run(cmd)

        # ── Custom simulation ─────────────────────────────────────────────────
        elif choice == "simulate":
            cmd = _collect_simulation()
            if cmd and _confirm_run(cmd):
                print()
                subprocess.run(cmd)

        # ── RL training ───────────────────────────────────────────────────────
        elif choice == "train":
            try:
                import torch  # noqa: F401
                import gymnasium  # noqa: F401
            except ImportError as e:
                _warn(f"Missing dependency: {e}")
                _info("Install with:  pip install torch gymnasium")
                print()
                continue
            cmd = _collect_training()
            if cmd and _confirm_run(cmd):
                print()
                subprocess.run(cmd)

        # ── Visualize ─────────────────────────────────────────────────────────
        elif choice == "visualize":
            cmd = _collect_visualize()
            if cmd and _confirm_run(cmd):
                print()
                subprocess.run(cmd)

        # ── Analysis tools ────────────────────────────────────────────────────
        elif choice == "analysis":
            cmd = _collect_analysis()
            if cmd and _confirm_run(cmd):
                print()
                subprocess.run(cmd)

        # ── SUMO Validation ──────────────────────────────────────────────────
        elif choice == "validate":
            try:
                from ev_charge_manager.sumo import check_sumo_available
                if not check_sumo_available():
                    raise ImportError("SUMO not found")
            except (ImportError, RuntimeError):
                _warn("SUMO is not installed or SUMO_HOME is not set.")
                _info("Install: pip install eclipse-sumo traci sumolib")
                _info("Then:    export SUMO_HOME=/usr/local/lib/python3.11/dist-packages/sumo")
                print()
                continue
            cmd = _collect_sumo_validation()
            if cmd and _confirm_run(cmd):
                print()
                subprocess.run(cmd)

        # ── Exit ──────────────────────────────────────────────────────────────
        elif choice == "exit":
            print()
            _ok("Goodbye!")
            print()
            sys.exit(0)

        # After each run, pause so the user can read the output
        if choice not in ("exit",):
            print()
            input(_c("  Press Enter to return to the main menu…", _DIM))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        _ok("Interrupted — goodbye!")
        sys.exit(0)

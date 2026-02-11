"""
SUMO/TraCI integration for EV charging infrastructure validation.

Provides a higher-fidelity traffic simulation backend using SUMO
(Simulation of Urban Mobility) to validate hierarchical RL models.
The macro agent's infrastructure config is translated into a SUMO
highway network, TraCI runs the traffic, and the micro agent manages
energy step-by-step.

Requires SUMO to be installed and SUMO_HOME environment variable set.
Install SUMO: https://sumo.dlr.de/docs/Installing/index.html
"""

import os
import sys


def check_sumo_available() -> bool:
    """Check if SUMO is installed and accessible via SUMO_HOME."""
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home is None:
        return False
    tools_path = os.path.join(sumo_home, "tools")
    if tools_path not in sys.path:
        sys.path.append(tools_path)
    try:
        import traci  # noqa: F401
        return True
    except ImportError:
        return False


def require_sumo():
    """Raise an error with installation instructions if SUMO is not available."""
    if not check_sumo_available():
        raise RuntimeError(
            "SUMO is not installed or SUMO_HOME is not set.\n"
            "Install SUMO:\n"
            "  Ubuntu/Debian: sudo apt-get install -y sumo sumo-tools\n"
            "  macOS:         brew install sumo\n"
            "  Other:         https://sumo.dlr.de/docs/Installing/index.html\n"
            "Then set: export SUMO_HOME=/usr/share/sumo  (or your install path)"
        )


SUMO_AVAILABLE = check_sumo_available()

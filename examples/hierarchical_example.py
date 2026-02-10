"""hierarchical_example – hierarchical RL training scenarios (SEQUENTIAL / CURRICULUM).

Wrapper for the root-level hierarchical_use_example module.
Run directly:  python examples/hierarchical_example.py scenario_1
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    import runpy
    runpy.run_path(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "hierarchical_use_example.py"),
        run_name="__main__",
    )

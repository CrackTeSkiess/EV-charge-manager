"""ppo_example – PPO training use-case scenarios.

Wrapper for the root-level ppo_use_case_example module.
Run directly:  python examples/ppo_example.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    import runpy
    runpy.run_path(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "ppo_use_case_example.py"),
        run_name="__main__",
    )

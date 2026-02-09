---
globs: "**/simulation.py"
description: Prevents memory overload and improves performance for large-scale
  simulations by distributing the workload across multiple processes. This
  applies specifically to simulation.py files.
alwaysApply: false
---

If simulating >5,000 vehicles, always use multiprocessing to distribute workload across CPU cores.
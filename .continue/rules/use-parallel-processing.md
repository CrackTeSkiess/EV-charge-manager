---
globs: '"**/simulation.py", "**/*.py"'
description: Prevents memory overload and improves performance for large-scale
  simulations by distributing the workload across multiple processes.
alwaysApply: false
---

If simulating >5,000 vehicles, always use multiprocessing to distribute workload across CPU cores.
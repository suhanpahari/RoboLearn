"""Habitat task construction (PointNav / ImageNav). STUB.

Runs only in the roborto-hab conda env; imports habitat lazily so the
roborto-gym env never needs Habitat installed.

TODO(claude-code): make_habitat_env(cfg) -> a habitat-lab env from a task config
with RGB/depth sensors per cfg.env. Verify EGL rendering works on the target
device first - MIG slices may not render (see docs/PLAN.md section 2.3).
"""

from __future__ import annotations

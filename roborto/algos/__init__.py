"""Algorithms.

Build-vs-reuse policy (docs/PLAN.md section 3.2):
  * Baselines reuse trusted libraries - Stable-Baselines3 + sb3-contrib for
    manipulation SAC/HER, Habitat-Lab native DD-PPO for navigation.
  * In-repo code is reserved for the Spine A contribution: HER relabeling
    (roborto/buffers/her.py) and frozen-encoder integration
    (roborto/models/encoders/).
Thin wrappers that adapt those libraries to roborto's config and logging live here.
"""

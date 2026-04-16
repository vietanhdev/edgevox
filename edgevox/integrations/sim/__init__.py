"""Optional simulator adapters.

Each adapter is import-guarded — users who don't install the matching
optional dependency never import the adapter module. Available:

- :mod:`edgevox.integrations.sim.irsim` — IR-SIM (Python 2D robot sim,
  needs ``pip install edgevox[sim]`` or ``pip install ir-sim>=2.9``).
- :mod:`edgevox.integrations.sim.mujoco_arm` — MuJoCo 3D tabletop arm
  (needs ``pip install 'edgevox[sim-mujoco]'`` or ``pip install 'mujoco>=3.2'``).

``ToyWorld`` (shipped in :mod:`edgevox.agents.sim`) is the stdlib-only
reference; these adapters are the next step up on the simulation tier
ladder.
"""

#!/usr/bin/env python3
"""Upload Unitree G1 / H1 MuJoCo scenes to ``nrl-ai/edgevox-models``.

Mirrors the Franka pattern already used by the tabletop arm: each
humanoid scene ships under ``mujoco_scenes/<name>/`` in the HF repo,
and :func:`~edgevox.integrations.sim.mujoco_humanoid._fetch_from_hf`
downloads just those paths on first use.

Usage::

    huggingface-cli login      # one-time
    python scripts/upload_humanoids.py
    # or to upload just one model:
    python scripts/upload_humanoids.py --model unitree_g1

The script sparse-clones ``google-deepmind/mujoco_menagerie`` into a
temp dir, copies the ``unitree_g1/`` and ``unitree_h1/`` subdirectories
verbatim (including meshes and LICENSE), and pushes them under the
``mujoco_scenes/<name>/`` prefix in the HF repo.

Both Unitree G1 and H1 in Menagerie are BSD-3-Clause licensed by
Unitree Robotics — the LICENSE file gets uploaded alongside so the
attribution stays with the assets.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "nrl-ai/edgevox-models"
MENAGERIE_REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"

HUMANOIDS = {
    "unitree_g1": {
        "menagerie_subdir": "unitree_g1",
        "license": "BSD-3-Clause",
        "attribution": "Unitree Robotics / MuJoCo Menagerie",
    },
    "unitree_h1": {
        "menagerie_subdir": "unitree_h1",
        "license": "BSD-3-Clause",
        "attribution": "Unitree Robotics / MuJoCo Menagerie",
    },
}


def _sparse_clone(tmp: Path, subdirs: list[str]) -> Path:
    clone_dir = tmp / "mujoco_menagerie"
    subprocess.check_call(
        [
            "git",
            "clone",
            "--depth=1",
            "--filter=blob:none",
            "--sparse",
            MENAGERIE_REPO,
            str(clone_dir),
        ]
    )
    subprocess.check_call(
        ["git", "-C", str(clone_dir), "sparse-checkout", "set", *subdirs],
    )
    return clone_dir


def upload_one(api: HfApi, name: str, source_root: Path) -> None:
    meta = HUMANOIDS[name]
    subdir_path = source_root / meta["menagerie_subdir"]
    scene = subdir_path / "scene.xml"
    if not scene.exists():
        raise SystemExit(f"scene.xml missing at {scene}")

    # Drop an EdgeVox-specific README beside the scene so browsers of
    # the HF repo know what they're looking at.
    readme = subdir_path / "EDGEVOX_README.md"
    readme.write_text(
        f"# {name}\n\n"
        f"Mirror of [google-deepmind/mujoco_menagerie/{meta['menagerie_subdir']}]"
        f"(https://github.com/google-deepmind/mujoco_menagerie/tree/main/"
        f"{meta['menagerie_subdir']}) bundled for fast downloads by\n"
        f"`edgevox.integrations.sim.mujoco_humanoid.MujocoHumanoidEnvironment`.\n\n"
        f"**License**: {meta['license']} — see `LICENSE` in this directory.\n"
        f"**Attribution**: {meta['attribution']}.\n\n"
        f"Entry point: `scene.xml`.\n",
        encoding="utf-8",
    )

    hf_target = f"mujoco_scenes/{name}"
    print(f">>> uploading {subdir_path} -> {REPO_ID}/{hf_target}")
    api.upload_folder(
        folder_path=str(subdir_path),
        path_in_repo=hf_target,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message=f"Add {name} humanoid scene ({meta['attribution']}, {meta['license']})",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload Unitree humanoids to the EdgeVox HF repo.")
    parser.add_argument(
        "--model",
        choices=[*HUMANOIDS, "all"],
        default="all",
        help="Which humanoid to upload (default: all).",
    )
    args = parser.parse_args()

    wanted = list(HUMANOIDS) if args.model == "all" else [args.model]
    subdirs = [HUMANOIDS[n]["menagerie_subdir"] for n in wanted]

    api = HfApi()
    # Ensure the repo exists — idempotent.
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="edgevox-humanoids-") as tmp:
        root = _sparse_clone(Path(tmp), subdirs)
        for name in wanted:
            upload_one(api, name, root)

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

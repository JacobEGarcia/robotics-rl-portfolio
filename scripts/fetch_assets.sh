#!/usr/bin/env bash
# Fetch the third-party assets this repository depends on but does not vendor.
#
#   MuJoCo Menagerie  ~2.3 GB  robot models (Franka Panda, LEAP/Shadow hands,
#                               MS-Human-700 musculoskeletal body)
#   DROID episodes    ~3.9 MB  32 real Franka trajectories, 632 s at 15 Hz
#
# Run from the repository root:
#     bash scripts/fetch_assets.sh
set -euo pipefail

mkdir -p assets/data/droid

# ---------------------------------------------------------------- Menagerie
if [ -d assets/menagerie/.git ]; then
  echo "Menagerie already present, skipping."
else
  echo "Cloning MuJoCo Menagerie (shallow, ~2.3 GB)..."
  git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git assets/menagerie
fi

# --------------------------------------------------------------------- DROID
# cadene/droid_1.0.1 has no bulk parquet export, but episodes are individually
# addressable. 32 of them is all this project needs and costs under 4 MB.
BASE="https://huggingface.co/datasets/cadene/droid_1.0.1/resolve/main/data/chunk-000"
echo "Fetching 32 DROID episodes..."
for i in $(seq 0 31); do
  n=$(printf "%06d" "$i")
  out="assets/data/droid/episode_${n}.parquet"
  [ -f "$out" ] || curl -sL -o "$out" "${BASE}/episode_${n}.parquet"
done

echo
echo "Done."
echo "  Menagerie: $(du -sh assets/menagerie 2>/dev/null | cut -f1)"
echo "  DROID:     $(ls assets/data/droid/*.parquet 2>/dev/null | wc -l) episodes"

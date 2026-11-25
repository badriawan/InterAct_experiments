#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$PROJECT_ROOT/custom_data/setup/activate_interact.sh"
BUILD_ROOT="${1:-/tmp/interact-stage4-build}"
WHEELS="$PROJECT_ROOT/custom_data/setup/wheels"
mkdir -p "$WHEELS"
export FORCE_CUDA=1

# Rebuild helper for source trees already cloned at the recorded revisions.
# This script does not clone or patch sources. Preparation is documented in custom_data/README.md.
# Source revisions are recorded in custom_data/manifests/source_revisions.json.
# PointNet2's build-time architecture adjustment is in setup/pointnet2_arch.patch.
for package in pytorch3d pointnet2 human_body_prior chumpy chamfer_distance bps_torch; do
    package_dir="$BUILD_ROOT/$package"
    wheel_pattern="$package"
    if [ "$package" = pointnet2 ]; then
        package_dir="$package_dir/pointnet2_ops_lib"
        wheel_pattern=pointnet2_ops
    fi
    python -m pip wheel --no-deps --no-build-isolation "$package_dir" -w "$WHEELS"
    python -m pip install --no-deps "$WHEELS"/"$wheel_pattern"-*.whl
done

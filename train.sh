#!/usr/bin/ nv bash
set -euo pipefail

placement="${PLACEMENT:-bottleneck}"
if [[ $# -gt 0 && "${1}" != -* ]]; then
  placement="${1}"
  shift
fi

case "${placement}" in
  none|output|bottleneck) ;;
  *)
    echo "Usage: $0 [none|output|bottleneck] [additional trainer arguments...]" >&2
    exit 2
    ;;
esac

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python}"
seed="${SEED:-42}"
run_name="${RUN_NAME:-${placement}-seed${seed}}"
run_dir="${LOG_ROOT:-${project_root}/lightning_logs/attention-placement}/${run_name}"

echo "Attention placement: ${placement}"
echo "Data: ${project_root}/data/brain"
echo "Logs/checkpoints: ${run_dir}"

cd "${project_root}"
exec "${python_bin}" fastmri_examples/varnet/train_varnet_demo.py \
  --data_path "${project_root}/data/brain" \
  --default_root_dir "${run_dir}" \
  --seed "${seed}" \
  --coil_attention_placement "${placement}" \
  "$@"

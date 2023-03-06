set -euxo pipefail

python3 -m pylint torch_musa/ --rcfile="$(dirname "$0")"/pylintrc

set -euxo pipefail

python -m pylint musa_torch/ --rcfile="$(dirname "$0")"/pylintrc

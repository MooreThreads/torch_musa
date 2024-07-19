set -euxo pipefail

python3 -m pylint torch_musa/ --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint --recursive=y tests/ --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint --recursive=y tools/codegen --rcfile="$(dirname "$0")"/pylintrc

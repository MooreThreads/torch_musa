#!/usr/bin/env bash
set -euo pipefail

FORMATTER=black
INPLACE=false
FORMAT_ALL=true
REVISION=

while (( $# )); do
    case "$1" in
        -i)
            INPLACE=true
            shift 1
            ;;
        -c)
            FORMAT_ALL=false
            if [ "$#" -lt 2  ]; then
                REVISION="HEAD~1"
                shift 1
            else
                REVISION=$2
                shift 2
            fi
            ;;
        *)
            echo "Usage: tools/lint/git-format.sh [-i] [-c <commit>]"
            echo ""
            echo "-i: Format all Python files in the in-place way."
            echo "-c: Format Python files using ${FORMATTER} on changes since <commit> or against a certain branch"
            echo "Examples:"
            echo "- Compare against the last commit: tools/lint/git-format.sh -c [HEAD~1]"
            echo "- Compare against the main branch: tools/lint/git-format.sh -c main"
            exit 1
            ;;
    esac
done

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

if [ ! -x "$(command -v ${FORMATTER})" ]; then
    echo "Cannot find command: ${FORMATTER}, please install it!"
    exit 1
else
    VERSION=$(${FORMATTER} --version)
    echo "Format Python using ${FORMATTER} version: $VERSION"
fi

if [[ "$FORMAT_ALL" == "true" ]]; then
    # Format all Python files.
    FILES=$(git ls-files | grep -E '\.py$')
    echo "Formatting all Python files"
else
    IFS=$'\n' read -a FILES -d'\n' < <(git diff --name-only --diff-filter=ACMRTUX $REVISION -- "*.py" "*.pyi") || true
    echo "Read returned $?"
    if [ -z ${FILES+x} ]; then
        echo "No changes in Python files"
        exit 0
    fi
    echo "Files: $FILES"
fi

if [[ "$INPLACE" == "true" ]]; then
    if [[ -n ${REVISION} ]]; then
        echo "Running ${FORMATTER} on Python files against revision" $REVISION:
    fi
    python3 -m ${FORMATTER} ${FILES[@]}
else
    echo "Running ${FORMATTER} in checking mode"
    python3 -m ${FORMATTER} --diff --check ${FILES[@]}
fi

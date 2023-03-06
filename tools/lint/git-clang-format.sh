#!/usr/bin/env bash
# This file provides a tool that formats the C++ files. It is mainly borrowed
# from the TVM project.
set -e
set -u
set -o pipefail


INPLACE_FORMAT=${INPLACE_FORMAT:=false}
LINT_ALL_FILES=true
REVISION=$(git rev-list --max-parents=0 HEAD)

while (( $# )); do
    case "$1" in
        -i)
            INPLACE_FORMAT=true
            shift 1
            ;;
        --rev)
            LINT_ALL_FILES=false
            REVISION=$2
            shift 2
            ;;
        *)
            echo "Usage: tests/lint/git-clang-format.sh [-i] [--rev <commit>]"
            echo ""
            echo "Run clang-format on files that changed since <commit> or on all files in the repo"
            echo "Examples:"
            echo "- Compare last one commit: tools/lint/git-clang-format.sh --rev HEAD~1"
            echo "- Compare against upstream/master: tools/lint/git-clang-format.sh --rev upstream/main"
            echo "The -i will use black to format files in-place instead of checking them."
            exit 1
            ;;
    esac
done


cleanup()
{
  rm -rf /tmp/$$.clang-format.txt
}
trap cleanup 0

CLANG_FORMAT=clang-format-11

if [ -x "$(command -v clang-format-11)" ]; then
    CLANG_FORMAT=clang-format-11
elif [ -x "$(command -v clang-format)" ]; then
    echo "clang-format might be different from clang-format-11, expect potential difference."
    CLANG_FORMAT=clang-format
else
    echo "Cannot find clang-format-11"
    exit 1
fi

# Print out specific version
${CLANG_FORMAT} --version

if [[ "$INPLACE_FORMAT" == "true" ]]; then
    echo "Running inplace git-clang-format against $REVISION"
    git-${CLANG_FORMAT} --extensions h,inc,c,cpp,mu,muh --binary=${CLANG_FORMAT} "$REVISION"
    exit 0
fi

if [[ "$LINT_ALL_FILES" == "true" ]]; then
    echo "Running git-clang-format against all C++ files"
    git-${CLANG_FORMAT} --diff --extensions h,inc,c,cpp,mu,muh --binary=${CLANG_FORMAT} "$REVISION" 1> /tmp/$$.clang-format.txt
else
    echo "Running git-clang-format against $REVISION"
    git-${CLANG_FORMAT} --diff --extensions h,inc,c,cpp,mu,muh --binary=${CLANG_FORMAT} "$REVISION" 1> /tmp/$$.clang-format.txt
fi

echo "---------clang-format log----------"
cat /tmp/$$.clang-format.txt
echo ""
if grep --quiet -E "diff" < /tmp/$$.clang-format.txt; then
    echo "clang-format lint error found. Consider running clang-format-11 on these files to fix them."
    exit 1
fi

#!/bin/bash
# Wrap mcc with ccache, plugged into the MUSA build via MUSA_MCC_EXECUTABLE.

CCACHE_BIN=${CCACHE_BIN:-ccache}
MCC_BIN=${MCC_BIN:-/usr/local/musa/bin/mcc}

# depend_mode needs the compiler to emit a depfile, but run_mcc.cmake omits -MD
# on the compile pass; inject it (only for a real compile, if not already there)
# so we don't fall back to the slower preprocessor mode.
out=""
has_md=0
prev=""
has_c=0
for arg in "$@"; do
  case "$prev" in
    -o) out="$arg" ;;
  esac
  case "$arg" in
    -c) has_c=1 ;;
    -MD|-MMD) has_md=1 ;;
  esac
  prev="$arg"
done

extra_args=()
if [[ "$has_c" == "1" ]] && [[ -n "$out" ]] && [[ "$has_md" == "0" ]]; then
  extra_args=("-MD" "-MF" "${out}.d")
fi

exec "${CCACHE_BIN}" "${MCC_BIN}" "${extra_args[@]}" "$@"

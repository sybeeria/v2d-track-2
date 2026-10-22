#!/usr/bin/env bash
# TSV rows: sequence<TAB>frames<TAB>absolute video<TAB>prompts<TAB>metric mesh.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
manifest=${1:?TSV manifest}; shift
failures=0
while IFS=$'\t' read -r sequence frames video prompts mesh; do
  [[ -z $sequence || $sequence = \#* ]] && continue
  if ! bash "$HERE/run.sh" "$@" --sequence "$sequence" --frames "$frames" --video "$video" --prompts "$prompts" --mesh "$mesh"; then
    echo "FAIL $sequence" >&2; failures=$((failures+1))
  fi
done < "$manifest"
[[ $failures -eq 0 ]]

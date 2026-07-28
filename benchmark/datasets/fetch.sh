#!/usr/bin/env bash
# Fetch the standard ANN benchmark datasets.
#
#   ./fetch.sh siftsmall     5 MB    10k vectors, for validating the oracle
#   ./fetch.sh sift        168 MB     1M vectors, 128 dimensions
#   ./fetch.sh gist        2.6 GB     1M vectors, 960 dimensions (3840 B each)
#
# Source: http://corpus-texmex.irisa.fr/
set -euo pipefail

dataset="${1:-siftsmall}"
case "$dataset" in
  siftsmall|sift|gist) ;;
  *) echo "unknown dataset '$dataset' (expected siftsmall, sift or gist)" >&2; exit 1 ;;
esac

cd "$(dirname "$0")"
archive="${dataset}.tar.gz"

if [ -d "$dataset" ]; then
  echo "$dataset/ already exists; delete it to re-download"
  exit 0
fi

if [ ! -f "$archive" ]; then
  echo "downloading $archive..."
  curl -fL --retry 3 -o "$archive" \
    "ftp://ftp.irisa.fr/local/texmex/corpus/$archive"
fi

echo "extracting..."
tar -xzf "$archive"

echo "done. Validate the brute-force oracle against the published ground truth:"
echo "  cargo run --release --example ann_groundtruth benchmark/datasets/$dataset/$dataset"

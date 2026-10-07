#!/bin/sh
# Extract the Chakra Open Trace Library download (Google-Drive split zips; parts hold disjoint files
# under common top dirs, so all parts go into ONE directory). No unzip on this host -> python zipfile.
set -e
SRC=${SRC:-$HOME/chakra-traces}                  # the downloaded zips
DST=${DST:-/scratch/$USER/chakra-traces}
mkdir -p "$DST"
python3 - "$SRC" "$DST" <<'PY'
import zipfile, glob, sys, os
src, dst = sys.argv[1], sys.argv[2]
for f in sorted(glob.glob(os.path.join(src, "Mixtral-*.zip")) + glob.glob(os.path.join(src, "Llama3-*.zip"))):   # the paper's traces only
    zipfile.ZipFile(f).extractall(dst)
# Mixtral-8x7B raw host (PyTorch ET JSON) + device (Kineto JSON) traces are in a nested zip
inner = os.path.join(dst, "Mixtral/Mixtral-8x7B/nemo-chakra-mixtral-8x7B-traces.zip")
zipfile.ZipFile(inner).extractall(os.path.join(dst, "Mixtral/Mixtral-8x7B/nemo_raw"))
PY

#!/bin/bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Download latest models from https://github.com/ultralytics/assets/releases
# Example usage: bash ultralytics_local/data/scripts/download_weights.sh
# parent
# └── weights
#     ├── yolov8n.pt  ← downloads here
#     ├── yolov8s.pt
#     └── ...

python << EOF
from ultralytics.utils.downloads import attempt_download_asset

assets = [f"yolov8{size}{suffix}.pt" for size in "nsmlx" for suffix in ("", "-cls", "-seg", "-pose")]
for x in assets:
    attempt_download_asset(f"weights/{x}")
EOF

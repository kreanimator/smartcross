"""Everything that turns camera frames into traffic estimates.

Ultralytics keeps its settings in one file per user and can rewrite it, so this project
uses its own copy. Offline mode stops it downloading weights or sending usage data.
"""

import os
from pathlib import Path

SETTINGS_FOLDER = Path(__file__).parent.parent / ".ultralytics"
SETTINGS_FOLDER.mkdir(exist_ok=True)  # Ultralytics treats a missing folder as unwritable.
os.environ.setdefault("YOLO_CONFIG_DIR", str(SETTINGS_FOLDER))
os.environ.setdefault("YOLO_OFFLINE", "true")

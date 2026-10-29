"""CropCare — Next-Generation Agricultural Intelligence Platform.

Main entrypoint for Streamlit web application.
Delegates to app_green_ui.py for the full-featured, guided 4-screen dashboard.
"""

from __future__ import annotations

import os
import runpy
import sys

# Ensure workspace root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Execute the master agronomic dashboard
ui_script = os.path.join(current_dir, "app_green_ui.py")
runpy.run_path(ui_script, run_name="__main__")

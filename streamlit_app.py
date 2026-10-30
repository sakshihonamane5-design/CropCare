"""CropCare — Primary Streamlit Cloud & Hosting Entrypoint."""

import runpy
from pathlib import Path

target = Path(__file__).resolve().parent / "app_green_ui.py"
runpy.run_path(str(target), run_name="__main__")


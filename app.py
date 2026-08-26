"""
Standalone launcher for the Gradio UI (local development / a dedicated
Gradio-only Railway service). All the actual logic lives in ui.py and
src/inference.py, which are also reused by api.py (FastAPI).
"""

import os
from ui import demo

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
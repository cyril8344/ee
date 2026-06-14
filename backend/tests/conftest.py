"""Pytest configuration: make the backend package importable and isolate the DB."""
import os
import sys
import tempfile

# Ensure `import strategy`, `import risk_manager`, ... resolve to backend/.
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Use a throwaway SQLite file and the offline synthetic feed for all tests.
os.environ.setdefault("XAU_DB_PATH", os.path.join(tempfile.gettempdir(), "xau_bot_test.db"))
os.environ.setdefault("XAU_DATA_PROVIDER", "synthetic")

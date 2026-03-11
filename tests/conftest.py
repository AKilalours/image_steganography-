# conftest.py — pytest configuration
import sys
from pathlib import Path

# Ensure the project root is on the path when running tests directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tier2_acquisition.fetch_controller import FetchController

fc = FetchController()
print("FetchController instance created")
print(f"Has fetch_air_quality method: {hasattr(fc, 'fetch_air_quality')}")
print(f"All methods: {[m for m in dir(fc) if not m.startswith('_')]}")

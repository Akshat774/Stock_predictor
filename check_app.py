import sys
from pathlib import Path

# Ensure root is in path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print("Checking syntax and imports...")

try:
    import app
    print("app.py imported successfully")
except Exception as e:
    print(f"Error in app.py: {e}")

pages = ["dashboard", "stock_analysis", "prediction", "model_performance", "_shared"]
for page in pages:
    try:
        # We need to simulate the streamlit context if necessary, but simple syntax check works
        import importlib
        importlib.import_module(f"pages.{page}")
        print(f"pages/{page}.py imported successfully")
    except Exception as e:
        print(f"Error in pages/{page}.py: {e}")

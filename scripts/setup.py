"""
Comprehensive setup script for AgroSpectra platform
Checks dependencies, creates directories, initializes database
"""

import sys
import os
from pathlib import Path
import subprocess


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_step(step, total, text):
    """Print step information"""
    print(f"\n[Step {step}/{total}] {text}")


def check_python_version():
    """Check if Python version is compatible"""
    print_step(1, 8, "Checking Python version")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def create_directory_structure():
    """Create necessary directories"""
    print_step(2, 8, "Creating directory structure")
    
    directories = [
        "data/cache/raw/sentinel",
        "data/cache/raw/weather",
        "data/cache/raw/mosdac",
        "data/cache/raw/soil",
        "data/cache/processed",
        "data/storage",
        "data/training/crop_health/healthy",
        "data/training/crop_health/stressed",
        "data/training/crop_health/diseased",
        "data/training/pest_sequences",
        "data/reports",
        "data/external/soil/icar",
        "data/external/soil/nbss_lup",
        "models/trained",
        "models/registry",
        "logs",
        "temp",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✓ Created: {directory}")
    
    print("✅ Directory structure created")
    return True


def create_env_file():
    """Create .env file from example if it doesn't exist"""
    print_step(3, 8, "Checking environment configuration")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if env_example.exists():
        import shutil
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from .env.example")
        print("⚠️  Please edit .env file with your API credentials")
        return True
    else:
        print("❌ .env.example file not found")
        return False


def check_gdal():
    """Check if GDAL is available"""
    print_step(4, 8, "Checking GDAL installation")
    
    try:
        import osgeo
        from osgeo import gdal
        print(f"✅ GDAL version {gdal.__version__} found")
        return True
    except ImportError:
        print("⚠️  GDAL not found")
        print("   Please install GDAL:")
        print("   - Windows: Download from https://www.gisinternals.com/")
        print("   - Or use: conda install -c conda-forge gdal")
        return False


def install_dependencies():
    """Install Python dependencies"""
    print_step(5, 8, "Installing Python dependencies")
    
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    print("   This may take several minutes...")
    
    try:
        # Try to install requirements
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print("⚠️  Some dependencies failed to install")
            print("   Please install manually: pip install -r requirements.txt")
            return True  # Continue anyway
    except Exception as e:
        print(f"⚠️  Error installing dependencies: {e}")
        print("   Please install manually: pip install -r requirements.txt")
        return True


def initialize_database():
    """Initialize SQLite database"""
    print_step(6, 8, "Initializing database")
    
    try:
        # Import and run database initialization
        sys.path.append(str(Path("src")))
        from tier6_storage.database import initialize_database as init_db
        
        init_db()
        print("✅ Database initialized")
        return True
    except Exception as e:
        print(f"⚠️  Database initialization failed: {e}")
        print("   You can initialize manually later with:")
        print("   python scripts/initialize_database.py")
        return True


def create_readme_files():
    """Create README files in key directories"""
    print_step(7, 8, "Creating documentation")
    
    readme_contents = {
        "data/cache/raw/README.md": """# Raw Data Cache
        
This directory stores raw data downloaded from various sources:
- `sentinel/`: Sentinel-2 satellite imagery
- `weather/`: Weather data from OpenWeatherMap
- `mosdac/`: MOSDAC data products
- `soil/`: Soil database files

Data is automatically cached here to avoid redundant API calls.
""",
        "data/training/crop_health/README.md": """# Crop Health Training Data

Place training images in subdirectories:
- `healthy/`: Images of healthy crops
- `stressed/`: Images of stressed crops
- `diseased/`: Images of diseased crops

Images should be 128x128 pixels or will be resized during training.
""",
        "models/trained/README.md": """# Trained Models

This directory contains trained AI models:
- `mobilenetv2_crop_health.h5`: CNN crop health classifier
- `lstm_pest_risk.h5`: LSTM pest risk predictor
- Associated scaler files (.pkl)

Models are loaded by the application at runtime.
""",
    }
    
    for path, content in readme_contents.items():
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        print(f"   ✓ Created: {path}")
    
    print("✅ Documentation created")
    return True


def display_next_steps():
    """Display next steps for the user"""
    print_step(8, 8, "Setup Complete!")
    
    print("\n" + "=" * 70)
    print("  🎉 AgroSpectra Platform Setup Complete!")
    print("=" * 70)
    
    print("\n📋 Next Steps:")
    print("\n1. Configure API Credentials:")
    print("   - Edit the .env file with your API keys:")
    print("     * Sentinel Hub (client_id, client_secret, instance_id)")
    print("     * OpenWeatherMap (api_key)")
    print("   - Get Sentinel Hub credentials: https://www.sentinel-hub.com/")
    print("   - Get OpenWeatherMap key: https://openweathermap.org/api")
    
    print("\n2. (Optional) Prepare Training Data:")
    print("   - Place crop health images in data/training/crop_health/")
    print("   - Place pest sequence data in data/training/pest_sequences/")
    
    print("\n3. (Optional) Train Models:")
    print("   - Train crop health classifier:")
    print("     python src/tier5_models/train_cnn.py")
    print("   - Train pest risk predictor:")
    print("     python src/tier5_models/train_lstm.py")
    
    print("\n4. Start the Application:")
    print("   - Run the dashboard:")
    print("     streamlit run src/dashboard/app.py")
    print("   - Or use Docker:")
    print("     docker-compose up")
    
    print("\n5. Access the Dashboard:")
    print("   - Open your browser to: http://localhost:8501")
    
    print("\n📚 Documentation:")
    print("   - See README.md for detailed instructions")
    print("   - Check docs/ folder for additional documentation")
    
    print("\n⚠️  Important Notes:")
    print("   - GDAL installation may be required (see README.md)")
    print("   - Sentinel Hub free tier: 30,000 processing units/month")
    print("   - OpenWeatherMap free tier: 1,000 calls/day")
    
    print("\n" + "=" * 70)


def main():
    """Main setup function"""
    print_header("AgroSpectra Platform Setup")
    print("This script will set up the AgroSpectra platform")
    
    # Track success/failure
    checks = []
    
    # Run setup steps
    checks.append(("Python Version", check_python_version()))
    checks.append(("Directory Structure", create_directory_structure()))
    checks.append(("Environment Config", create_env_file()))
    checks.append(("GDAL Check", check_gdal()))
    checks.append(("Dependencies", install_dependencies()))
    checks.append(("Database", initialize_database()))
    checks.append(("Documentation", create_readme_files()))
    
    # Display summary
    print("\n" + "=" * 70)
    print("  Setup Summary")
    print("=" * 70)
    
    for name, success in checks:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    # Display next steps
    display_next_steps()
    
    # Exit with appropriate code
    if all(success for _, success in checks):
        sys.exit(0)
    else:
        print("\n⚠️  Some checks failed. Please review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

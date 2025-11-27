#!/usr/bin/env python3
"""
DiamondHood System Test Script
Validates installation and runs basic functionality tests
"""

import sys
from pathlib import Path

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_header(text):
    """Print formatted header."""
    print(f"\n{BLUE}{'='*60}")
    print(f"{text}")
    print(f"{'='*60}{RESET}\n")


def print_success(text):
    """Print success message."""
    print(f"{GREEN}✓ {text}{RESET}")


def print_error(text):
    """Print error message."""
    print(f"{RED}✗ {text}{RESET}")


def print_warning(text):
    """Print warning message."""
    print(f"{YELLOW}⚠ {text}{RESET}")


def test_imports():
    """Test if all required libraries can be imported."""
    print_header("TEST 1: Checking Library Imports")
    
    required_libs = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib.pyplot',
        'seaborn': 'seaborn',
        'sklearn': 'sklearn',
        'tensorflow': 'tensorflow',
        'PIL': 'PIL',
        'cv2': 'cv2',
        'streamlit': 'streamlit',
        'joblib': 'joblib'
    }
    
    failed = []
    
    for name, import_path in required_libs.items():
        try:
            __import__(import_path.split('.')[0])
            print_success(f"{name} imported successfully")
        except ImportError as e:
            print_error(f"{name} import failed: {e}")
            failed.append(name)
    
    if failed:
        print_warning(f"\nFailed imports: {', '.join(failed)}")
        print_warning("Run: pip install -r requirements.txt")
        return False
    
    print_success("\nAll libraries imported successfully!")
    return True


def test_project_structure():
    """Test if project structure is correct."""
    print_header("TEST 2: Checking Project Structure")
    
    base_dir = Path(__file__).parent
    
    required_dirs = [
        'data',
        'data/raw',
        'data/processed',
        'data/images',
        'models',
        'notebooks',
        'src',
        'app'
    ]
    
    required_files = [
        'requirements.txt',
        'README.md',
        'QUICKSTART.md',
        'PROJECT_SUMMARY.md',
        '.gitignore',
        'download_data.sh',
        'notebooks/diamond_analysis.ipynb',
        'src/data_preprocessing.py',
        'src/model_training.py',
        'src/visualization.py',
        'src/cnn_models.py',
        'src/recommendation.py',
        'src/utils.py',
        'app/streamlit_app.py'
    ]
    
    all_good = True
    
    # Check directories
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        if full_path.exists():
            print_success(f"Directory exists: {dir_path}/")
        else:
            print_error(f"Directory missing: {dir_path}/")
            all_good = False
    
    # Check files
    for file_path in required_files:
        full_path = base_dir / file_path
        if full_path.exists():
            print_success(f"File exists: {file_path}")
        else:
            print_error(f"File missing: {file_path}")
            all_good = False
    
    if all_good:
        print_success("\nProject structure is complete!")
    else:
        print_warning("\nSome files or directories are missing")
    
    return all_good


def test_module_imports():
    """Test if custom modules can be imported."""
    print_header("TEST 3: Checking Custom Modules")
    
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    
    modules = [
        'data_preprocessing',
        'model_training',
        'visualization',
        'cnn_models',
        'recommendation',
        'utils'
    ]
    
    failed = []
    
    for module in modules:
        try:
            __import__(module)
            print_success(f"{module}.py imported successfully")
        except Exception as e:
            print_error(f"{module}.py import failed: {e}")
            failed.append(module)
    
    if failed:
        print_warning(f"\nFailed modules: {', '.join(failed)}")
        return False
    
    print_success("\nAll custom modules imported successfully!")
    return True


def test_data_availability():
    """Check if datasets are downloaded."""
    print_header("TEST 4: Checking Data Availability")
    
    base_dir = Path(__file__).parent
    raw_dir = base_dir / 'data' / 'raw'
    
    # Check for CSV files
    csv_files = list(raw_dir.glob('*.csv'))
    
    if csv_files:
        print_success(f"Found {len(csv_files)} CSV file(s) in data/raw/")
        for csv_file in csv_files:
            print(f"  - {csv_file.name}")
    else:
        print_warning("No CSV files found in data/raw/")
        print_warning("Download datasets with: bash download_data.sh")
        return False
    
    return True


def test_tensorflow_gpu():
    """Test TensorFlow GPU availability."""
    print_header("TEST 5: Checking TensorFlow GPU Support")
    
    try:
        import tensorflow as tf
        
        print(f"TensorFlow version: {tf.__version__}")
        
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print_success(f"GPU available: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"  - {gpu}")
        else:
            print_warning("No GPU detected - will use CPU")
            print("  (This is fine, but training will be slower)")
        
        return True
    
    except Exception as e:
        print_error(f"TensorFlow check failed: {e}")
        return False


def test_sample_processing():
    """Test basic data processing functionality."""
    print_header("TEST 6: Testing Sample Data Processing")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample data
        sample_data = {
            'carat': [0.5, 1.0, 1.5, 2.0],
            'cut': ['Good', 'Ideal', 'Premium', 'Very Good'],
            'color': ['E', 'F', 'G', 'D'],
            'clarity': ['VS1', 'VVS2', 'SI1', 'IF'],
            'depth': [61.5, 62.0, 60.5, 61.0],
            'table': [57.0, 58.0, 56.0, 57.5],
            'price': [2000, 5000, 8000, 15000],
            'x': [5.0, 6.5, 7.5, 8.5],
            'y': [5.0, 6.5, 7.5, 8.5],
            'z': [3.0, 4.0, 4.5, 5.0]
        }
        
        df = pd.DataFrame(sample_data)
        
        # Test basic operations
        assert df.shape == (4, 10), "DataFrame shape incorrect"
        assert df['price'].mean() == 7500, "Mean calculation incorrect"
        assert df['carat'].max() == 2.0, "Max calculation incorrect"
        
        # Test feature creation
        df['volume'] = df['x'] * df['y'] * df['z']
        assert 'volume' in df.columns, "Feature creation failed"
        
        print_success("Sample data processing successful")
        print(f"  - Created DataFrame with {df.shape[0]} rows")
        print(f"  - Calculated statistics (mean price: ${df['price'].mean():,.0f})")
        print(f"  - Created derived features (volume)")
        
        return True
    
    except Exception as e:
        print_error(f"Sample processing failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print(f"\n{BLUE}╔═══════════════════════════════════════════════════════╗")
    print(f"║       DiamondHood System Validation Test Suite       ║")
    print(f"╚═══════════════════════════════════════════════════════╝{RESET}\n")
    
    tests = [
        ("Library Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("Custom Modules", test_module_imports),
        ("Data Availability", test_data_availability),
        ("TensorFlow GPU", test_tensorflow_gpu),
        ("Sample Processing", test_sample_processing)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"{test_name:.<40} {status}")
    
    print(f"\n{BLUE}{'─'*60}{RESET}")
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print_success("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Download datasets: bash download_data.sh")
        print("  2. Run notebook: jupyter notebook notebooks/diamond_analysis.ipynb")
        print("  3. Launch app: streamlit run app/streamlit_app.py")
    else:
        print_warning(f"\n⚠ {total - passed} test(s) failed. Please fix issues before proceeding.")
        print("\nTroubleshooting:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Check Python version: python --version (need 3.9+)")
        print("  - Verify project structure is complete")
    
    print(f"\n{BLUE}{'─'*60}{RESET}\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

# DiamondHood - Quick Start Guide 🚀

This guide will get you up and running with DiamondHood in under 10 minutes.

---

## Prerequisites Checklist ✅

Before starting, ensure you have:
- [ ] Python 3.9 or higher installed
- [ ] pip or conda package manager
- [ ] Kaggle account (for dataset download)
- [ ] 5GB of free disk space
- [ ] Internet connection

---

## Installation Steps

### Step 1: Navigate to Project Directory

```bash
cd "/Users/mvaishak/Developer/Homeworks & Assignments/CSE258R Recommender Systems & Web Mining/Assignment 2/DiamondHood"
```

### Step 2: Create Virtual Environment

**Option A: Using Conda (Recommended)**
```bash
conda create -n diamondhood python=3.9 -y
conda activate diamondhood
```

**Option B: Using venv**
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- Core ML: pandas, numpy, scikit-learn
- Deep Learning: tensorflow, torch
- Visualization: matplotlib, seaborn, plotly
- Web App: streamlit
- Image Processing: opencv-python, pillow

**Expected time: 5-10 minutes**

### Step 4: Setup Kaggle API

1. **Get API Credentials**
   - Go to https://www.kaggle.com/settings
   - Click "Create New API Token"
   - Download `kaggle.json`

2. **Install Credentials**
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Step 5: Download Datasets

**Automatic Download (Recommended)**
```bash
bash download_data.sh
```

**Manual Download (If automatic fails)**
1. Tabular Data:
   - Visit: https://www.kaggle.com/datasets/colearninglounge/gemstone-price-prediction
   - Download and extract to `data/raw/`

2. Image Data:
   - Visit: https://www.kaggle.com/datasets/aayushpurswani/diamond-images-dataset
   - Download and extract to `data/images/`

**Expected time: 3-5 minutes**

---

## Verify Installation

Run this quick test to verify everything is working:

```bash
python -c "
import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf
import streamlit
print('✅ All core libraries imported successfully!')
print(f'TensorFlow version: {tf.__version__}')
print(f'GPU Available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')
"
```

Expected output:
```
✅ All core libraries imported successfully!
TensorFlow version: 2.13.x
GPU Available: True/False
```

---

## Usage Options

### Option 1: Run Complete Analysis (Jupyter Notebook)

**Start Jupyter:**
```bash
jupyter notebook
```

**Open:** `notebooks/diamond_analysis.ipynb`

This notebook contains:
- Section 1: Problem Definition
- Section 2: Data Preprocessing
- Section 3: Exploratory Data Analysis (EDA)
- Section 4: Baseline Models
- Section 5: Advanced Models (CNN)
- Section 6: Recommendation Engine
- Section 7: Model Evaluation
- Section 8: Related Work

**Expected time: Run all cells in ~20-30 minutes**

---

### Option 2: Run Individual Python Modules

**1. Data Preprocessing**
```bash
python src/data_preprocessing.py
```
Output: Cleaned datasets in `data/processed/`

**2. Train Regression Models**
```bash
python src/model_training.py
```
Output: Trained models in `models/`

**3. Generate Visualizations**
```bash
python src/visualization.py
```
Output: EDA plots and insights

**4. Train CNN (Image Classification)**
```bash
python src/cnn_models.py
```
Output: CNN model in `models/`

**5. Build Recommendations**
```bash
python src/recommendation.py
```
Output: Recommendation engine ready

---

### Option 3: Launch Web Application

```bash
streamlit run app/streamlit_app.py
```

The web app will open in your browser at `http://localhost:8501`

**Features:**
- 📸 Upload diamond image → Get price estimate
- 🔍 Search by features (carat, cut, color, clarity)
- 💰 Find best value within budget
- 💎 Visual recommendations with similar diamonds
- 📊 Interactive price prediction charts

---

## Quick Test Run

Run this minimal example to test the pipeline:

```python
# test_pipeline.py
from pathlib import Path
from src.data_preprocessing import DiamondDataProcessor

# Initialize
base_dir = Path.cwd()
data_dir = base_dir / 'data'

# Process data
processor = DiamondDataProcessor(data_dir)
train_df, val_df, test_df, full_df = processor.process_pipeline()

print(f"\n✅ SUCCESS!")
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")
print(f"\nDataset ready for modeling!")
```

Run:
```bash
python test_pipeline.py
```

---

## Troubleshooting

### Issue: Kaggle API not working
**Solution:**
```bash
# Check if kaggle.json exists
ls -la ~/.kaggle/

# Verify permissions
chmod 600 ~/.kaggle/kaggle.json

# Test kaggle CLI
kaggle datasets list
```

### Issue: TensorFlow not using GPU
**Solution:**
```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Install GPU support (if needed)
pip install tensorflow-metal  # For Apple Silicon Macs
```

### Issue: Import errors
**Solution:**
```bash
# Verify you're in the correct environment
which python

# Reinstall packages
pip install --force-reinstall -r requirements.txt
```

### Issue: Out of memory during training
**Solution:**
- Reduce batch size in CNN training
- Use smaller image size (e.g., 128x128 instead of 224x224)
- Close other applications

---

## Next Steps

After installation, you can:

1. **Explore the Data**
   - Open `notebooks/diamond_analysis.ipynb`
   - Run Section 3 (EDA) to see visualizations

2. **Train Your First Model**
   - Run baseline regression models
   - Compare Linear vs Tree-based approaches

3. **Build Recommendations**
   - Load trained models
   - Generate recommendations
   - Test on sample diamonds

4. **Deploy Web App**
   - Customize Streamlit interface
   - Add your own features
   - Share with users

---

## Expected Results

After running the complete pipeline, you should achieve:

### Regression Models (Price Prediction)
- **Best R² Score**: ~0.98
- **RMSE**: < $700
- **MAE**: < $450

### CNN Models (Image Classification)
- **Accuracy**: > 85%
- **F1 Score**: > 0.83

### Recommendation System
- **Top-5 Precision**: > 0.75
- **Hit Rate**: > 0.80

---

## Project Structure Summary

```
DiamondHood/
├── data/
│   ├── raw/              # Downloaded datasets
│   ├── processed/        # Cleaned data (CSV)
│   └── images/           # Diamond images
├── notebooks/
│   └── diamond_analysis.ipynb  # Main analysis
├── src/
│   ├── data_preprocessing.py   # Data cleaning
│   ├── model_training.py       # Regression models
│   ├── cnn_models.py          # Image classification
│   ├── recommendation.py       # Recommender system
│   └── visualization.py        # EDA plots
├── app/
│   └── streamlit_app.py       # Web interface
├── models/                     # Saved models (.pkl, .h5)
└── requirements.txt           # Dependencies
```

---

## Time Estimates

| Task | Duration |
|------|----------|
| Installation | 5-10 min |
| Dataset Download | 3-5 min |
| Data Preprocessing | 2-3 min |
| EDA | 5 min |
| Train Baseline Models | 5-10 min |
| Train CNN | 15-30 min |
| Build Recommendations | 5 min |
| **Total** | **40-70 min** |

---

## Support

If you encounter issues:

1. Check the main `README.md` for detailed documentation
2. Review error messages carefully
3. Verify all dependencies are installed
4. Ensure datasets are downloaded correctly
5. Check Python version (must be 3.9+)

---

## Quick Commands Reference

```bash
# Activate environment
conda activate diamondhood

# Run full notebook
jupyter notebook notebooks/diamond_analysis.ipynb

# Train all models
python src/data_preprocessing.py && python src/model_training.py

# Launch app
streamlit run app/streamlit_app.py

# Deactivate environment
conda deactivate
```

---

**Ready to start? Run this command:**

```bash
conda activate diamondhood && jupyter notebook notebooks/diamond_analysis.ipynb
```

---

**Built with ❤️ for CSE258R Assignment 2**

For questions or issues, refer to the main README.md or project documentation.

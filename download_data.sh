#!/bin/bash

# DiamondHood Dataset Download Script
# Downloads datasets from Kaggle for diamond price prediction and visual recommendation

echo "======================================"
echo "DiamondHood Data Download Script"
echo "======================================"

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null
then
    echo "Kaggle CLI not found. Installing..."
    pip install kaggle
fi

# # Check for Kaggle credentials
# if [ ! -f ~/.kaggle/kaggle.json ]; then
#     echo "ERROR: Kaggle credentials not found!"
#     echo "Please follow these steps:"
#     echo "1. Go to https://www.kaggle.com/settings"
#     echo "2. Click 'Create New API Token'"
#     echo "3. Add KAGGLE_API_TOKEN to env"
#     exit 1
# fi

# Create data directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/images

echo ""
echo "Downloading tabular dataset..."
kaggle datasets download -d colearninglounge/gemstone-price-prediction -p data/raw --unzip

echo ""
echo "Downloading image dataset..."
kaggle datasets download -d aayushpurswani/diamond-images-dataset -p data/images --unzip

echo ""
echo "======================================"
echo "Download complete!"
echo "======================================"
echo "Tabular data: data/raw/"
echo "Images: data/images/"

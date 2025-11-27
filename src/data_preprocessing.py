"""
Data Preprocessing Module for DiamondHood
Handles loading, cleaning, and preprocessing of diamond datasets
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings('ignore')


class DiamondDataProcessor:
    """
    Main class for processing diamond tabular and image data.
    """
    
    def __init__(self, data_dir):
        """
        Initialize the data processor.
        
        Args:
            data_dir: Path to the data directory
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.image_dir = self.data_dir / 'images'
        
        # Quality mappings
        self.cut_order = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
        self.color_order = {'D': 7, 'E': 6, 'F': 5, 'G': 4, 'H': 3, 'I': 2, 'J': 1}
        self.clarity_order = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 
                             'VVS2': 6, 'VVS1': 7, 'IF': 8}
        
        self.scaler = StandardScaler()
        
    def load_tabular_data(self):
        """
        Load the diamond tabular dataset from CSV.
        
        Returns:
            DataFrame with diamond features
        """
        possible_files = [
            self.raw_dir / 'cubic_zirconia.csv',
            self.raw_dir / 'diamonds.csv',
            self.raw_dir / 'gemstone.csv'
        ]
        
        for filepath in possible_files:
            if filepath.exists():
                print(f"Loading data from: {filepath}")
                df = pd.read_csv(filepath)
                print(f"Dataset shape: {df.shape}")
                return df
        
        raise FileNotFoundError(
            f"No data file found in {self.raw_dir}. "
            "Please download the dataset first."
        )
    
    def clean_data(self, df):
        """
        Clean the diamond dataset.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        print("Starting data cleaning...")
        print(f"Initial shape: {df.shape}")
        
        df_clean = df.copy()
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        print(f"After removing duplicates: {df_clean.shape}")
        
        # Remove missing values
        df_clean = df_clean.dropna()
        print(f"After removing missing values: {df_clean.shape}")
        
        # Remove impossible values (zero dimensions)
        dimension_cols = [col for col in ['x', 'y', 'z'] if col in df_clean.columns]
        for col in dimension_cols:
            df_clean = df_clean[df_clean[col] > 0]
        print(f"After removing zero dimensions: {df_clean.shape}")
        
        # Remove zero/negative carat
        if 'carat' in df_clean.columns:
            df_clean = df_clean[df_clean['carat'] > 0]
            print(f"After removing invalid carat: {df_clean.shape}")
        
        # Remove price outliers (1st and 99th percentile)
        if 'price' in df_clean.columns:
            Q1 = df_clean['price'].quantile(0.01)
            Q3 = df_clean['price'].quantile(0.99)
            df_clean = df_clean[(df_clean['price'] >= Q1) & (df_clean['price'] <= Q3)]
            print(f"After removing price outliers: {df_clean.shape}")
        
        print(f"Final cleaned shape: {df_clean.shape}")
        print(f"Rows removed: {len(df) - len(df_clean)}")
        
        return df_clean
    
    def create_features(self, df):
        """
        Create derived features from existing columns.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df_featured = df.copy()
        
        # Volume
        if all(col in df.columns for col in ['x', 'y', 'z']):
            df_featured['volume'] = df_featured['x'] * df_featured['y'] * df_featured['z']
            
            # Ratio features
            df_featured['ratio_xy'] = df_featured['x'] / df_featured['y']
            df_featured['ratio_xz'] = df_featured['x'] / df_featured['z']
            
            print("✓ Created volume and ratio features")
        
        # Price per carat
        if 'price' in df.columns and 'carat' in df.columns:
            df_featured['price_per_carat'] = df_featured['price'] / df_featured['carat']
            print("✓ Created price_per_carat feature")
        
        return df_featured
    
    def encode_categorical(self, df):
        """
        Encode categorical variables with ordinal mappings.
        
        Args:
            df: DataFrame with categorical columns
            
        Returns:
            DataFrame with encoded columns
        """
        df_encoded = df.copy()
        
        # Ordinal encoding for quality features
        if 'cut' in df.columns:
            df_encoded['cut_encoded'] = df_encoded['cut'].map(self.cut_order)
            df_encoded['cut_encoded'].fillna(df_encoded['cut_encoded'].median(), inplace=True)
        
        if 'color' in df.columns:
            df_encoded['color_encoded'] = df_encoded['color'].map(self.color_order)
            df_encoded['color_encoded'].fillna(df_encoded['color_encoded'].median(), inplace=True)
        
        if 'clarity' in df.columns:
            df_encoded['clarity_encoded'] = df_encoded['clarity'].map(self.clarity_order)
            df_encoded['clarity_encoded'].fillna(df_encoded['clarity_encoded'].median(), inplace=True)
        
        print("✓ Encoded categorical variables")
        
        return df_encoded
    
    def prepare_features_target(self, df, target_col='price'):
        """
        Prepare feature matrix and target variable.
        
        Args:
            df: DataFrame with all features
            target_col: Name of target column
            
        Returns:
            X (features), y (target)
        """
        # Select numeric features
        feature_cols = [col for col in df.columns if col not in 
                       [target_col, 'cut', 'color', 'clarity', 'diamond_id'] 
                       and df[col].dtype in ['int64', 'float64']]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy() if target_col in df.columns else None
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Features: {feature_cols}")
        
        return X, y
    
    def split_data(self, df, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: DataFrame to split
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed
            
        Returns:
            train_df, val_df, test_df
        """
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        
        # Second split: train vs val
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size, random_state=random_state
        )
        
        print("Data Splits:")
        print(f"  Training:   {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
        print(f"  Validation: {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
        print(f"  Test:       {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, train_df, val_df, test_df, full_df):
        """
        Save processed datasets to CSV files.
        
        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            full_df: Full cleaned dataset
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        train_df.to_csv(self.processed_dir / 'train_data.csv', index=False)
        val_df.to_csv(self.processed_dir / 'val_data.csv', index=False)
        test_df.to_csv(self.processed_dir / 'test_data.csv', index=False)
        full_df.to_csv(self.processed_dir / 'diamonds_clean.csv', index=False)
        
        print(f"✓ Saved processed data to: {self.processed_dir}")
    
    def process_pipeline(self):
        """
        Execute the complete data processing pipeline.
        
        Returns:
            train_df, val_df, test_df, full_df
        """
        print("="*60)
        print("DIAMOND DATA PROCESSING PIPELINE")
        print("="*60)
        
        # Load data
        df = self.load_tabular_data()
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Create features
        df_featured = self.create_features(df_clean)
        
        # Encode categorical
        df_encoded = self.encode_categorical(df_featured)
        
        # Add unique ID
        df_encoded['diamond_id'] = range(len(df_encoded))
        
        # Split data
        train_df, val_df, test_df = self.split_data(df_encoded)
        
        # Save processed data
        self.save_processed_data(train_df, val_df, test_df, df_encoded)
        
        print("="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        
        return train_df, val_df, test_df, df_encoded


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    
    processor = DiamondDataProcessor(data_dir)
    train_df, val_df, test_df, full_df = processor.process_pipeline()
    
    print("\nProcessing complete!")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

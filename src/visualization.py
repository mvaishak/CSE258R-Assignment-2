"""
Visualization and EDA Module for DiamondHood
Comprehensive exploratory data analysis functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class DiamondEDA:
    """
    Exploratory Data Analysis for diamond datasets.
    """
    
    def __init__(self, df):
        """
        Initialize EDA with diamond dataset.
        
        Args:
            df: DataFrame with diamond data
        """
        self.df = df.copy()
        self.numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
    def plot_price_distribution(self, figsize=(15, 5)):
        """
        Plot price distribution with histogram and box plot.
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Histogram
        axes[0].hist(self.df['price'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Price ($)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Price Distribution', fontsize=14, fontweight='bold')
        axes[0].axvline(self.df['price'].mean(), color='red', linestyle='--', label=f'Mean: ${self.df["price"].mean():,.0f}')
        axes[0].axvline(self.df['price'].median(), color='green', linestyle='--', label=f'Median: ${self.df["price"].median():,.0f}')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Log-scale histogram
        axes[1].hist(np.log10(self.df['price']), bins=50, color='coral', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('log10(Price)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Price Distribution (Log Scale)', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        # Box plot
        axes[2].boxplot(self.df['price'], vert=True)
        axes[2].set_ylabel('Price ($)', fontsize=12)
        axes[2].set_title('Price Box Plot', fontsize=14, fontweight='bold')
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_distributions(self, features=None, figsize=(15, 10)):
        """
        Plot distributions of numerical features.
        
        Args:
            features: List of features to plot (default: all numeric)
        """
        if features is None:
            features = [col for col in self.numeric_cols if col != 'price'][:9]
        
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(features):
            if feature in self.df.columns:
                axes[idx].hist(self.df[feature], bins=30, color='skyblue', alpha=0.7, edgecolor='black')
                axes[idx].set_xlabel(feature, fontsize=11)
                axes[idx].set_ylabel('Frequency', fontsize=11)
                axes[idx].set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
                axes[idx].axvline(self.df[feature].mean(), color='red', linestyle='--', linewidth=2)
                axes[idx].grid(alpha=0.3)
        
        # Hide extra subplots
        for idx in range(len(features), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self, figsize=(12, 10)):
        """
        Plot correlation heatmap for numerical features.
        """
        # Select only numeric columns
        numeric_df = self.df[self.numeric_cols]
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            fmt='.2f', 
            cmap='coolwarm', 
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        return fig, corr_matrix
    
    def plot_price_vs_features(self, features=['carat', 'depth', 'table'], figsize=(15, 5)):
        """
        Scatter plots of price vs key features.
        
        Args:
            features: List of features to plot against price
        """
        n_features = len(features)
        fig, axes = plt.subplots(1, n_features, figsize=figsize)
        
        if n_features == 1:
            axes = [axes]
        
        for idx, feature in enumerate(features):
            if feature in self.df.columns:
                axes[idx].scatter(self.df[feature], self.df['price'], alpha=0.3, s=10)
                axes[idx].set_xlabel(feature, fontsize=12)
                axes[idx].set_ylabel('Price ($)', fontsize=12)
                axes[idx].set_title(f'Price vs {feature}', fontsize=13, fontweight='bold')
                axes[idx].grid(alpha=0.3)
                
                # Add correlation coefficient
                corr = self.df[[feature, 'price']].corr().iloc[0, 1]
                axes[idx].text(
                    0.05, 0.95, f'r = {corr:.3f}',
                    transform=axes[idx].transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                )
        
        plt.tight_layout()
        return fig
    
    def plot_categorical_impact(self, categorical_cols=None, figsize=(15, 12)):
        """
        Violin plots showing price distribution by categorical variables.
        
        Args:
            categorical_cols: List of categorical columns to analyze
        """
        if categorical_cols is None:
            categorical_cols = self.categorical_cols
        
        n_cats = len(categorical_cols)
        fig, axes = plt.subplots(n_cats, 2, figsize=figsize)
        
        if n_cats == 1:
            axes = axes.reshape(1, -1)
        
        for idx, cat_col in enumerate(categorical_cols):
            if cat_col in self.df.columns:
                # Violin plot
                sns.violinplot(
                    data=self.df, x=cat_col, y='price', 
                    ax=axes[idx, 0], palette='Set2'
                )
                axes[idx, 0].set_title(f'Price by {cat_col}', fontsize=12, fontweight='bold')
                axes[idx, 0].set_xlabel(cat_col, fontsize=11)
                axes[idx, 0].set_ylabel('Price ($)', fontsize=11)
                axes[idx, 0].tick_params(axis='x', rotation=45)
                axes[idx, 0].grid(alpha=0.3)
                
                # Box plot
                sns.boxplot(
                    data=self.df, x=cat_col, y='price',
                    ax=axes[idx, 1], palette='Set3'
                )
                axes[idx, 1].set_title(f'Price by {cat_col} (Boxplot)', fontsize=12, fontweight='bold')
                axes[idx, 1].set_xlabel(cat_col, fontsize=11)
                axes[idx, 1].set_ylabel('Price ($)', fontsize=11)
                axes[idx, 1].tick_params(axis='x', rotation=45)
                axes[idx, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_carat_vs_price_by_quality(self, quality_col='cut', figsize=(12, 6)):
        """
        Scatter plot of carat vs price colored by quality variable.
        
        Args:
            quality_col: Categorical column to color by (cut, color, or clarity)
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if quality_col in self.df.columns:
            # Create scatter plot with color mapping
            unique_vals = self.df[quality_col].unique()
            colors = sns.color_palette('husl', len(unique_vals))
            
            for val, color in zip(unique_vals, colors):
                mask = self.df[quality_col] == val
                ax.scatter(
                    self.df.loc[mask, 'carat'],
                    self.df.loc[mask, 'price'],
                    label=val, alpha=0.5, s=20, color=color
                )
            
            ax.set_xlabel('Carat', fontsize=13)
            ax.set_ylabel('Price ($)', fontsize=13)
            ax.set_title(f'Price vs Carat by {quality_col}', fontsize=15, fontweight='bold')
            ax.legend(title=quality_col, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_summary_statistics(self):
        """
        Generate comprehensive summary statistics.
        
        Returns:
            Dictionary of summary stats
        """
        summary = {
            'shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict(),
            'numeric_summary': self.df[self.numeric_cols].describe().to_dict(),
            'categorical_summary': {
                col: self.df[col].value_counts().to_dict() 
                for col in self.categorical_cols
            }
        }
        
        return summary
    
    def plot_price_statistics_by_category(self, category='cut', figsize=(12, 6)):
        """
        Bar plot showing mean, median, and std of price by category.
        
        Args:
            category: Categorical column to group by
        """
        if category not in self.df.columns:
            print(f"Column '{category}' not found")
            return
        
        # Calculate statistics
        stats_df = self.df.groupby(category)['price'].agg([
            ('Mean', 'mean'),
            ('Median', 'median'),
            ('Std', 'std'),
            ('Count', 'count')
        ]).reset_index()
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Bar plot for mean and median
        x = np.arange(len(stats_df))
        width = 0.35
        
        axes[0].bar(x - width/2, stats_df['Mean'], width, label='Mean', alpha=0.8)
        axes[0].bar(x + width/2, stats_df['Median'], width, label='Median', alpha=0.8)
        axes[0].set_xlabel(category, fontsize=12)
        axes[0].set_ylabel('Price ($)', fontsize=12)
        axes[0].set_title(f'Average Price by {category}', fontsize=13, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(stats_df[category], rotation=45)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Sample count
        axes[1].bar(stats_df[category], stats_df['Count'], color='coral', alpha=0.7)
        axes[1].set_xlabel(category, fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title(f'Sample Count by {category}', fontsize=13, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        return fig, stats_df
    
    def detect_outliers(self, column='price', method='iqr'):
        """
        Detect outliers using IQR or Z-score method.
        
        Args:
            column: Column to check for outliers
            method: 'iqr' or 'zscore'
            
        Returns:
            DataFrame with outliers marked
        """
        if method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        else:  # zscore
            z_scores = np.abs(stats.zscore(self.df[column]))
            outliers = z_scores > 3
        
        n_outliers = outliers.sum()
        pct_outliers = (n_outliers / len(self.df)) * 100
        
        print(f"Outliers in '{column}' ({method} method):")
        print(f"  Count: {n_outliers}")
        print(f"  Percentage: {pct_outliers:.2f}%")
        
        return self.df[outliers]
    
    def generate_full_report(self, save_path=None):
        """
        Generate complete EDA report with all visualizations.
        
        Args:
            save_path: Path to save figures (optional)
        """
        print("="*60)
        print("COMPREHENSIVE EDA REPORT")
        print("="*60)
        
        # 1. Price Distribution
        print("\n1. Analyzing price distribution...")
        fig1 = self.plot_price_distribution()
        if save_path:
            fig1.savefig(f"{save_path}/price_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Feature Distributions
        print("\n2. Analyzing feature distributions...")
        fig2 = self.plot_feature_distributions()
        if save_path:
            fig2.savefig(f"{save_path}/feature_distributions.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Correlation Heatmap
        print("\n3. Generating correlation heatmap...")
        fig3, corr_matrix = self.plot_correlation_heatmap()
        if save_path:
            fig3.savefig(f"{save_path}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nTop correlations with price:")
        price_corr = corr_matrix['price'].sort_values(ascending=False)
        print(price_corr.head(10))
        
        # 4. Price vs Features
        print("\n4. Analyzing price relationships...")
        fig4 = self.plot_price_vs_features()
        if save_path:
            fig4.savefig(f"{save_path}/price_vs_features.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 5. Categorical Impact
        if self.categorical_cols:
            print("\n5. Analyzing categorical variables...")
            fig5 = self.plot_categorical_impact()
            if save_path:
                fig5.savefig(f"{save_path}/categorical_impact.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        print("\n" + "="*60)
        print("EDA REPORT COMPLETE")
        print("="*60)


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from pathlib import Path
    
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / 'data' / 'processed' / 'diamonds_clean.csv'
    
    if data_file.exists():
        df = pd.read_csv(data_file)
        eda = DiamondEDA(df)
        eda.generate_full_report()
    else:
        print(f"Data file not found: {data_file}")
        print("Please run data preprocessing first.")

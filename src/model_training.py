"""
Model Training Module for DiamondHood
Implements regression models for diamond price prediction
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import joblib
from pathlib import Path
import json


class DiamondPricePredictor:
    """
    Train and evaluate regression models for diamond price prediction.
    """
    
    def __init__(self, models_dir):
        """
        Initialize the price predictor.
        
        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.results = {}
        
    def get_baseline_models(self):
        """
        Get a dictionary of baseline regression models.
        
        Returns:
            Dictionary of model_name: model_instance
        """
        return {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        }
    
    def train_model(self, model, X_train, y_train, model_name):
        """
        Train a single model.
        
        Args:
            model: Sklearn model instance
            X_train: Training features
            y_train: Training target
            model_name: Name of the model
            
        Returns:
            Trained model
        """
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        print(f"✓ {model_name} trained successfully")
        
        return model
    
    def evaluate_model(self, model, X, y, dataset_name='Test'):
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            X: Features
            y: True target values
            dataset_name: Name of the dataset (for display)
            
        Returns:
            Dictionary of metrics
        """
        y_pred = model.predict(X)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'mape': np.mean(np.abs((y - y_pred) / y)) * 100
        }
        
        print(f"\n{dataset_name} Set Performance:")
        print(f"  RMSE:  ${metrics['rmse']:,.2f}")
        print(f"  MAE:   ${metrics['mae']:,.2f}")
        print(f"  R²:    {metrics['r2']:.4f}")
        print(f"  MAPE:  {metrics['mape']:.2f}%")
        
        return metrics, y_pred
    
    def train_all_models(self, X_train, y_train, X_val, y_val):
        """
        Train all baseline models.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Dictionary of trained models and results
        """
        print("="*60)
        print("TRAINING BASELINE MODELS")
        print("="*60)
        
        baseline_models = self.get_baseline_models()
        
        for model_name, model in baseline_models.items():
            # Train
            trained_model = self.train_model(model, X_train, y_train, model_name)
            
            # Evaluate on training set
            train_metrics, _ = self.evaluate_model(
                trained_model, X_train, y_train, 'Training'
            )
            
            # Evaluate on validation set
            val_metrics, _ = self.evaluate_model(
                trained_model, X_val, y_val, 'Validation'
            )
            
            # Store results
            self.models[model_name] = trained_model
            self.results[model_name] = {
                'train': train_metrics,
                'val': val_metrics
            }
            
            print("-"*60)
        
        return self.models, self.results
    
    def get_best_model(self, metric='r2'):
        """
        Get the best performing model based on validation metric.
        
        Args:
            metric: Metric to use for comparison ('r2', 'rmse', 'mae')
            
        Returns:
            Best model name and model instance
        """
        if metric in ['rmse', 'mae', 'mape']:
            # Lower is better
            best_name = min(
                self.results.keys(),
                key=lambda k: self.results[k]['val'][metric]
            )
        else:
            # Higher is better (r2)
            best_name = max(
                self.results.keys(),
                key=lambda k: self.results[k]['val'][metric]
            )
        
        print(f"\nBest Model (by {metric}): {best_name}")
        print(f"Validation {metric}: {self.results[best_name]['val'][metric]:.4f}")
        
        return best_name, self.models[best_name]
    
    def save_model(self, model, model_name, metadata=None):
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model
            model_name: Name for saving
            metadata: Optional metadata dictionary
        """
        # Save model
        model_path = self.models_dir / f"{model_name.replace(' ', '_').lower()}.pkl"
        joblib.dump(model, model_path)
        print(f"✓ Model saved: {model_path}")
        
        # Save metadata
        if metadata:
            metadata_path = self.models_dir / f"{model_name.replace(' ', '_').lower()}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"✓ Metadata saved: {metadata_path}")
    
    def save_all_models(self):
        """
        Save all trained models and results.
        """
        print("\nSaving models...")
        
        for model_name, model in self.models.items():
            metadata = {
                'model_name': model_name,
                'train_metrics': self.results[model_name]['train'],
                'val_metrics': self.results[model_name]['val']
            }
            self.save_model(model, model_name, metadata)
        
        # Save results summary
        results_path = self.models_dir / 'results_summary.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Results summary saved: {results_path}")
    
    def load_model(self, model_name):
        """
        Load a saved model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model
        """
        model_path = self.models_dir / f"{model_name.replace(' ', '_').lower()}.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = joblib.load(model_path)
        print(f"✓ Model loaded: {model_path}")
        
        return model
    
    def predict(self, model, X):
        """
        Make predictions with a trained model.
        
        Args:
            model: Trained model
            X: Features
            
        Returns:
            Predictions
        """
        return model.predict(X)
    
    def compare_models(self):
        """
        Create a comparison table of all models.
        
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Train R²': results['train']['r2'],
                'Val R²': results['val']['r2'],
                'Train RMSE': results['train']['rmse'],
                'Val RMSE': results['val']['rmse'],
                'Train MAE': results['train']['mae'],
                'Val MAE': results['val']['mae']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('Val R²', ascending=False)
        
        return df_comparison


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    import pandas as pd
    from data_preprocessing import DiamondDataProcessor
    
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    models_dir = base_dir / 'models'
    
    # Load processed data
    processor = DiamondDataProcessor(data_dir)
    train_df = pd.read_csv(processor.processed_dir / 'train_data.csv')
    val_df = pd.read_csv(processor.processed_dir / 'val_data.csv')
    
    # Prepare features
    X_train, y_train = processor.prepare_features_target(train_df)
    X_val, y_val = processor.prepare_features_target(val_df)
    
    # Train models
    predictor = DiamondPricePredictor(models_dir)
    models, results = predictor.train_all_models(X_train, y_train, X_val, y_val)
    
    # Get best model
    best_name, best_model = predictor.get_best_model()
    
    # Save models
    predictor.save_all_models()
    
    # Show comparison
    print("\nModel Comparison:")
    print(predictor.compare_models())

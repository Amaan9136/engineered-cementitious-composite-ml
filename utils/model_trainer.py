import numpy as np
import pickle

# Models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
    BaggingRegressor
)
from sklearn.svm import SVR
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.neighbors import KNeighborsRegressor

# Evaluation Metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# External Libraries
import xgboost as xgb
import lightgbm as lgb


class ModelTrainer:
    def __init__(self):
        self.models = {
            # Tree-based models
            'decision_tree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'extra_trees': ExtraTreesRegressor(n_estimators=100, random_state=42, max_depth=10),
            
            # Boosting models
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
            'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1),
            'ada_boost': AdaBoostRegressor(n_estimators=50, random_state=42),
            
            # Linear models
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0),
            'lasso_regression': Lasso(alpha=1.0),
            'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5),
            
            # Other models
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'knn_regression': KNeighborsRegressor(n_neighbors=5),
            'bagging': BaggingRegressor(n_estimators=10, random_state=42)
        }
        self.results = {}
    
    def train_model(self, model_name, X_train, y_train, X_test, y_test):
        """Train a single model and evaluate"""
        try:
            model = self.models[model_name]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test))
            }
            
            return model, metrics, y_pred_test
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            # Return dummy values if model fails
            dummy_pred = np.full_like(y_test, np.mean(y_train))
            metrics = {
                'train_r2': 0.0,
                'test_r2': 0.0,
                'train_mae': 999.0,
                'test_mae': 999.0,
                'train_rmse': 999.0,
                'test_rmse': 999.0
            }
            return None, metrics, dummy_pred
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        """Train all models and compare results"""
        self.results = {}
        
        for model_name in self.models.keys():
            print(f"Training {model_name}...")
            model, metrics, predictions = self.train_model(
                model_name, X_train, y_train, X_test, y_test
            )
            
            self.results[model_name] = {
                'metrics': metrics,
                'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
                'actual': y_test.tolist() if hasattr(y_test, 'tolist') else y_test
            }
            
            # Update the model in the dictionary if training was successful
            if model is not None:
                self.models[model_name] = model
        
        # Save results for comparison
        try:
            with open('data/test_results.pkl', 'wb') as f:
                pickle.dump(self.results, f)
        except Exception as e:
            print(f"Warning: Could not save results: {e}")
        
        return self.results
    
    def get_model_names(self):
        """Get list of all available model names"""
        return list(self.models.keys())
    
    def get_best_model(self):
        """Get the best performing model based on test R2 score"""
        if not self.results:
            return None, None
        
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['metrics']['test_r2'])
        best_model = self.models[best_model_name]
        
        return best_model_name, best_model
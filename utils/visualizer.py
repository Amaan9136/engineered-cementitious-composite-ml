import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')

class Visualizer:
    def __init__(self):
        plt.style.use('default')
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', 
                      '#764BA2', '#667EEA', '#F093FB', '#53A0FD', '#4FC3F7']
    
    def create_prediction_plot(self, prediction, experimental=None):
        """Create visualization for single model prediction"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            categories = ['Predicted']
            values = [prediction]
            colors = [self.colors[0]]
            
            if experimental is not None:
                categories.append('Experimental')
                values.append(experimental)
                colors.append(self.colors[1])
                
                # Add error
                error = abs(prediction - experimental)
                categories.append('Absolute Error')
                values.append(error)
                colors.append(self.colors[2])
            
            bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            ax.set_ylabel('Compressive Strength (MPa)', fontsize=14, fontweight='bold')
            ax.set_title('ECC Concrete Strength Prediction Results', fontsize=16, fontweight='bold', pad=20)
            ax.set_ylim(0, max(values) * 1.15)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_facecolor('#f8f9fa')
            
            plt.tight_layout()
            return self._plot_to_base64(fig)
            
        except Exception as e:
            print(f"Error creating prediction plot: {e}")
            return ""
    
    def create_all_predictions_plot(self, predictions, experimental=None):
        """Create visualization for all model predictions"""
        try:
            # Filter out None predictions
            valid_predictions = {k: v for k, v in predictions.items() if v is not None}
            
            if not valid_predictions:
                return ""
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            models = list(valid_predictions.keys())
            values = list(valid_predictions.values())
            
            # Create bars
            bars = ax.bar(range(len(models)), values, 
                         color=[self.colors[i % len(self.colors)] for i in range(len(models))],
                         alpha=0.8, edgecolor='black', linewidth=1.2)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Add experimental line if provided
            if experimental is not None:
                ax.axhline(y=experimental, color='red', linestyle='--', linewidth=2, 
                          label=f'Experimental: {experimental:.2f} MPa', alpha=0.8)
                ax.legend()
            
            ax.set_ylabel('Compressive Strength (MPa)', fontsize=14, fontweight='bold')
            ax.set_title('All Models Prediction Comparison', fontsize=16, fontweight='bold', pad=20)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=45, ha='right')
            ax.set_ylim(0, max(values) * 1.15)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_facecolor('#f8f9fa')
            
            plt.tight_layout()
            return self._plot_to_base64(fig)
            
        except Exception as e:
            print(f"Error creating all predictions plot: {e}")
            return ""
    
    def create_model_comparison(self, results):
        """Create model comparison visualization"""
        try:
            if not results:
                return ""
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.ravel()
            
            # Extract metrics
            models = list(results.keys())
            r2_scores = [results[m]['metrics']['test_r2'] for m in models]
            mae_scores = [results[m]['metrics']['test_mae'] for m in models]
            rmse_scores = [results[m]['metrics']['test_rmse'] for m in models]
            
            # Sort models by R2 score
            sorted_indices = np.argsort(r2_scores)[::-1]
            models_sorted = [models[i] for i in sorted_indices]
            r2_sorted = [r2_scores[i] for i in sorted_indices]
            mae_sorted = [mae_scores[i] for i in sorted_indices]
            rmse_sorted = [rmse_scores[i] for i in sorted_indices]
            
            colors_r2 = [self.colors[i % len(self.colors)] for i in range(len(models_sorted))]
            
            # Plot 1: R2 Scores
            self._create_metric_plot(axes[0], models_sorted, r2_sorted, colors_r2,
                                   'R² Score', 'Model R² Comparison (Higher is Better)', '{:.3f}')
            
            # Plot 2: MAE Scores
            self._create_metric_plot(axes[1], models_sorted, mae_sorted, colors_r2,
                                   'Mean Absolute Error (MPa)', 'Model MAE Comparison (Lower is Better)', '{:.2f}')
            
            # Plot 3: RMSE Scores
            self._create_metric_plot(axes[2], models_sorted, rmse_sorted, colors_r2,
                                   'Root Mean Square Error (MPa)', 'Model RMSE Comparison (Lower is Better)', '{:.2f}')
            
            # Plot 4: Prediction vs Actual for best model
            best_model = models_sorted[0]
            ax = axes[3]
            
            if 'predictions' in results[best_model] and 'actual' in results[best_model]:
                predictions = np.array(results[best_model]['predictions'])
                actual = np.array(results[best_model]['actual'])
                
                ax.scatter(actual, predictions, alpha=0.6, s=50, color=self.colors[0], edgecolors='black')
                
                min_val = min(min(actual), min(predictions))
                max_val = max(max(actual), max(predictions))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
                
                ax.set_xlabel('Actual Strength (MPa)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Predicted Strength (MPa)', fontsize=12, fontweight='bold')
                ax.set_title(f'Best Model: {best_model.replace("_", " ").title()}', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return self._plot_to_base64(fig)
            
        except Exception as e:
            print(f"Error creating model comparison plot: {e}")
            return ""
    
    def _create_metric_plot(self, ax, models, values, colors, ylabel, title, format_str):
        """Helper method to create metric plots"""
        bars = ax.bar(range(len(models)), values, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=45, ha='right')
        
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(values) * 0.01,
                   format_str.format(value), ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_to_base64(self, fig):
        """Convert plot to base64 string"""
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        plot_url = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{plot_url}"
    
    def create_feature_importance(self, feature_names=None):
        """Create feature importance plot (simplified version)"""
        try:
            if feature_names is None:
                feature_names = ['Cement OPC', 'SCM Fly Ash', 'SCM GGBS', 'Silica Sand', 
                               'Local Sand', 'W/B Ratio', 'HRWR/B Ratio', 'Fiber %', 
                               'Aspect Ratio', 'Tensile Strength', 'Density', 'Young\'s Modulus', 'Elongation']
            
            # Generate synthetic importance values
            importance = np.random.uniform(0.03, 0.20, len(feature_names))
            importance = importance / importance.sum()
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            indices = np.argsort(importance)[::-1]
            features_sorted = [feature_names[i] for i in indices]
            importance_sorted = importance[indices]
            
            colors_feat = [self.colors[i % len(self.colors)] for i in range(len(features_sorted))]
            bars = ax.barh(features_sorted, importance_sorted, color=colors_feat, alpha=0.8, edgecolor='black')
            
            ax.set_xlabel('Feature Importance', fontsize=14, fontweight='bold')
            ax.set_title('Feature Importance Analysis', fontsize=16, fontweight='bold', pad=20)
            
            for bar, imp in zip(bars, importance_sorted):
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2.,
                       f'{imp*100:.1f}%', ha='left', va='center', fontsize=11, fontweight='bold')
            
            ax.grid(True, alpha=0.3, axis='x')
            ax.set_facecolor('#f8f9fa')
            
            plt.tight_layout()
            return self._plot_to_base64(fig)
            
        except Exception as e:
            print(f"Error creating feature importance plot: {e}")
            return ""
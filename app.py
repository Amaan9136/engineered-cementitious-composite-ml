from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from utils.data_processor import DataProcessor
from utils.model_trainer import ModelTrainer
from utils.visualizer import Visualizer

app = Flask(__name__)

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('static/plots', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Initialize components
data_processor = DataProcessor()
model_trainer = ModelTrainer()
visualizer = Visualizer()

# Available datasets
DATASETS = {
    'sample_data.csv': 'Dataset 1 - Standard ECC Mix',
    'SCM-based-concrete-formated.csv': 'Dataset 2 - SCM-concrete-global'
}

# Load trained models on startup
models = {}
current_dataset = None
trained_models = set()  # Track which models are actually trained

def load_models():
    """Load all trained models"""
    global models, trained_models
    models = {}
    trained_models = set()
    
    for model_name in model_trainer.get_model_names():
        model_path = f'models/{model_name}_model.pkl'
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    models[model_name] = pickle.load(f)
                    trained_models.add(model_name)
                    print(f"Loaded model: {model_name}")
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")

@app.route('/')
def index():
    available_models = list(trained_models) if trained_models else []
    return render_template('index.html', 
                         datasets=DATASETS, 
                         models=available_models,
                         trained_models=list(trained_models))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_type = request.form.get('model_type')
        
        # Check if model is trained and available
        if not model_type or model_type not in trained_models:
            return jsonify({
                'error': f'Model "{model_type}" is not trained or available. Please train models first.',
                'available_models': list(trained_models)
            }), 400
        
        # Check if scaler exists
        scaler_path = 'models/scaler.pkl'
        if not os.path.exists(scaler_path):
            return jsonify({'error': 'No scaler found. Please train models first.'}), 400
        
        # Get input features
        features = {
            'cement_opc': float(request.form.get('cement', 0)),
            'scm_flyash': float(request.form.get('fly_ash', 0)),
            'scm_ggbs': float(request.form.get('ggbs', 0)),
            'silica_sand': float(request.form.get('silica_sand', 0)),
            'locally_avail_sand': float(request.form.get('sand', 0)),
            'w_b': float(request.form.get('water_binder', 0)),
            'hrwr_b': float(request.form.get('hrwr_binder', 0)),
            'perc_of_fibre': float(request.form.get('fiber_volume', 0)),
            'aspect_ratio': float(request.form.get('aspect_ratio', 0)),
            'tensile_strength': float(request.form.get('tensile_strength', 0)),
            'density': float(request.form.get('density', 0)),
            'youngs_modulus': float(request.form.get('youngs_modulus', 0)),
            'elongation': float(request.form.get('elongation', 0))
        }
        
        # Create and scale input
        input_df = pd.DataFrame([features])
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        input_scaled = scaler.transform(input_df)
        
        # Make prediction with the trained model
        prediction = models[model_type].predict(input_scaled)[0]
        
        # Get experimental value if provided
        experimental = request.form.get('experimental')
        experimental = float(experimental) if experimental else None
        
        # Generate visualization
        plot_url = visualizer.create_prediction_plot(prediction, experimental)
        
        return render_template('results.html',
                             prediction=round(prediction, 2),
                             experimental=experimental,
                             model_type=model_type.replace('_', ' ').title(),
                             features=features,
                             plot_url=plot_url)
    
    except ValueError as e:
        return jsonify({'error': f'Invalid input values: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 400

@app.route('/predict_all', methods=['POST'])
def predict_all():
    """Predict using all trained models"""
    try:
        if not trained_models:
            return jsonify({'error': 'No trained models available. Please train models first.'}), 400
        
        # Check if scaler exists
        scaler_path = 'models/scaler.pkl'
        if not os.path.exists(scaler_path):
            return jsonify({'error': 'No scaler found. Please train models first.'}), 400
        
        # Get input features
        features = {
            'cement_opc': float(request.form.get('cement', 0)),
            'scm_flyash': float(request.form.get('fly_ash', 0)),
            'scm_ggbs': float(request.form.get('ggbs', 0)),
            'silica_sand': float(request.form.get('silica_sand', 0)),
            'locally_avail_sand': float(request.form.get('sand', 0)),
            'w_b': float(request.form.get('water_binder', 0)),
            'hrwr_b': float(request.form.get('hrwr_binder', 0)),
            'perc_of_fibre': float(request.form.get('fiber_volume', 0)),
            'aspect_ratio': float(request.form.get('aspect_ratio', 0)),
            'tensile_strength': float(request.form.get('tensile_strength', 0)),
            'density': float(request.form.get('density', 0)),
            'youngs_modulus': float(request.form.get('youngs_modulus', 0)),
            'elongation': float(request.form.get('elongation', 0))
        }
        
        # Create and scale input
        input_df = pd.DataFrame([features])
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        input_scaled = scaler.transform(input_df)
        
        # Get predictions from all trained models
        predictions = {}
        for model_name in trained_models:
            try:
                pred = models[model_name].predict(input_scaled)[0]
                predictions[model_name] = round(pred, 2)
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
                predictions[model_name] = None
        
        # Get experimental value if provided
        experimental = request.form.get('experimental')
        experimental = float(experimental) if experimental else None
        
        # Generate comparison visualization
        plot_url = visualizer.create_all_predictions_plot(predictions, experimental)
        
        return render_template('all_predictions.html',
                             predictions=predictions,
                             experimental=experimental,
                             features=features,
                             plot_url=plot_url)
    
    except ValueError as e:
        return jsonify({'error': f'Invalid input values: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 400

@app.route('/train_models', methods=['POST'])
def train_models():
    try:
        dataset_name = request.form.get('dataset')
        
        if dataset_name not in DATASETS:
            return jsonify({'error': 'Invalid dataset selected'}), 400
        
        dataset_path = f'data/{dataset_name}'
        if not os.path.exists(dataset_path):
            return jsonify({'error': f'Dataset {dataset_name} not found.'}), 400
        
        # Load and process data
        df = pd.read_csv(dataset_path)
        is_valid, message = data_processor.validate_data(df)
        if not is_valid:
            return jsonify({'error': f'Data validation failed: {message}'}), 400
        
        X_train, X_test, y_train, y_test, scaler = data_processor.process_data(df)
        
        # Train all models
        results = model_trainer.train_all_models(X_train, X_test, y_train, y_test)
        
        # Save successful models and scaler
        global models, current_dataset, trained_models
        models = {}
        trained_models = set()
        
        for model_name, model in model_trainer.models.items():
            if model is not None:
                try:
                    with open(f'models/{model_name}_model.pkl', 'wb') as f:
                        pickle.dump(model, f)
                    models[model_name] = model
                    trained_models.add(model_name)
                except Exception as e:
                    print(f"Error saving model {model_name}: {e}")
        
        # Save scaler
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        current_dataset = dataset_name
        
        # Generate comparison plot
        comparison_plot = visualizer.create_model_comparison(results)
        best_model_name, _ = model_trainer.get_best_model()
        
        return jsonify({
            'success': True,
            'results': {k: {'metrics': v['metrics']} for k, v in results.items()},
            'comparison_plot': comparison_plot,
            'dataset': DATASETS[dataset_name],
            'best_model': best_model_name.replace('_', ' ').title() if best_model_name else 'Unknown',
            'total_models': len(trained_models),
            'trained_models': list(trained_models)
        })
    
    except Exception as e:
        return jsonify({'error': f'Training error: {str(e)}'}), 400

@app.route('/get_model_list')
def get_model_list():
    """API endpoint to get list of available trained models"""
    return jsonify({
        'trained_models': list(trained_models),
        'total_trained': len(trained_models)
    })

# Load models on startup
load_models()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
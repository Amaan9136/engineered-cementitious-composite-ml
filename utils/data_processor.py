import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        # Updated feature columns to match the CSV headers
        self.feature_columns = [
            'cement_opc', 'scm_flyash', 'scm_ggbs', 'silica_sand', 'locally_avail_sand',
            'w_b', 'hrwr_b', 'perc_of_fibre', 'aspect_ratio', 'tensile_strength',
            'density', 'youngs_modulus', 'elongation'
        ]
        self.target_column = 'compressive_stregth'  # Note: keeping original spelling from CSV
    
    def process_data(self, df):
        """Process raw data for training"""
        # Handle missing values
        df = df.fillna(0)
        
        # Extract features and target
        try:
            # Check if all feature columns exist
            missing_cols = [col for col in self.feature_columns if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns {missing_cols}. Using available columns.")
                available_features = [col for col in self.feature_columns if col in df.columns]
                X = df[available_features]
            else:
                X = df[self.feature_columns]
            
            # Handle target column
            if self.target_column in df.columns:
                y = df[self.target_column]
            else:
                # If target column not found, use last column
                y = df.iloc[:, -1]
                print(f"Target column '{self.target_column}' not found. Using last column: {df.columns[-1]}")
            
        except Exception as e:
            print(f"Error in column selection: {e}")
            # Fallback: use all columns except last as features
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
        
        # Remove any non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Ensure we have valid data
        if X.empty or len(y) == 0:
            raise ValueError("No valid data found for training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, self.scaler
    
    def prepare_input(self, input_dict):
        """Prepare single input for prediction"""
        # Create DataFrame with the correct column order
        input_df = pd.DataFrame([input_dict])
        
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Select only the feature columns in correct order
        input_df = input_df[self.feature_columns]
        
        return self.scaler.transform(input_df)
    
    def get_feature_names(self):
        """Get the list of feature names"""
        return self.feature_columns
    
    def validate_data(self, df):
        """Validate input data"""
        required_cols = self.feature_columns + [self.target_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            return False, f"Missing columns: {missing_cols}"
        
        # Check for non-numeric data
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            return False, f"Non-numeric columns found: {non_numeric}"
        
        return True, "Data validation passed"
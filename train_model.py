"""
Model Training and Evaluation Pipeline for Bengaluru House Price Prediction.
Phases 2-5 of the machine learning pipeline.
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import shap

# Set styling for plots
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def load_data(url: str) -> pd.DataFrame:
    """Loads dataset from URL."""
    print(f"Loading raw dataset from {url}...")
    try:
        return pd.read_csv(url)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def parse_total_sqft(x):
    """Parses range or text-based total_sqft into a numerical value."""
    if isinstance(x, str):
        tokens = x.split('-')
        if len(tokens) == 2:
            try:
                return (float(tokens[0].strip()) + float(tokens[1].strip())) / 2
            except ValueError:
                return np.nan
        try:
            return float(x.strip())
        except ValueError:
            # Drop or return nan for complex strings (perch, sq yards, etc.)
            return np.nan
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Performs domain-specific data cleaning for Indian real estate data."""
    print("Starting data cleaning...")
    df_clean = df.copy()
    
    # 1. Parse size column to BHK
    df_clean['bhk'] = df_clean['size'].apply(
        lambda x: int(x.split()[0]) if isinstance(x, str) and len(x.split()) > 0 else np.nan
    )
    
    # 2. Parse total_sqft to numeric
    df_clean['total_sqft_cleaned'] = df_clean['total_sqft'].apply(parse_total_sqft)
    
    # 3. Drop rows where key target or features are missing
    df_clean = df_clean.dropna(subset=['price', 'total_sqft_cleaned', 'bhk'])
    
    # 4. Clean locations (strip whitespaces)
    df_clean['location'] = df_clean['location'].apply(lambda x: x.strip() if isinstance(x, str) else 'other')
    
    # 5. Fill missing values for bath and balcony with medians
    df_clean['bath'] = df_clean['bath'].fillna(df_clean['bath'].median())
    df_clean['balcony'] = df_clean['balcony'].fillna(df_clean['balcony'].median())
    
    # 6. Outlier Removal based on business logic
    print(f"Shape before business-logic outlier removal: {df_clean.shape}")
    
    # Standard BHK size logic: A BHK cannot be less than 300 sqft in ordinary circumstances
    df_clean = df_clean[df_clean['total_sqft_cleaned'] / df_clean['bhk'] >= 300]
    
    # Price per square foot outliers (remove houses with unrealistic price per sqft)
    # price is in Lakhs, so price * 100,000 gives price in INR
    df_clean['price_per_sqft'] = (df_clean['price'] * 100000) / df_clean['total_sqft_cleaned']
    
    # Global outlier filtering (rather than per-location group mean)
    # Remove top 1% prices and top 1% square footage, and extreme price_per_sqft
    q_price = df_clean['price'].quantile(0.99)
    q_sqft = df_clean['total_sqft_cleaned'].quantile(0.99)
    df_clean = df_clean[(df_clean['price'] < q_price) & (df_clean['total_sqft_cleaned'] < q_sqft)]
    df_clean = df_clean[(df_clean['price_per_sqft'] >= 1000) & (df_clean['price_per_sqft'] <= 50000)]
    print(f"Shape after global price and sqft outlier removal: {df_clean.shape}")
    
    # Bathroom outliers: It is abnormal to have bathrooms > BHK + 2
    df_clean = df_clean[df_clean['bath'] <= df_clean['bhk'] + 2]
    
    print(f"Shape after bathroom outlier removal: {df_clean.shape}")
    return df_clean

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates engineered features for Bengaluru housing."""
    print("Engineering features...")
    df_feat = df.copy()
    
    # Feature 1: Ratio of bathrooms to bedrooms
    df_feat['bath_to_bhk_ratio'] = df_feat['bath'] / df_feat['bhk']
    
    # Feature 2: Sqft per bedroom (average size of BHK)
    df_feat['sqft_per_bhk'] = df_feat['total_sqft_cleaned'] / df_feat['bhk']
    
    # Clean up locations: Keep locations with count >= 10, rest are 'other'
    location_counts = df_feat['location'].value_counts()
    less_than_10_locations = location_counts[location_counts < 10]
    df_feat['location'] = df_feat['location'].apply(lambda x: 'other' if x in less_than_10_locations else x)
    
    # Keep only relevant columns for training
    features_to_keep = [
        'area_type', 'location', 'total_sqft_cleaned', 'bath', 
        'balcony', 'bhk', 'bath_to_bhk_ratio', 'sqft_per_bhk', 'price'
    ]
    df_feat = df_feat[features_to_keep]
    
    # Rename columns for cleanliness
    df_feat = df_feat.rename(columns={'total_sqft_cleaned': 'total_sqft'})
    
    return df_feat

def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    """Trains a dictionary of models and reports metrics."""
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        
        # Predictions
        preds = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        results[name] = {
            'model_object': model,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'time': elapsed_time
        }
        
        print(f"{name} Results: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}, Time={elapsed_time:.3f}s")
        
    return results

def main():
    # Setup paths
    dataset_url = "https://raw.githubusercontent.com/bapujik/dataSets/refs/heads/main/Bengaluru_House_Data.csv"
    models_dir = "models"
    plots_dir = "plots"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Load data
    raw_df = load_data(dataset_url)
    
    # 2. Preprocess & Clean
    cleaned_df = clean_data(raw_df)
    
    # 3. Feature Engineering
    engineered_df = engineer_features(cleaned_df)
    
    # 4. Prepare data for model split
    X = engineered_df.drop('price', axis=1)
    y = engineered_df['price']
    
    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocessor Pipelines: OneHotEncoder & StandardScaler
    print("Fitting preprocessors (Scaler & Encoder)...")
    
    num_cols = ['total_sqft', 'bath', 'balcony', 'bhk', 'bath_to_bhk_ratio', 'sqft_per_bhk']
    cat_cols = ['area_type', 'location']
    
    # Numerical Scaling
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[num_cols])
    X_test_num = scaler.transform(X_test[num_cols])
    
    # Categorical Encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_cat = encoder.fit_transform(X_train[cat_cols])
    X_test_cat = encoder.transform(X_test[cat_cols])
    
    # Concatenate numerical and categorical arrays
    X_train_processed = np.hstack((X_train_num, X_train_cat))
    X_test_processed = np.hstack((X_test_num, X_test_cat))
    
    # Generate feature names for evaluation / importance plotting
    cat_feature_names = encoder.get_feature_names_out(cat_cols).tolist()
    feature_names = num_cols + cat_feature_names
    
    # Save preprocessing states and category lists for Streamlit
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    joblib.dump(encoder, os.path.join(models_dir, "encoder.joblib"))
    
    # Save category options for Streamlit selections
    categories = {
        'locations': sorted(list(X_train['location'].unique())),
        'area_types': sorted(list(X_train['area_type'].unique()))
    }
    with open(os.path.join(models_dir, "categories.json"), "w") as f:
        json.dump(categories, f, indent=4)
    print("Saved scaler, encoder, and category choices.")
    
    # 5. Model Training & Evaluation
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    }
    
    eval_results = train_and_evaluate(models, X_train_processed, X_test_processed, y_train, y_test)
    
    # Save comparisons to print summary table
    summary_data = []
    best_model_name = None
    best_r2 = -1.0
    
    for name, res in eval_results.items():
        summary_data.append({
            'Model': name,
            'RMSE (Lakhs)': f"{res['rmse']:.3f}",
            'MAE (Lakhs)': f"{res['mae']:.3f}",
            'R² Score': f"{res['r2']:.3f}",
            'Training Time (s)': f"{res['time']:.3f}"
        })
        if res['r2'] > best_r2:
            best_r2 = res['r2']
            best_model_name = name
            
    summary_df = pd.DataFrame(summary_data)
    print("\n--- MODEL PERFORMANCE COMPARISON SUMMARY ---")
    print(summary_df.to_string(index=False))
    
    print(f"\nBest performing model based on R² score: {best_model_name}")
    best_model = eval_results[best_model_name]['model_object']
    
    # 6. Hyperparameter Tuning on Best Model
    print(f"\nStarting hyperparameter tuning on {best_model_name}...")
    tuned_model = None
    
    if best_model_name == 'XGBoost':
        # Fast grid for XGBoost to ensure reliable running time
        param_grid = {
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'n_estimators': [100, 150]
        }
        grid_search = GridSearchCV(
            estimator=xgb.XGBRegressor(random_state=42, n_jobs=-1),
            param_grid=param_grid,
            cv=3,
            scoring='r2',
            verbose=1,
            n_jobs=-1
        )
        start_tune = time.time()
        grid_search.fit(X_train_processed, y_train)
        tune_time = time.time() - start_tune
        
        print(f"Hyperparameter tuning completed in {tune_time:.2f}s")
        print(f"Best Parameters Found: {grid_search.best_params_}")
        tuned_model = grid_search.best_estimator_
        
    elif best_model_name == 'Random Forest':
        # Fast grid for Random Forest
        param_grid = {
            'n_estimators': [100, 150],
            'max_depth': [10, 15, None]
        }
        grid_search = GridSearchCV(
            estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
            param_grid=param_grid,
            cv=3,
            scoring='r2',
            verbose=1,
            n_jobs=-1
        )
        start_tune = time.time()
        grid_search.fit(X_train_processed, y_train)
        tune_time = time.time() - start_tune
        
        print(f"Hyperparameter tuning completed in {tune_time:.2f}s")
        print(f"Best Parameters Found: {grid_search.best_params_}")
        tuned_model = grid_search.best_estimator_
    else:
        # Linear Regression has no heavy hyperparameters, we keep it as is
        print("Tuning skipped as Linear Regression is the best model (unlikely) or not tree-based.")
        tuned_model = best_model
        
    # Evaluate Tuned Model
    if tuned_model is not None:
        tuned_preds = tuned_model.predict(X_test_processed)
        tuned_rmse = np.sqrt(mean_squared_error(y_test, tuned_preds))
        tuned_mae = mean_absolute_error(y_test, tuned_preds)
        tuned_r2 = r2_score(y_test, tuned_preds)
        
        print("\n--- TUNED MODEL PERFORMANCE ---")
        print(f"Tuned RMSE: {tuned_rmse:.3f} Lakhs")
        print(f"Tuned MAE: {tuned_mae:.3f} Lakhs")
        print(f"Tuned R² Score: {tuned_r2:.3f}")
        
        # Save tuned model as best model
        joblib.dump(tuned_model, os.path.join(models_dir, "best_model.joblib"))
        final_model = tuned_model
    else:
        joblib.dump(best_model, os.path.join(models_dir, "best_model.joblib"))
        final_model = best_model
        
    # 7. Feature Importance & SHAP Visualizations
    print("\nAnalyzing feature importances...")
    if hasattr(final_model, 'feature_importances_'):
        importances = final_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Top 10 Features
        top_n = 10
        top_indices = indices[:top_n]
        top_importances = importances[top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_importances, y=top_names, palette="viridis")
        plt.title(f"Top 10 Feature Importances ({best_model_name})", fontsize=14)
        plt.xlabel("Importance Score", fontsize=12)
        plt.tight_layout()
        importance_path = os.path.join(plots_dir, "feature_importance.png")
        plt.savefig(importance_path, dpi=150)
        plt.close()
        print(f"Saved: {importance_path}")
        
        # SHAP Analysis
        print("Generating SHAP summary plot (on sample of 200 test rows for speed)...")
        try:
            # Tree explainer is optimized for RF / XGBoost
            explainer = shap.TreeExplainer(final_model)
            # Sample for speed
            sample_size = min(200, X_test_processed.shape[0])
            X_sample = X_test_processed[:sample_size]
            shap_values = explainer.shap_values(X_sample)
            
            plt.figure(figsize=(10, 6))
            # Specify feature names for SHAP
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
            plt.title("SHAP Summary Plot", fontsize=14)
            plt.tight_layout()
            shap_path = os.path.join(plots_dir, "shap_summary.png")
            plt.savefig(shap_path, dpi=150)
            plt.close()
            print(f"Saved: {shap_path}")
        except Exception as e:
            print(f"Warning: SHAP visualization failed or was skipped due to: {e}")
    else:
        print("Feature importance plotting skipped (best model does not support feature_importances_).")

if __name__ == "__main__":
    main()

"""
Indian Real Estate Synthetic Dataset Generator & Model Training Pipeline.
Generates realistic Indian housing data and fits ML models for valuation.
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

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def generate_synthetic_indian_data(cities_csv_path: str, num_samples: int = 8000) -> pd.DataFrame:
    """Generates synthetic Indian real estate data with realistic pricing coefficients."""
    print(f"Loading city configurations from {cities_csv_path}...")
    cities_df = pd.read_csv(cities_csv_path)
    
    np.random.seed(42)
    
    # Pre-allocate list
    data = []
    
    prop_types = ['Flat/Apartment', 'Independent House/Villa', 'Builder Floor', 'Penthouse', 'Studio (1RK)']
    facings = ['North', 'North-East', 'East', 'South-East', 'South', 'South-West', 'West', 'North-West']
    vastu_options = ['Yes', 'Partially', 'No']
    age_categories = ['New Launch', 'Ready to Move', 'Resale', 'Old', 'Heritage']
    
    # Pull city records into dict for fast lookup
    city_configs = cities_df.set_index('City').to_dict(orient='index')
    city_names = list(city_configs.keys())
    
    for i in range(num_samples):
        # 1. Select City and get properties
        city = np.random.choice(city_names)
        config = city_configs[city]
        base_rate = config['Base_Price_SqFt']
        
        # City multipliers (Tier weights)
        city_multipliers = {
            'Mumbai': 2.5,
            'Delhi NCR': 1.8,
            'Bangalore': 1.6,
            'Hyderabad': 1.4,
            'Pune': 1.15,
            'Chennai': 1.1,
            'Ahmedabad': 1.0,
            'Jaipur': 0.8,
            'Lucknow': 0.75,
            'Indore': 0.7
        }
        city_mult = city_multipliers.get(city, 1.0)
        
        # 2. Select Property Type and size
        prop_type = np.random.choice(prop_types, p=[0.6, 0.15, 0.15, 0.05, 0.05])
        
        # Area depending on type (Super area in sqft)
        if prop_type == 'Studio (1RK)':
            area = np.random.uniform(250, 450)
            bhk = 1
        elif prop_type == 'Penthouse':
            area = np.random.uniform(2500, 6000)
            bhk = np.random.choice([4, 5, 6])
        elif prop_type == 'Independent House/Villa':
            area = np.random.uniform(1500, 5000)
            bhk = np.random.choice([3, 4, 5])
        else: # Flat or Builder Floor
            area = np.random.uniform(500, 2500)
            if area < 800:
                bhk = np.random.choice([1, 2], p=[0.6, 0.4])
            elif area < 1500:
                bhk = np.random.choice([2, 3], p=[0.7, 0.3])
            else:
                bhk = np.random.choice([3, 4], p=[0.8, 0.2])
                
        # 3. Bathrooms & Balconies
        bath = max(1, bhk + np.random.choice([-1, 0, 1], p=[0.1, 0.6, 0.3]))
        # Independent Villa or Penthouse might have more baths
        if prop_type in ['Independent House/Villa', 'Penthouse']:
            bath = bhk + np.random.choice([0, 1, 2])
            
        balcony = np.random.choice([0, 1, 2, 3])
        
        # 4. Building Height & Floor Number
        if prop_type == 'Independent House/Villa':
            floor = 1
            total_floors = np.random.choice([1, 2, 3])
        elif prop_type == 'Builder Floor':
            total_floors = np.random.choice([3, 4, 5])
            floor = np.random.randint(1, total_floors + 1)
        else:
            total_floors = np.random.randint(5, 41)
            floor = np.random.randint(1, total_floors + 1)
            
        # 5. Locality Features (km)
        dist_metro = np.random.exponential(2.5) + 0.1
        # No metro in Indore
        if city == 'Indore':
            dist_metro = 99.0
            
        dist_it = np.random.exponential(4.0) + 0.1
        dist_school = np.random.uniform(0.1, 5.0)
        dist_hospital = np.random.uniform(0.1, 5.0)
        
        # 6. Age and status
        age_cat = np.random.choice(age_categories, p=[0.15, 0.35, 0.35, 0.13, 0.02])
        if age_cat == 'New Launch':
            age_years = 0
        elif age_cat == 'Ready to Move':
            age_years = np.random.randint(0, 4)
        elif age_cat == 'Resale':
            age_years = np.random.randint(4, 11)
        elif age_cat == 'Old':
            age_years = np.random.randint(11, 31)
        else: # Heritage
            age_years = np.random.randint(50, 90)
            
        # 7. Vastu Compliant & Facing
        facing = np.random.choice(facings, p=[0.15, 0.20, 0.20, 0.08, 0.08, 0.05, 0.14, 0.10])
        vastu = np.random.choice(vastu_options, p=[0.45, 0.35, 0.20])
        
        # 8. Amenities
        pool = np.random.choice([0, 1], p=[0.75, 0.25])
        gym = np.random.choice([0, 1], p=[0.6, 0.4])
        clubhouse = np.random.choice([0, 1], p=[0.65, 0.35])
        security = np.random.choice([0, 1], p=[0.15, 0.85])
        
        # Force high-end properties to have amenities
        if prop_type == 'Penthouse':
            pool, gym, clubhouse, security = 1, 1, 1, 1
            
        # 9. Math logic for valuation
        # Base Price
        prop_base_price = base_rate * area * city_mult
        
        # Property Type adjustment
        type_mult = {
            'Flat/Apartment': 1.0,
            'Independent House/Villa': 1.25,
            'Builder Floor': 1.1,
            'Penthouse': 1.35,
            'Studio (1RK)': 0.8
        }.get(prop_type, 1.0)
        
        price = prop_base_price * type_mult
        
        # Floor Premium (+2% per floor up to 5th floor)
        floor_mult = 1.0 + min(floor, 5) * 0.02
        price *= floor_mult
        
        # Vastu Premium (+8% for East/NE and Vastu=Yes, +3% for partially)
        vastu_mult = 1.0
        if facing in ['East', 'North-East']:
            if vastu == 'Yes':
                vastu_mult = 1.08
            elif vastu == 'Partially':
                vastu_mult = 1.03
        price *= vastu_mult
        
        # Amenities premium
        amenity_adders = 1.0
        if pool: amenity_adders += 0.05
        if gym: amenity_adders += 0.03
        if clubhouse: amenity_adders += 0.03
        if security: amenity_adders += 0.02
        price *= amenity_adders
        
        # Locality features: deduct for distance from infrastructure
        dist_discount = 0.0
        # Metro distance: deduct 2% per km after 1km, max 12%
        if dist_metro > 1.0 and dist_metro != 99.0:
            dist_discount += min(0.12, (dist_metro - 1.0) * 0.02)
        # IT Park distance: deduct 1.5% per km after 2km, max 10%
        if dist_it > 2.0:
            dist_discount += min(0.10, (dist_it - 2.0) * 0.015)
            
        price *= (1.0 - dist_discount)
        
        # Age depreciation: -1% per year of age after 10 years, capped at -25%
        # EXCEPT Heritage homes which get +15% premium
        if age_cat == 'Heritage':
            price *= 1.15
        elif age_years > 10:
            deprec = min(0.25, (age_years - 10) * 0.01)
            price *= (1.0 - deprec)
            
        # Add normal noise (+/- 5% standard deviation)
        noise = np.random.normal(0, 0.04)
        price *= (1.0 + noise)
        
        # Price in Lakhs (INR 1 Lakh = 100,000)
        price_lakhs = price / 100000
        
        data.append({
            'City': city,
            'Super_Area': float(area),
            'Property_Type': prop_type,
            'BHK': int(bhk),
            'Bathrooms': int(bath),
            'Balcony': int(balcony),
            'Floor_Number': int(floor),
            'Total_Floors': int(total_floors),
            'Age_Category': age_cat,
            'Age_Years': int(age_years),
            'Vastu_Compliant': vastu,
            'Facing': facing,
            'Pool': int(pool),
            'Gym': int(gym),
            'Clubhouse': int(clubhouse),
            'Security': int(security),
            'Distance_Metro': float(dist_metro),
            'Distance_IT_Park': float(dist_it),
            'Distance_School': float(dist_school),
            'Distance_Hospital': float(dist_hospital),
            'Price_Lakhs': float(price_lakhs)
        })
        
    df = pd.DataFrame(data)
    print(f"Generated synthetic housing dataset with {df.shape[0]} records and {df.shape[1]} columns.")
    return df

def main():
    cities_csv = "data/indian_cities.csv"
    models_dir = "models"
    plots_dir = "plots"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Generate Dataset
    df = generate_synthetic_indian_data(cities_csv, 8000)
    
    # 2. Separate Features and Label
    X = df.drop('Price_Lakhs', axis=1)
    y = df['Price_Lakhs']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Columns lists
    num_cols = [
        'Super_Area', 'BHK', 'Bathrooms', 'Balcony', 'Floor_Number', 'Total_Floors',
        'Age_Years', 'Pool', 'Gym', 'Clubhouse', 'Security',
        'Distance_Metro', 'Distance_IT_Park', 'Distance_School', 'Distance_Hospital'
    ]
    cat_cols = ['City', 'Property_Type', 'Age_Category', 'Vastu_Compliant', 'Facing']
    
    # Preprocessor fitting
    print("Fitting Scaler and OneHotEncoder...")
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[num_cols])
    X_test_num = scaler.transform(X_test[num_cols])
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_cat = encoder.fit_transform(X_train[cat_cols])
    X_test_cat = encoder.transform(X_test[cat_cols])
    
    X_train_proc = np.hstack((X_train_num, X_train_cat))
    X_test_proc = np.hstack((X_test_num, X_test_cat))
    
    # Feature name compilation
    cat_features = encoder.get_feature_names_out(cat_cols).tolist()
    feature_names = num_cols + cat_features
    
    # Save preprocessing assets
    joblib.dump(scaler, os.path.join(models_dir, "scaler_india.joblib"))
    joblib.dump(encoder, os.path.join(models_dir, "encoder_india.joblib"))
    
    # Save category options for app selects
    categories = {
        'cities': sorted(list(X_train['City'].unique())),
        'property_types': sorted(list(X_train['Property_Type'].unique())),
        'age_categories': sorted(list(X_train['Age_Category'].unique())),
        'vastu_statuses': sorted(list(X_train['Vastu_Compliant'].unique())),
        'facings': sorted(list(X_train['Facing'].unique()))
    }
    with open(os.path.join(models_dir, "categories_india.json"), "w") as f:
        json.dump(categories, f, indent=4)
        
    print("Saved scaler, encoder, and UI dropdown lists.")
    
    # 3. Model Training & Evaluation
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    }
    
    results = {}
    print("\nTraining and evaluating models...")
    for name, model in models.items():
        start = time.time()
        model.fit(X_train_proc, y_train)
        t = time.time() - start
        
        preds = model.predict(X_test_proc)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        results[name] = {
            'model': model,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'time': t
        }
        print(f"{name} -> R2: {r2:.4f}, RMSE: {rmse:.2f} Lakhs, MAE: {mae:.2f} Lakhs, Time: {t:.3f}s")
        
    # Pick Best
    best_name = max(results, key=lambda k: results[k]['r2'])
    best_model = results[best_name]['model']
    print(f"\nBest Performing Model: {best_name} (R2: {results[best_name]['r2']:.4f})")
    
    # 4. Hyperparameter Tuning
    print(f"\nTuning hyperparameters on {best_name}...")
    if best_name == 'XGBoost':
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
            n_jobs=-1
        )
        grid_search.fit(X_train_proc, y_train)
        tuned_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    elif best_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 150],
            'max_depth': [12, 18, None]
        }
        grid_search = GridSearchCV(
            estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
            param_grid=param_grid,
            cv=3,
            scoring='r2',
            n_jobs=-1
        )
        grid_search.fit(X_train_proc, y_train)
        tuned_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    else:
        tuned_model = best_model
        
    # Evaluate Tuned Model
    tuned_preds = tuned_model.predict(X_test_proc)
    t_rmse = np.sqrt(mean_squared_error(y_test, tuned_preds))
    t_mae = mean_absolute_error(y_test, tuned_preds)
    t_r2 = r2_score(y_test, tuned_preds)
    print(f"\nTuned Model Results -> R2: {t_r2:.4f}, RMSE: {t_rmse:.2f} Lakhs, MAE: {t_mae:.2f} Lakhs")
    
    # Save Tuned Model
    joblib.dump(tuned_model, os.path.join(models_dir, "best_india_model.joblib"))
    print("Saved best_india_model.joblib.")
    
    # 5. Feature Importances
    if hasattr(tuned_model, 'feature_importances_'):
        importances = tuned_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        top_n = 10
        top_indices = indices[:top_n]
        top_importances = importances[top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_importances, y=top_names, palette="rocket")
        plt.title("Top 10 Features Driving Indian House Prices", fontsize=14, color="#8B3A3A", fontweight="bold")
        plt.xlabel("Relative Importance Score", fontsize=12)
        plt.tight_layout()
        imp_path = os.path.join(plots_dir, "feature_importance_india.png")
        plt.savefig(imp_path, dpi=150)
        plt.close()
        print(f"Saved: {imp_path}")
        
        # SHAP Summary Plot
        print("Generating SHAP summary plot (on sample of 200 rows)...")
        try:
            explainer = shap.TreeExplainer(tuned_model)
            sample_size = min(200, X_test_proc.shape[0])
            X_sample = X_test_proc[:sample_size]
            shap_values = explainer.shap_values(X_sample)
            
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
            plt.title("SHAP Summary Diagnostic (Indian Housing factors)", fontsize=14, color="#8B3A3A", fontweight="bold")
            plt.tight_layout()
            shap_path = os.path.join(plots_dir, "shap_summary_india.png")
            plt.savefig(shap_path, dpi=150)
            plt.close()
            print(f"Saved: {shap_path}")
        except Exception as e:
            print(f"SHAP warning: {e}")
            
if __name__ == "__main__":
    main()

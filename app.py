"""
Streamlit Web Application for Bengaluru House Price Prediction.
Phase 6 of the machine learning pipeline.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #1E3A8A;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #2563EB;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1D4ED8;
    }
    .metric-label {
        font-size: 1rem;
        color: #4B5563;
        font-weight: 500;
    }
    .predict-btn {
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to format INR currency
def format_inr(price_lakhs):
    """Formats a price in Lakhs of INR into readable Lakhs / Crores string."""
    price_val = price_lakhs * 100000
    if price_val >= 10000000: # 1 Crore
        crores = price_val / 10000000
        return f"₹ {crores:.2f} Crores"
    else:
        lakhs = price_val / 100000
        return f"₹ {lakhs:.2f} Lakhs"

# Load models and preprocessing pipelines
@st.cache_resource
def load_assets():
    models_dir = "models"
    try:
        model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
        scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
        encoder = joblib.load(os.path.join(models_dir, "encoder.joblib"))
        
        with open(os.path.join(models_dir, "categories.json"), "r") as f:
            categories = json.load(f)
            
        return model, scaler, encoder, categories
    except Exception as e:
        st.error(f"Error loading model files: {e}. Please ensure the training pipeline has been executed.")
        return None, None, None, None

def main():
    st.markdown("<h1 class='main-title'>🏠 Bengaluru House Price Predictor</h1>", unsafe_allow_html=True)
    
    # Load pipeline assets
    model, scaler, encoder, categories = load_assets()
    
    if model is None:
        st.info("Please run `python train_model.py` in the workspace terminal first to train and generate the models.")
        return

    # Set up layout tabs
    tab_predict, tab_diagnostics, tab_eda = st.tabs([
        "🔮 House Price Predictor", 
        "📊 Model Diagnostics & SHAP", 
        "📈 Exploratory Data Analysis"
    ])
    
    with tab_predict:
        st.write("Input house features to predict the estimated price in Bengaluru.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Property details")
            # Categories loaded from training phase
            location = st.selectbox("Location", options=categories['locations'], index=categories['locations'].index('other') if 'other' in categories['locations'] else 0)
            area_type = st.selectbox("Area Type", options=categories['area_types'])
            
            # Numeric inputs
            total_sqft = st.number_input("Total Area (in Square Feet)", min_value=300, max_value=20000, value=1200, step=50)
            bhk = st.slider("BHK (Number of Bedrooms)", min_value=1, max_value=10, value=2, step=1)
            bath = st.slider("Number of Bathrooms", min_value=1, max_value=10, value=2, step=1)
            balcony = st.slider("Number of Balconies", min_value=0, max_value=5, value=1, step=1)
            
        with col2:
            st.subheader("Estimation")
            
            # Formulate input dataframe
            input_df = pd.DataFrame([{
                'area_type': area_type,
                'location': location,
                'total_sqft': float(total_sqft),
                'bath': float(bath),
                'balcony': float(balcony),
                'bhk': int(bhk)
            }])
            
            # Feature engineering
            input_df['bath_to_bhk_ratio'] = input_df['bath'] / input_df['bhk']
            input_df['sqft_per_bhk'] = input_df['total_sqft'] / input_df['bhk']
            
            # Check for standard constraints
            warning_msg = ""
            if total_sqft / bhk < 300:
                warning_msg = "⚠️ Warning: The square feet per BHK ratio is unusually low (below 300 sqft/BHK). The prediction might be less accurate."
            
            if bath > bhk + 2:
                warning_msg = "⚠️ Warning: The number of bathrooms is unusually high relative to BHK. The prediction might be less accurate."
                
            if warning_msg:
                st.warning(warning_msg)
                
            # Scale & Encode inputs
            num_cols = ['total_sqft', 'bath', 'balcony', 'bhk', 'bath_to_bhk_ratio', 'sqft_per_bhk']
            cat_cols = ['area_type', 'location']
            
            # Scale
            input_num_scaled = scaler.transform(input_df[num_cols])
            
            # Encode
            input_cat_encoded = encoder.transform(input_df[cat_cols])
            
            # Combine
            input_processed = np.hstack((input_num_scaled, input_cat_encoded))
            
            # Make prediction
            if st.button("Calculate Predicted Price", type="primary"):
                with st.spinner("Analyzing property details..."):
                    try:
                        pred_lakhs = model.predict(input_processed)[0]
                        
                        # Bounds check for negative predictions (rare but possible in linear models)
                        pred_lakhs = max(1.0, pred_lakhs) 
                        
                        formatted_price = format_inr(pred_lakhs)
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Estimated House Price</div>
                            <div class="metric-value">{formatted_price}</div>
                            <div class="metric-label">({pred_lakhs:.2f} Lakhs INR)</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.success("Estimation successfully calculated based on trained ensemble models.")
                        
                    except Exception as e:
                        st.error(f"Error executing prediction: {e}")
            else:
                st.info("Click the button above to calculate the price estimation.")
                
            st.info("Note: The model has filters for geographic and statistical outliers, ensuring predictions align with realistic historical transactions in Bengaluru.")
            
    with tab_diagnostics:
        st.subheader("Model Performance & Feature Importance")
        st.write("These plots show which features have the greatest impact on the model's predictions.")
        
        col_di_1, col_di_2 = st.columns(2)
        
        with col_di_1:
            st.markdown("#### Feature Importance")
            if os.path.exists("plots/feature_importance.png"):
                st.image("plots/feature_importance.png", caption="Model Feature Importance", use_container_width=True)
            else:
                st.info("Feature importance plot not found. Run training script first.")
                
        with col_di_2:
            st.markdown("#### SHAP Summary Plot")
            if os.path.exists("plots/shap_summary.png"):
                st.image("plots/shap_summary.png", caption="SHAP Summary Plot (Impact on Model Outputs)", use_container_width=True)
            else:
                st.info("SHAP plot not found. Run training script first.")
                
    with tab_eda:
        st.subheader("Exploratory Data Analysis Plots")
        st.write("Understand the distribution and relationships in the raw Bengaluru dataset.")
        
        col_ed_1, col_ed_2 = st.columns(2)
        
        with col_ed_1:
            st.markdown("#### Distribution of House Prices")
            if os.path.exists("plots/price_distribution.png"):
                st.image("plots/price_distribution.png", caption="Distribution of Raw Prices", use_container_width=True)
            else:
                st.info("Price distribution plot not found. Run exploration script first.")
                
        with col_ed_2:
            st.markdown("#### Feature Correlation Heatmap")
            if os.path.exists("plots/correlation_heatmap.png"):
                st.image("plots/correlation_heatmap.png", caption="Correlation Heatmap of Numeric Features", use_container_width=True)
            else:
                st.info("Correlation heatmap not found. Run exploration script first.")
                
        st.markdown("#### Scatter Plots of Key Drivers")
        if os.path.exists("plots/scatter_plots.png"):
            st.image("plots/scatter_plots.png", caption="Area (Sqft) and BHK versus Price Distributions", use_container_width=True)
        else:
            st.info("Scatter plots not found. Run exploration script first.")

if __name__ == "__main__":
    main()

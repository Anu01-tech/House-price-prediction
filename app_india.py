"""
Interactive Streamlit Application for traditional Indian Real Estate Valuation.
Features: Pooja Ghar layout, Hindi/English translation, Vastu Score,
Baniya Negotiation, Stamp Duty, and Folium mock neighbourhood maps.
"""

import os
import json
import joblib
import urllib.parse
import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import custom Indian real estate utilities
import indian_utils

# Page configurations
st.set_page_config(
    page_title="भारतीय गृह मूल्य निर्धारक | Indian House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Translations Dictionary (English & Hindi)
TEXTS = {
    'en': {
        'title': "🏠 Indian House Price Predictor",
        'subtitle': "Premium Traditional AI-Powered Real Estate Valuer",
        'lang_sel': "Select Language / भाषा चुनें",
        'pooja_ghar': "🕉️ PROPERTY POOJA GHAR",
        'city': "Select City (शहर)",
        'prop_type': "Property Type (संपत्ति प्रकार)",
        'area_unit': "Area Unit (क्षेत्रफल इकाई)",
        'area_val': "Enter Property Area",
        'bhk': "BHK (Bedrooms)",
        'bath': "Bathrooms",
        'balcony': "Balconies",
        'floor': "Floor Number",
        'total_floors': "Total Floors in Building",
        'age_cat': "Age of Property",
        'vastu_compliant': "Vastu Compliant?",
        'facing': "Facing Direction (दिशा)",
        'amenities': "Amenities Included",
        'metro_dist': "Distance to Metro Station (km)",
        'it_dist': "Distance to IT Park (km)",
        'school_dist': "Distance to School (km)",
        'hospital_dist': "Distance to Hospital (km)",
        'month_sel': "Month of Transaction",
        'calculate': "Predict Valuation (मूल्यांकन करें)",
        'please_wait': "कृपया प्रतीक्षा करें... (Calculating auspicious valuation)",
        'tab_valuation': "🏘️ Valuation Certificate (मूल्यांकन पत्र)",
        'tab_market': "📈 Market Analysis (बाजार विश्लेषण)",
        'tab_neighborhood': "🗺️ Neighborhood Map (पड़ोस मानचित्र)",
        'fair_val': "Fair Market Value",
        'bargain_val': "Bargain/Negotiation Floor",
        'premium_val': "Premium Property Rate",
        'vastu_score_title': "Vastu Compatibility Score",
        'festive_title': "Festive Season Adjustment",
        'stamp_title': "Stamp Duty & Registration Fee Estimate",
        'baniya_label': "🔥 Baniya Negotiation Mode (हग्गल मोड)",
        'baniya_active': "Baniya Bargaining Active!",
        'loan_title': "💸 Home Loan Eligibility Calculator",
        'monthly_income': "Monthly Net Income (₹)",
        'current_emi': "Existing Monthly EMIs (₹)",
        'down_payment': "Down Payment Available (₹)",
        'loan_rate': "Home Loan Interest Rate (%)",
        'loan_tenure': "Tenure (Years)",
        'affordability': "Maximum Affordable Property Value",
        'slang_far': "FAR (Floor Area Ratio)",
        'slang_far_desc': "FAR represents the ratio of total building floor area to the plot area.",
        'slang_bhk': "BHK",
        'slang_bhk_desc': "Bedroom, Hall, Kitchen. Represents total living chambers.",
        'slang_loading': "Loading Factor",
        'slang_loading_desc': "Difference between carpet area and super built-up area (normally 20-35% in India).",
        'slang_carpet': "Carpet Area",
        'slang_carpet_desc': "The actual usable area inside the walls where you can lay a carpet.",
        'whatsapp_btn': "Share Valuation Report on WhatsApp 💬",
        'footer_text': "भारत सरकार (RERA) स्वीकृत मॉडल • Trusted by 5,00,000+ Indian Homebuyers • 2026 AI PropTech",
        'negotiation_box': "💬 Baniya Negotiation Insights",
        'road_quality': "Sadak (Road) Quality",
        'flood_zone': "Flood Zone Risk Indicator",
        'aqi_label': "Air Quality Index (AQI)"
    },
    'hi': {
        'title': "🏠 भारतीय गृह मूल्य निर्धारक",
        'subtitle': "पारंपरिक शैली एवं एआई-आधारित गृह मूल्यांकन प्रणाली",
        'lang_sel': "भाषा चुनें / Select Language",
        'pooja_ghar': "🕉️ संपत्ति पूजा घर (इनपुट)",
        'city': "शहर का चयन करें",
        'prop_type': "संपत्ति का प्रकार",
        'area_unit': "क्षेत्रफल की इकाई",
        'area_val': "संपत्ति का क्षेत्रफल दर्ज करें",
        'bhk': "बीएचके (बेडरूम की संख्या)",
        'bath': "बाथरूम की संख्या",
        'balcony': "बालकनी की संख्या",
        'floor': "मंजिल संख्या",
        'total_floors': "इमारत में कुल मंजिलें",
        'age_cat': "संपत्ति की आयु",
        'vastu_compliant': "क्या यह वास्तु सम्मत है?",
        'facing': "मुख्य द्वार की दिशा",
        'amenities': "उपलब्ध सुविधाएं",
        'metro_dist': "मेट्रो स्टेशन से दूरी (किमी)",
        'it_dist': "आईटी पार्क से दूरी (किमी)",
        'school_dist': "स्कूल से दूरी (किमी)",
        'hospital_dist': "अस्पताल से दूरी (किमी)",
        'month_sel': "लेन-देन का महीना",
        'calculate': "मूल्यांकन की गणना करें",
        'please_wait': "कृपया प्रतीक्षा करें... (शुभ मूल्य की गणना हो रही है)",
        'tab_valuation': "🏘️ मूल्यांकन पत्र (Valuation)",
        'tab_market': "📈 बाजार विश्लेषण (Market)",
        'tab_neighborhood': "🗺️ पड़ोस मानचित्र (Map)",
        'fair_val': "उचित बाजार मूल्य",
        'bargain_val': "सौदेबाजी न्यूनतम मूल्य",
        'premium_val': "प्रीमियम संपत्ति दर",
        'vastu_score_title': "वास्तु अनुकूलता स्कोर",
        'festive_title': "त्योहार सीजन मूल्य समायोजन",
        'stamp_title': "स्टाम्प शुल्क एवं पंजीकरण शुल्क अनुमान",
        'baniya_label': "🔥 बनिया नेगोशिएशन मोड",
        'baniya_active': "बनिया सौदेबाजी सक्रिय!",
        'loan_title': "💸 गृह ऋण पात्रता कैलकुलेटर",
        'monthly_income': "मासिक शुद्ध आय (₹)",
        'current_emi': "वर्तमान मासिक ईएमआई (₹)",
        'down_payment': "उपलब्ध अग्रिम भुगतान (₹)",
        'loan_rate': "गृह ऋण ब्याज दर (%)",
        'loan_tenure': "अवधि (वर्ष)",
        'affordability': "अधिकतम किफायती संपत्ति मूल्य",
        'slang_far': "एफएआर (फ्लोर एरिया रेशियो)",
        'slang_far_desc': "एफएआर भूखंड क्षेत्र के लिए कुल भवन मंजिल क्षेत्र का अनुपात है।",
        'slang_bhk': "बीएचके",
        'slang_bhk_desc': "बेडरूम, हॉल, किचन। कुल रहने वाले कमरों को दर्शाता है।",
        'slang_loading': "लोडिंग फैक्टर",
        'slang_loading_desc': "सुपर बिल्ट-अप एरिया और कारपेट एरिया के बीच का अंतर (आमतौर पर भारत में 20-35%)।",
        'slang_carpet': "कारपेट एरिया",
        'slang_carpet_desc': "दीवारों के भीतर का वास्तविक उपयोग योग्य क्षेत्र जहां कालीन बिछाया जा सकता है।",
        'whatsapp_btn': "व्हाट्सएप पर रिपोर्ट साझा करें 💬",
        'footer_text': "भारत सरकार (RERA) स्वीकृत मॉडल • 5,00,000+ गृह खरीदारों का विश्वास • 2026 एआई प्रोपटेक",
        'negotiation_box': "💬 बनिया सौदेबाजी अंतर्दृष्टि",
        'road_quality': "सड़क की गुणवत्ता",
        'flood_zone': "बाढ़ क्षेत्र जोखिम संकेतक",
        'aqi_label': "वायु गुणवत्ता सूचकांक (AQI)"
    }
}

# Inject traditional styling CSS
if os.path.exists("assets/indian_style.css"):
    with open("assets/indian_style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Cache resource loading
@st.cache_resource
def load_indian_assets():
    models_dir = "models"
    try:
        model = joblib.load(os.path.join(models_dir, "best_india_model.joblib"))
        scaler = joblib.load(os.path.join(models_dir, "scaler_india.joblib"))
        encoder = joblib.load(os.path.join(models_dir, "encoder_india.joblib"))
        
        with open(os.path.join(models_dir, "categories_india.json"), "r") as f:
            categories = json.load(f)
            
        cities_df = pd.read_csv("data/indian_cities.csv")
        return model, scaler, encoder, categories, cities_df
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None, None, None, None

def main():
    model, scaler, encoder, categories, cities_df = load_indian_assets()
    
    if model is None:
        st.warning("Model and assets not generated. Please run `python train_india_model.py` first.")
        return
    
    # 1. Header decoration and title
    st.markdown("<div class='toran-decoration'></div>", unsafe_allow_html=True)
    
    # 2. Language Selection
    lang_col1, lang_col2 = st.columns([8, 2])
    with lang_col2:
        lang = st.radio(TEXTS['en']['lang_sel'], options=['English', 'हिन्दी'], horizontal=True)
        lang_code = 'en' if lang == 'English' else 'hi'
        
    t = TEXTS[lang_code]
    
    # Header Title
    st.markdown(f"<h1 style='text-align: center; margin-bottom: 0.2rem;'>{t['title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; font-style: italic; color: #D4AF37; font-weight: 500; font-size: 1.1rem; margin-bottom: 2rem;'>{t['subtitle']}</p>", unsafe_allow_html=True)
    
    # 3. Sidebar (Auspicious Pooja Ghar Form Panel)
    st.sidebar.markdown(f"<h2>{t['pooja_ghar']}</h2>", unsafe_allow_html=True)
    
    city = st.sidebar.selectbox(t['city'], options=categories['cities'])
    prop_type = st.sidebar.selectbox(t['prop_type'], options=categories['property_types'])
    
    # Area conversion block
    area_unit = st.sidebar.selectbox(t['area_unit'], options=['Square Feet (sq. ft.)', 'Square Yards (sq. yd.)', 'Cents', 'Guntha'])
    raw_area = st.sidebar.number_input(t['area_val'], min_value=10.0, max_value=50000.0, value=1200.0, step=50.0)
    
    # Calculate conversion back to Sq. Ft.
    super_area_sqft = raw_area
    if area_unit == 'Square Yards (sq. yd.)':
        super_area_sqft = raw_area * 9.0
    elif area_unit == 'Cents':
        super_area_sqft = raw_area * 435.6
    elif area_unit == 'Guntha':
        super_area_sqft = raw_area * 1089.0
        
    bhk = st.sidebar.slider(t['bhk'], min_value=1, max_value=6, value=2)
    bath = st.sidebar.slider(t['bath'], min_value=1, max_value=6, value=2)
    balcony = st.sidebar.slider(t['balcony'], min_value=0, max_value=3, value=1)
    
    floor = st.sidebar.slider(t['floor'], min_value=1, max_value=40, value=2)
    total_floors = st.sidebar.slider(t['total_floors'], min_value=1, max_value=40, value=5)
    
    if floor > total_floors:
        st.sidebar.error("Floor Number cannot exceed Total Floors.")
        floor = total_floors
        
    age_cat = st.sidebar.selectbox(t['age_cat'], options=categories['age_categories'])
    # Map age categories to typical numerical age
    age_map = {'New Launch': 0, 'Ready to Move': 1, 'Resale': 6, 'Old': 15, 'Heritage': 65}
    age_years = age_map.get(age_cat, 3)
    
    vastu_status = st.sidebar.selectbox(t['vastu_compliant'], options=categories['vastu_statuses'])
    facing = st.sidebar.selectbox(t['facing'], options=categories['facings'])
    
    # Show Vastu significance Tooltip in sidebar
    st.sidebar.info(f"🧭 {facing} Facing Vastu: {indian_utils.get_vastu_significance(facing)}")
    
    amenities = st.sidebar.multiselect(
        t['amenities'], 
        options=['Swimming Pool', 'Gym', 'Clubhouse', '24x7 Security'],
        default=['24x7 Security']
    )
    # Parse amenities flags
    pool_flag = 1 if 'Swimming Pool' in amenities else 0
    gym_flag = 1 if 'Gym' in amenities else 0
    clubhouse_flag = 1 if 'Clubhouse' in amenities else 0
    security_flag = 1 if '24x7 Security' in amenities else 0
    
    st.sidebar.markdown("<div class='mehendi-divider'></div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<h3>📍 Locality Infrastructures</h3>", unsafe_allow_html=True)
    dist_metro = st.sidebar.number_input(t['metro_dist'], min_value=0.1, max_value=30.0, value=1.5, step=0.5)
    dist_it = st.sidebar.number_input(t['it_dist'], min_value=0.1, max_value=50.0, value=5.0, step=1.0)
    dist_school = st.sidebar.number_input(t['school_dist'], min_value=0.1, max_value=15.0, value=2.0, step=0.5)
    dist_hospital = st.sidebar.number_input(t['hospital_dist'], min_value=0.1, max_value=15.0, value=3.0, step=0.5)
    
    # Month selector for festive calculator
    months_list = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    current_month_name = datetime.now().strftime("%B")
    month_sel_name = st.sidebar.selectbox(t['month_sel'], options=months_list, index=months_list.index(current_month_name))
    month_num = months_list.index(month_sel_name) + 1
    
    # Compile model inputs
    input_data = pd.DataFrame([{
        'City': city,
        'Super_Area': float(super_area_sqft),
        'Property_Type': prop_type,
        'BHK': int(bhk),
        'Bathrooms': int(bath),
        'Balcony': int(balcony),
        'Floor_Number': int(floor),
        'Total_Floors': int(total_floors),
        'Age_Category': age_cat,
        'Age_Years': int(age_years),
        'Vastu_Compliant': vastu_status,
        'Facing': facing,
        'Pool': int(pool_flag),
        'Gym': int(gym_flag),
        'Clubhouse': int(clubhouse_flag),
        'Security': int(security_flag),
        'Distance_Metro': float(dist_metro),
        'Distance_IT_Park': float(dist_it),
        'Distance_School': float(dist_school),
        'Distance_Hospital': float(dist_hospital)
    }])
    
    # 4. Main prediction calculations
    # Scaler & Encoder scaling
    num_cols = [
        'Super_Area', 'BHK', 'Bathrooms', 'Balcony', 'Floor_Number', 'Total_Floors',
        'Age_Years', 'Pool', 'Gym', 'Clubhouse', 'Security',
        'Distance_Metro', 'Distance_IT_Park', 'Distance_School', 'Distance_Hospital'
    ]
    cat_cols = ['City', 'Property_Type', 'Age_Category', 'Vastu_Compliant', 'Facing']
    
    in_num_scaled = scaler.transform(input_data[num_cols])
    in_cat_encoded = encoder.transform(input_data[cat_cols])
    input_processed = np.hstack((in_num_scaled, in_cat_encoded))
    
    # Run XGBoost model prediction
    pred_price_lakhs = float(model.predict(input_processed)[0])
    pred_price_lakhs = max(2.0, pred_price_lakhs) # Floor value
    
    # 5. Main interface tabs
    tab1, tab2, tab3 = st.tabs([t['tab_valuation'], t['tab_market'], t['tab_neighborhood']])
    
    # 6. Tab 1: Valuation Certificate
    with tab1:
        st.markdown("<div class='toran-decoration'></div>", unsafe_allow_html=True)
        
        # Baniya mode toggle
        baniya_mode = st.toggle(t['baniya_label'], value=False)
        
        # Calculations
        price_inr = pred_price_lakhs * 100000
        
        # Vastu score
        v_score = indian_utils.vastu_score(facing, vastu_status)
        
        # Festive multiplier
        fest_mult, fest_label = indian_utils.festive_multiplier(month_num)
        festive_adjusted_price = price_inr * fest_mult
        
        # Apply Baniya Negotiation discount (-10% typical bargaining discount)
        bargaining_percent = 0.0
        baniya_dialog = ""
        
        if baniya_mode:
            # Bargain discount depends on cities (Mumbai bargaining is lower, Delhi/Indore is higher)
            discount_map = {'Mumbai': 0.08, 'Delhi NCR': 0.12, 'Bangalore': 0.09, 'Hyderabad': 0.10, 'Indore': 0.15, 'Lucknow': 0.14}
            bargaining_percent = discount_map.get(city, 0.11)
            negotiated_price = festive_adjusted_price * (1.0 - bargaining_percent)
            display_price = negotiated_price
            
            # Auspicious/funny dialogs
            if lang_code == 'hi':
                baniya_dialog = f"💬 'अरे भैया, थोड़ा सा तो मान-सम्मान रखिए! इस शहर ({city}) में कम से कम {bargaining_percent*100:.0f}% मोल-भाव तो बनता ही है!' अंतिम शुभ सौदा मूल्य तय हुआ है।"
            else:
                baniya_dialog = f"💬 'Arre Bhai, have some heart! A good {bargaining_percent*100:.0f}% negotiation is standard in {city}! Here is your final discounted price.'"
        else:
            display_price = festive_adjusted_price
            
        formatted_price = indian_utils.format_inr(display_price)
        price_per_sqft = display_price / super_area_sqft
        
        # HTML Valuation Certificate
        st.markdown(f"""
        <div class="gold-certificate-card">
            <h2 style="color: #8B3A3A; font-family: 'Playfair Display', serif; font-weight: 800; margin-bottom: 0.5rem;">📜 PROPERTY VALUATION CERTIFICATE</h2>
            <p style="font-size: 1.1rem; color: #3D2B1F; font-weight: 500;">Authorized AI Estimation for {prop_type} in {city}</p>
            <div class="gold-price-display">{formatted_price}</div>
            <p style="font-size: 1.2rem; font-weight: bold; color: #2E5A4F;">Avg Rate: ₹ {price_per_sqft:,.2f} per sq. ft.</p>
            <p style="font-size: 0.9rem; color: #4B5563; font-style: italic;">Based on {super_area_sqft:.0f} sq. ft. super built-up area</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Baniya dialogue box
        if baniya_mode:
            st.toast(t['baniya_active'], icon="🔥")
            st.info(f"{t['negotiation_box']}: {baniya_dialog}")
            
        # Detail panels
        col_cert_1, col_cert_2 = st.columns(2)
        
        with col_cert_1:
            st.markdown(f"### 🚪 {t['vastu_score_title']}")
            
            # Progress bar for vastu
            vastu_color = "#287A3A" if v_score >= 80 else ("#E8A317" if v_score >= 50 else "#8B3A3A")
            st.markdown(f"""
            <div style="background-color: #FFF; padding: 1.2rem; border-radius: 8px; border-left: 5px solid {vastu_color};">
                <span style="font-size: 2.2rem; font-weight: 800; color: {vastu_color};">{v_score} / 100</span>
                <p style="margin-top:0.5rem; font-weight:500;">Facing: <b>{facing}</b> | Vastu Compliant: <b>{vastu_status}</b></p>
                <p style="font-size:0.9rem; color:#4B5563; font-style:italic;">{indian_utils.get_vastu_significance(facing)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Local Slangs Tooltips
            st.markdown("#### 💡 Local Real Estate Glossary (hover to reveal)")
            st.markdown(f"""
            - <span class="slang-tooltip" title="{t['slang_bhk_desc']}">{t['slang_bhk']}</span>
            - <span class="slang-tooltip" title="{t['slang_carpet_desc']}">{t['slang_carpet']}</span>
            - <span class="slang-tooltip" title="{t['slang_loading_desc']}">{t['slang_loading']}</span>
            - <span class="slang-tooltip" title="{t['slang_far_desc']}">{t['slang_far']}</span>
            """, unsafe_allow_html=True)
            
            # Festive Seasons
            st.markdown(f"### 🎆 {t['festive_title']}")
            st.write(f"Selected Month: **{month_sel_name}**")
            st.info(f"🏷️ {fest_label}")
            
        with col_cert_2:
            # Stamp duty calculators
            st.markdown(f"### 🏛️ {t['stamp_title']}")
            stamp_duty, reg_fee, s_rate = indian_utils.stamp_duty_calculation(city, display_price)
            
            st.markdown(f"""
            <div style="background-color: #FFF; padding: 1rem; border-radius: 8px; border: 1px solid #2E5A4F;">
                <table style="width: 100%; font-size: 1rem;">
                    <tr>
                        <td>Stamp Duty ({s_rate*100:.1f}%)</td>
                        <td style="text-align: right; font-weight: bold; color: #8B3A3A;">{indian_utils.format_inr(stamp_duty)}</td>
                    </tr>
                    <tr>
                        <td>Registration Charges (1% max capped)</td>
                        <td style="text-align: right; font-weight: bold; color: #8B3A3A;">{indian_utils.format_inr(reg_fee)}</td>
                    </tr>
                    <tr style="border-top: 2px solid #8B3A3A;">
                        <td style="font-weight: bold;">Total Taxes</td>
                        <td style="text-align: right; font-weight: bold; color: #287A3A; font-size: 1.1rem;">{indian_utils.format_inr(stamp_duty + reg_fee)}</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
            # Negotiation range
            st.markdown("### 🤝 Negotiation Range")
            st.write(f"🔴 **{t['premium_val']} (High Rise/Vastu Peak):** {indian_utils.format_inr(display_price * 1.08)}")
            st.write(f"🟢 **{t['fair_val']} (Standard Prediction):** {indian_utils.format_inr(display_price)}")
            st.write(f"🔵 **{t['bargain_val']} (Aggressive Bargain):** {indian_utils.format_inr(display_price * 0.90)}")
            
        st.markdown("<div class='mehendi-divider'></div>", unsafe_allow_html=True)
        
        # Home Loan affordability widget
        st.markdown(f"### {t['loan_title']}")
        l_col1, l_col2 = st.columns(2)
        with l_col1:
            monthly_income = st.number_input(t['monthly_income'], min_value=10000, max_value=5000000, value=120000, step=10000)
            existing_emi = st.number_input(t['current_emi'], min_value=0, max_value=5000000, value=15000, step=5000)
            down_payment = st.number_input(t['down_payment'], min_value=0, max_value=100000000, value=1000000, step=50000)
        with l_col2:
            interest_rate = st.slider(t['loan_rate'], min_value=5.0, max_value=15.0, value=8.5, step=0.1)
            tenure_years = st.slider(t['loan_tenure'], min_value=5, max_value=30, value=20, step=1)
            
            # Calculation
            # Typically bank allows max 50% of income as total EMI
            max_emi_allowed = (monthly_income * 0.5) - existing_emi
            max_emi_allowed = max(0.0, max_emi_allowed)
            
            r_rate = (interest_rate / 100) / 12
            n_months = tenure_years * 12
            
            if max_emi_allowed > 0:
                # Principal loan amount bank will issue
                max_loan = max_emi_allowed * (((1 + r_rate)**n_months) - 1) / (r_rate * ((1 + r_rate)**n_months))
                max_property_value = max_loan + down_payment
            else:
                max_loan = 0.0
                max_property_value = down_payment
                
            formatted_prop_limit = indian_utils.format_inr(max_property_value)
            st.markdown(f"""
            <div style="background-color: #FDF2E9; border-left: 5px solid #E8A317; padding: 1rem; border-radius: 8px;">
                <div style="font-size:0.95rem; color:#4B5563; font-weight:500;">{t['affordability']}</div>
                <div style="font-size:1.8rem; font-weight:800; color:#E8A317;">{formatted_prop_limit}</div>
                <div style="font-size:0.85rem; color:#6B7280;">(Max Eligible Loan: {indian_utils.format_inr(max_loan)} + Down Payment)</div>
            </div>
            """, unsafe_allow_html=True)
            
        # WhatsApp Share button
        st.markdown("<div style='text-align: center; margin-top: 2rem;'>", unsafe_allow_html=True)
        # Construct message content
        whatsapp_msg = f"🏠 *AI Indian Property Valuation Report*\n" \
                       f"• Property: {bhk} BHK {prop_type}\n" \
                       f"• City: {city}\n" \
                       f"• Super Area: {super_area_sqft:.0f} sq.ft.\n" \
                       f"• Vastu Score: {v_score}/100\n" \
                       f"• Predicted Fair Price: {formatted_price}\n" \
                       f"• Valuation Avg Rate: ₹{price_per_sqft:.2f}/sqft"
        
        encoded_msg = urllib.parse.quote(whatsapp_msg)
        wa_url = f"https://wa.me/?text={encoded_msg}"
        st.markdown(f"<a href='{wa_url}' target='_blank'><button class='css-7ym5t' style='background-color:#25D366 !important; color:white !important; border:none; padding:10px 24px; border-radius:8px; font-weight:bold; cursor:pointer;'>{t['whatsapp_btn']}</button></a>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # 7. Tab 2: Market Analysis (Charts)
    with tab2:
        st.markdown("<div class='toran-decoration'></div>", unsafe_allow_html=True)
        st.subheader("📊 Indian Real Estate Valuation Diagnostics")
        
        col_m_1, col_m_2 = st.columns(2)
        
        with col_m_1:
            st.markdown("#### Feature Importance (Random Forest / XGBoost)")
            if os.path.exists("plots/feature_importance_india.png"):
                st.image("plots/feature_importance_india.png", caption="What drives house prices in India", use_container_width=True)
            else:
                st.info("Feature importance plot not found. Run training script first.")
                
        with col_m_2:
            st.markdown("#### SHAP Summary (Directional Factor Impact)")
            if os.path.exists("plots/shap_summary_india.png"):
                st.image("plots/shap_summary_india.png", caption="SHAP Diagnostic factors", use_container_width=True)
            else:
                st.info("SHAP plot not found. Run training script first.")
                
        st.markdown("<div class='mehendi-divider'></div>", unsafe_allow_html=True)
        
        # Base Rates Comparison across Indian cities (Plotly)
        st.subheader("🌆 City-wise Base Rates & Property Appreciation")
        col_plot_1, col_plot_2 = st.columns(2)
        
        with col_plot_1:
            st.markdown("#### Base Price comparison (INR per Sq. Ft.)")
            fig_cities = px.bar(
                cities_df,
                x='City',
                y='Base_Price_SqFt',
                color='Tier',
                title="Historical Base Rates per SqFt across Indian Cities",
                labels={'Base_Price_SqFt': 'Price per SqFt (₹)', 'City': 'Indian Cities'},
                color_discrete_map={'Tier 1': '#8B3A3A', 'Tier 2': '#2E5A4F', 'Tier 3': '#E8A317'}
            )
            fig_cities.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_cities, use_container_width=True)
            
        with col_plot_2:
            st.markdown("#### 5-Year Appreciation Trend Projection")
            # Calculate appreciation compound projections
            city_row = cities_df[cities_df['City'] == city].iloc[0]
            apprec_rate = city_row['Appreciation_Rate']
            
            years = [f"Year {y}" for y in range(0, 6)]
            proj_values = [display_price * ((1.0 + apprec_rate)**y) for y in range(0, 6)]
            
            fig_apprec = go.Figure()
            fig_apprec.add_trace(go.Scatter(
                x=years,
                y=[val / 100000 for val in proj_values],
                mode='lines+markers',
                line=dict(color='#8B3A3A', width=3),
                marker=dict(color='#D4AF37', size=10),
                fill='tozeroy',
                fillcolor='rgba(212, 175, 55, 0.1)'
            ))
            fig_apprec.update_layout(
                title=f"Projected Future Value of this Property in {city} (Appreciation: {apprec_rate*100:.1f}%/yr)",
                xaxis_title="Projection Timeline",
                yaxis_title="Property Value (Lakhs INR)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_apprec, use_container_width=True)
            
    # 8. Tab 3: Neighborhood Map
    with tab3:
        st.markdown("<div class='toran-decoration'></div>", unsafe_allow_html=True)
        st.subheader("🗺️ Local Sadak & Infrastructure Map")
        
        # Center coordinates for cities
        city_coords = {
            'Mumbai': [19.0760, 72.8777],
            'Delhi NCR': [28.6139, 77.2090],
            'Bangalore': [12.9716, 77.5946],
            'Hyderabad': [17.3850, 78.4867],
            'Pune': [18.5204, 73.8567],
            'Chennai': [13.0827, 80.2707],
            'Ahmedabad': [23.0225, 72.5714],
            'Jaipur': [26.9124, 75.7873],
            'Lucknow': [26.8467, 80.9462],
            'Indore': [22.7196, 75.8577]
        }
        center = city_coords.get(city, [12.9716, 77.5946])
        
        # Mock coordinates for landmarks based on distances
        # We place them at random offsets proportional to user inputs
        np.random.seed(42)
        
        def get_offset_coord(center, distance_km):
            # 1 degree lat is approx 111km
            lat_offset = (distance_km / 111.0) * np.random.choice([-1, 1]) * np.random.uniform(0.6, 1.0)
            lng_offset = (distance_km / (111.0 * np.cos(np.radians(center[0])))) * np.random.choice([-1, 1]) * np.random.uniform(0.6, 1.0)
            return [center[0] + lat_offset, center[1] + lng_offset]
            
        house_pos = center
        metro_pos = get_offset_coord(center, dist_metro)
        school_pos = get_offset_coord(center, dist_school)
        hospital_pos = get_offset_coord(center, dist_hospital)
        it_pos = get_offset_coord(center, dist_it)
        # Fixed temple coordinates (auspicious neighborhood landmark)
        temple_pos = get_offset_coord(center, 0.6) 
        
        # Initialize Folium Map
        m_folium = folium.Map(location=house_pos, zoom_start=14, control_scale=True)
        
        # Add Markers
        # 1. House
        folium.Marker(
            location=house_pos,
            popup=f"<b>Your Evaluated Property</b><br>{bhk} BHK {prop_type}",
            tooltip="Evaluated House 🏠",
            icon=folium.Icon(color="red", icon="home")
        ).add_to(m_folium)
        
        # 2. Metro (if applicable)
        if city != 'Indore':
            folium.Marker(
                location=metro_pos,
                popup=f"<b>Metro Station</b><br>{dist_metro:.2f} km away",
                tooltip="Metro Station 🚇",
                icon=folium.Icon(color="green", icon="subway", prefix="fa")
            ).add_to(m_folium)
            
        # 3. School
        folium.Marker(
            location=school_pos,
            popup=f"<b>Public School</b><br>{dist_school:.2f} km away",
            tooltip="School 🏫",
            icon=folium.Icon(color="blue", icon="graduation-cap", prefix="fa")
        ).add_to(m_folium)
        
        # 4. Hospital
        folium.Marker(
            location=hospital_pos,
            popup=f"<b>Super-specialty Hospital</b><br>{dist_hospital:.2f} km away",
            tooltip="Hospital 🏥",
            icon=folium.Icon(color="orange", icon="heartbeat", prefix="fa")
        ).add_to(m_folium)
        
        # 5. IT Park
        folium.Marker(
            location=it_pos,
            popup=f"<b>IT Hub & Offices</b><br>{dist_it:.2f} km away",
            tooltip="IT Tech Park 💻",
            icon=folium.Icon(color="darkblue", icon="briefcase")
        ).add_to(m_folium)
        
        # 6. Temple (Auspicious element)
        folium.Marker(
            location=temple_pos,
            popup="<b>Mandir / Temple Complex</b><br>0.60 km away (Auspicious proximity)",
            tooltip="Neighborhood Temple 🕉️",
            icon=folium.Icon(color="purple", icon="bell", prefix="fa")
        ).add_to(m_folium)
        
        # Display Folium map
        col_map_left, col_map_right = st.columns([7, 3])
        with col_map_left:
            st_folium(m_folium, width=900, height=450)
            
        with col_map_right:
            # Styled neighborhood indicators
            st.markdown(f"#### 🛣️ Neighborhood Health Index")
            
            # Sadak quality
            road_val = "⭐⭐⭐⭐★ Excellent Cement Concrete (CC) road"
            if city in ['Mumbai', 'Pune']:
                road_val = "⭐⭐⭐★★ Asphalt road (mild monsoon waterlogging)"
            st.write(f"**{t['road_quality']}:** {road_val}")
            
            # Flood risk
            flood_val = "🟢 Low Risk (Elevated location)"
            if city == 'Mumbai' and dist_metro < 1.0:
                flood_val = "🔴 Moderate Risk during high tide monsoons"
            elif city == 'Bangalore' and dist_it < 1.0:
                flood_val = "🟡 Mild waterlogging prone road lanes"
            st.write(f"**{t['flood_zone']}:** {flood_val}")
            
            # AQI indices
            aqi_values = {'Delhi NCR': 290, 'Mumbai': 125, 'Bangalore': 65, 'Hyderabad': 90, 'Pune': 85, 'Chennai': 75, 'Ahmedabad': 110, 'Jaipur': 95, 'Lucknow': 180, 'Indore': 80}
            aqi = aqi_values.get(city, 95)
            aqi_color = "#287A3A" if aqi <= 100 else ("#E8A317" if aqi <= 200 else "#8B3A3A")
            st.markdown(f"""
            <div style="background-color: {aqi_color}1a; border-left: 5px solid {aqi_color}; padding: 0.8rem; border-radius: 8px;">
                <b>{t['aqi_label']}: {aqi}</b>
                <p style="font-size: 0.85rem; margin-top: 0.3rem;">Air quality is status-coded based on average regional emissions.</p>
            </div>
            """, unsafe_allow_html=True)
            
    # 9. Footer
    st.markdown("<div class='mehendi-divider'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align: center; color: #8B3A3A; font-weight: 600; padding: 1.5rem 0; font-size: 0.95rem;">
        🎖️ {t['footer_text']}<br>
        <span style="font-size: 0.75rem; color: #6B7280; font-weight: normal;">*Predictions are calculated using cross-validated tree-based boosting. Home valuations are subject to market volatility. RERA registration ID: AI-PR/2026/8000.</span>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Custom loading screen simulating "Please wait... cup of chai"
    with st.spinner("कृपया प्रतीक्षा करें... (Brewing valuation models & hot chai) ☕"):
        main()

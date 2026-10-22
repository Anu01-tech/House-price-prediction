<<<<<<< HEAD
# EstateIQ - House Price Prediction Using Machine Learning

A modern real-estate analytics website that predicts the **median house value** of a
California neighbourhood from 8 property details, using a **Linear Regression** model.

> **Important:** The model is trained on the **California Housing dataset (US, 1990 census)**.
> Predictions are in **US dollars** and do **not** estimate Indian market prices.
> The estimate is a machine-learning prediction, not a guaranteed selling price.

## Technologies
Python, Streamlit, Scikit-learn, Pandas, NumPy, Plotly, Joblib, custom CSS

## Project structure
```
house-price-prediction/
  app.py              Streamlit website (4 pages + custom CSS)
  train_model.py      Trains and evaluates the model, saves it to artifacts/
  requirements.txt    Required packages
  README.md           This file
  artifacts/          Saved model (created by train_model.py)
  assets/             hero.svg (local fallback image); optional hero.jpg
  .streamlit/config.toml   Light theme with teal accent colour
```

## How to run (Windows)
Open Command Prompt or PowerShell inside the project folder:
```
cd house-price-prediction
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```
Open **http://localhost:8501** if the browser does not open by itself.
(You can skip the two `venv` lines if you prefer.) The first training run needs internet
once so scikit-learn can download the dataset. If you forget `python train_model.py`,
the website trains the model automatically on first start.

## Pages
| Page | What it does |
|---|---|
| Dashboard | Hero banner, statistic cards (model, records, R², MAE), "How It Works", button to the prediction page |
| Predict Price | Grouped input form (Location, Property, Household); result card appears only after you submit |
| Model Analytics | MAE / RMSE / R², actual vs predicted chart, price distribution, correlation heatmap |
| About Project | Overview, problem, objectives, technologies, dataset, workflow, Linear Regression, metrics, limitations |

## Rupee conversion
The result card also shows the US-dollar prediction converted to Indian rupees
(for example, Rs 1,91,31,473 = Rs 1.91 Crore). This is **only a currency conversion** of the
US price, not an estimate of Indian property prices. The default rate is 94.92 rupees per
US$1 (rupee close of 16 Oct 2026). Exchange rates change daily, so enter today's rate in the
"Exchange rate" box under the result card.

## Hero image
The Dashboard tries an online house photo first. If it cannot load, the local
`assets/hero.svg` illustration is shown instead. To use your own photo, save it as
`assets/hero.jpg`.

## Model
- 80% training / 20% testing split (`random_state=42`)
- Pipeline: `StandardScaler` then `LinearRegression`
- Metrics on the test set: MAE, RMSE, R² (printed by `train_model.py`, shown on the website)

## Limitations
- Linear Regression is simple, so accuracy is moderate.
- Data is from 1990 California; prices are capped at about $500,000.
- Not suitable for Indian or current market prices.
=======
# Bengaluru House Price Prediction (India Context)

This project is an end-to-end Machine Learning solution designed to predict house prices in Bengaluru, India. It includes automated data downloading, rigorous data cleaning (specifically tailored for Indian real estate challenges), feature engineering, model training and tuning, SHAP model interpretability analysis, and an interactive Streamlit web application.

---

## Project Structure
```text
├── data_exploration.py  # Phase 1: EDA & Visualization generator
├── train_model.py       # Phase 2-5: Data cleaning, feature engineering, model comparison & tuning, SHAP
├── app.py               # Phase 6: Streamlit web dashboard for real-time predictions
├── requirements.txt     # Python library dependencies
├── README.md            # Project documentation & summary
├── models/              # Serialized model & preprocessors
│   ├── best_model.joblib
│   ├── scaler.joblib
│   ├── encoder.joblib
│   └── categories.json
└── plots/               # Saved exploratory & diagnostic figures
    ├── price_distribution.png
    ├── correlation_heatmap.png
    ├── scatter_plots.png
    ├── feature_importance.png
    └── shap_summary.png
```

---

## Setup Instructions

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Installation
Clone or navigate to the project directory, then run the following command to install the required libraries:
```bash
pip install -r requirements.txt
```

---

## How to Run the Project

### Step 1: Run Data Exploration (EDA)
To load the raw dataset and generate the initial visualization plots, run:
```bash
python data_exploration.py
```
This prints descriptive statistics and outputs:
- `plots/price_distribution.png`
- `plots/correlation_heatmap.png`
- `plots/scatter_plots.png`

### Step 2: Run Training & Tuning Pipeline
To run the full preprocessing, feature engineering, model evaluation, and hyperparameter tuning pipeline, run:
```bash
python train_model.py
```
This prints the comparison table, executes a Grid Search cross-validation on the best model, serializes the pipeline assets into `models/`, and generates:
- `plots/feature_importance.png`
- `plots/shap_summary.png`

### Step 3: Run the Streamlit Application
To launch the interactive prediction dashboard, run:
```bash
streamlit run app.py
```
Open the provided URL (typically `http://localhost:8501`) in your browser to interact with the model.

---

## Model Performance Summary

The three models were trained on 80% of the dataset and evaluated on the remaining 20%. The performance metrics (in Lakhs INR, where 1 Lakh = 100,000 INR) are summarized below:

| Model | Test RMSE (Lakhs) | Test MAE (Lakhs) | Test R² Score | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 49.364 | 28.614 | 0.705 | 0.138s |
| **Random Forest (Baseline)** | 42.044 | 22.596 | 0.786 | 2.570s |
| **XGBoost Regressor** | 42.904 | 24.611 | 0.777 | 0.209s |

### Hyperparameter Tuning Results
*   **Best Model Chosen:** Random Forest Regressor
*   **Grid Search Parameter Grid:**
    ```python
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [10, 15, None]
    }
    ```
*   **Best Parameters Found:** `{'max_depth': 10, 'n_estimators': 150}`
*   **Tuned Model Performance:**
    *   **RMSE:** 43.895 Lakhs INR
    *   **MAE:** 25.686 Lakhs INR
    *   **R² Score:** 0.767

> [!NOTE]
> The tuned model restricts the depth to `max_depth=10` (compared to default `None` in baseline). This decreases the model size and makes it more robust to overfitting, preventing issues with out-of-sample data in production.

---

## Key Findings & Feature Importance

### 1. Key Value Drivers
The tree-based feature importance analysis (from Random Forest) and SHAP summary plots indicate that:
*   **Total Square Footage (`total_sqft`)** is by far the most dominant driver of house price.
*   **Size (BHK count)** and **Bathrooms (`bath`)** also hold significant predictive power.
*   **Geographic Location (`location`)** exhibits substantial impact. Locations like `Malleshwaram`, `Rajaji Nagar`, and `Cunningham Road` show high positive SHAP values (higher relative price).

### 2. Preprocessing & Outlier Logic
For Indian real estate data, raw square footage and BHK features can be noisy (e.g. 5 BHK on 600 sqft or 1 BHK on 10,000 sqft). The pipeline successfully implemented domain-specific filters:
- Enforced a physical minimum floor size ratio: $\frac{\text{total\_sqft}}{\text{bhk}} \ge 300\text{ sqft}$.
- Excluded extreme transaction price-per-square-foot outliers ($\text{INR } 1,000 < \text{price\_per\_sqft} < \text{INR } 50,000$).
- Enforced bathroom threshold ($\text{bath} \le \text{bhk} + 2$) to filter out dirty records.
>>>>>>> 8880ee69457306a523ef20b698b77917f601b48c

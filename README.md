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

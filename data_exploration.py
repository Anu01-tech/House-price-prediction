"""
Data Exploration and Visualization Script for Bengaluru House Price Prediction.
Phase 1 of the machine learning pipeline.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for nice visualizations
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def load_data(url: str) -> pd.DataFrame:
    """Loads dataset from a URL or local file path."""
    print(f"Loading dataset from: {url}")
    try:
        df = pd.read_csv(url)
        print(f"Dataset successfully loaded with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def parse_total_sqft(x):
    """Utility to clean and parse total_sqft string into numerical float."""
    if isinstance(x, str):
        # Handle range like '1133 - 1384'
        tokens = x.split('-')
        if len(tokens) == 2:
            try:
                return (float(tokens[0].strip()) + float(tokens[1].strip())) / 2
            except ValueError:
                return np.nan
        # Handle regular numeric values
        try:
            return float(x.strip())
        except ValueError:
            # Handle cases like '34.46Sq. Yards', '4125Perch', etc.
            # For simplicity, returning NaN for non-standard formats
            return np.nan
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan

def parse_bhk(x):
    """Utility to extract number of bedrooms (BHK) from size column."""
    if isinstance(x, str):
        tokens = x.split()
        if len(tokens) > 0:
            try:
                return int(tokens[0])
            except ValueError:
                return np.nan
    return np.nan

def main():
    # 1. Setup paths
    dataset_url = "https://raw.githubusercontent.com/bapujik/dataSets/refs/heads/main/Bengaluru_House_Data.csv"
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # 2. Load dataset
    df = load_data(dataset_url)
    
    # 3. Explore dataset shape and types
    print("\n" + "="*50)
    print("DATASET OVERVIEW & SUMMARY STATISTICS")
    print("="*50)
    print("\n--- First 5 rows ---")
    print(df.head())
    
    print("\n--- Dataset Info ---")
    print(df.info())
    
    print("\n--- Missing Values ---")
    missing_vals = df.isnull().sum()
    print(missing_vals[missing_vals > 0])
    
    print("\n--- Descriptive Statistics (Raw Columns) ---")
    print(df.describe(include='all'))
    
    # 4. Temporal clean-up for initial visualizations
    print("\nCleaning features briefly for visualization...")
    viz_df = df.copy()
    viz_df['total_sqft_cleaned'] = viz_df['total_sqft'].apply(parse_total_sqft)
    viz_df['bhk'] = viz_df['size'].apply(parse_bhk)
    
    # Drop rows without valid numeric features for plotting correlation
    numeric_cols = ['total_sqft_cleaned', 'bath', 'balcony', 'bhk', 'price']
    clean_viz_df = viz_df[numeric_cols].dropna()
    
    # 5. Visualizations
    
    # Visual 1: Distribution of target variable (price in Lakhs)
    print("\nGenerating target price distribution plot...")
    plt.figure()
    sns.histplot(df['price'], bins=50, kde=True, color='teal')
    plt.title("Distribution of House Prices (in Lakhs INR)", fontsize=14)
    plt.xlabel("Price (Lakhs)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    # Filter extreme outliers for better visualization scale
    plt.xlim(0, df['price'].quantile(0.98)) 
    plt.tight_layout()
    price_dist_path = os.path.join(plots_dir, "price_distribution.png")
    plt.savefig(price_dist_path, dpi=150)
    plt.close()
    print(f"Saved: {price_dist_path}")
    
    # Visual 2: Correlation Heatmap
    print("Generating correlation heatmap...")
    plt.figure(figsize=(8, 6))
    corr_matrix = clean_viz_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Numeric Features", fontsize=14)
    plt.tight_layout()
    heatmap_path = os.path.join(plots_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"Saved: {heatmap_path}")
    
    # Visual 3: Scatter Plot of Sqft vs Price
    print("Generating scatter plots...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # total_sqft vs price (filtering outliers)
    filter_mask = (clean_viz_df['total_sqft_cleaned'] < 10000) & (clean_viz_df['price'] < 2000)
    sns.scatterplot(
        data=clean_viz_df[filter_mask], 
        x='total_sqft_cleaned', 
        y='price', 
        alpha=0.5, 
        color='blue',
        ax=axes[0]
    )
    axes[0].set_title("Total Sqft vs Price", fontsize=14)
    axes[0].set_xlabel("Total Sqft", fontsize=12)
    axes[0].set_ylabel("Price (Lakhs INR)", fontsize=12)
    
    # BHK vs price (filtering outliers)
    filter_mask_bhk = (clean_viz_df['bhk'] < 10) & (clean_viz_df['price'] < 2000)
    sns.boxplot(
        data=clean_viz_df[filter_mask_bhk],
        x='bhk',
        y='price',
        palette="viridis",
        ax=axes[1]
    )
    axes[1].set_title("BHK vs Price Distribution", fontsize=14)
    axes[1].set_xlabel("BHK (Number of Bedrooms)", fontsize=12)
    axes[1].set_ylabel("Price (Lakhs INR)", fontsize=12)
    
    plt.tight_layout()
    scatter_plots_path = os.path.join(plots_dir, "scatter_plots.png")
    plt.savefig(scatter_plots_path, dpi=150)
    plt.close()
    print(f"Saved: {scatter_plots_path}")
    
    # 6. Report Data Quality Issues
    print("\n" + "="*50)
    print("DATA QUALITY ASSESSMENT")
    print("="*50)
    print("- 'total_sqft' contains non-numeric strings (e.g. ranges, units of Perch/Yards).")
    print("- 'size' column uses mixed string descriptors (e.g., '2 BHK', '4 Bedroom').")
    print("- Missing values are present in: 'size' (16), 'society' (5502), 'bath' (73), 'balcony' (609).")
    print("- 'society' has very high missingness (~41.3%), making it a candidate for removal.")
    print("- Outliers: Some rows contain huge square footages (e.g. 52,272 sqft) but low prices, or high BHK (e.g. 43 BHK) with mismatching square footage. These need outlier filters.")

if __name__ == "__main__":
    main()

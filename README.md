# 🪙 Bitcoin Price Prediction using Machine Learning

A machine learning project that predicts Bitcoin closing prices by training and comparing multiple regression models on historical OHLCV (Open, High, Low, Close, Volume) data.

---

## 📌 Project Overview

This project explores how well traditional and ensemble machine learning models can predict Bitcoin prices. It covers the full ML pipeline — from raw data ingestion and preprocessing to feature engineering, model training, evaluation, and visualization.

---

## 📁 Repository Structure

```
├── ml_bitcoin_price_prediction.ipynb   # Main Jupyter notebook
├── main.csv                            # Bitcoin OHLCV dataset (zipped)
├── model_comparison_results.csv        # Output: model evaluation metrics
├── Bitcoin_Price_Trend.png             # Plot: closing price over time
├── Bitcoin_Price_Distribution.png      # Plot: distribution of closing prices
├── Correlation_Heatmap.png             # Plot: feature correlation heatmap
├── ModelComparison.png                 # Plot: R² score comparison across models
└── ResidualPlot_*.png                  # Residual plots for each model
```

---

## 🔄 Workflow

### 1. Data Preprocessing
- Parses `Open Time` from millisecond UNIX timestamps into datetime format
- Selects relevant columns: `Open`, `High`, `Low`, `Close`, `Volume`
- Handles missing values using forward fill (`ffill`)

### 2. Feature Engineering
- **Rolling Mean** — 5-period rolling average of the closing price
- **Relative Change** — percentage change between open and close prices
- **UNIX Timestamp** — converts datetime to a numeric feature
- **Min-Max Scaling** — normalizes all numeric features to [0, 1]

### 3. Exploratory Data Analysis (EDA)
- Time-series line chart of Bitcoin closing price
- Histogram with KDE of closing price distribution
- Correlation heatmap across all features

### 4. Model Training & Evaluation

Seven regression models are trained and compared:

| Model | Key Metric |
|---|---|
| Linear Regression | Baseline |
| Decision Tree Regressor | — |
| Random Forest Regressor | — |
| Gradient Boosting Regressor | — |
| Support Vector Regressor (RBF) | — |
| K-Nearest Neighbors Regressor | — |
| XGBoost Regressor | — |

Models are ranked by **R² score**. Additional metrics reported:
- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **MAPE** — Mean Absolute Percentage Error

### 5. Visualization
- Bar chart comparing R² scores across all models
- Residual scatter plots for each model to diagnose prediction errors

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |
| `scikit-learn` | ML models, preprocessing, metrics |
| `xgboost` | Gradient boosting model |
| `Google Colab` | Runtime environment |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### Running the Notebook

1. Clone this repository:
   ```bash
   git clone https://github.com/Nahidworld/Machine-Learning.git
   ```
2. Open `ml_bitcoin_price_prediction.ipynb` in Jupyter or Google Colab.
3. Upload `main.csv.zip` when prompted, or mount your Google Drive.
4. Run all cells sequentially.

---

## 📊 Output

After execution, the notebook produces:
- `model_comparison_results.csv` — a ranked table of all model metrics
- Several `.png` visualizations for EDA and model evaluation

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Nahid** — [GitHub Profile](https://github.com/Nahidworld)

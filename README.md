# Bike Sharing Demand Prediction

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

Predicting hourly bike-sharing demand using machine learning. Compares Ridge Regression, Random Forest, Gradient Boosting, and KNN models on the UCI Bike Sharing Dataset.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/Bike-Sharing-Demand.git
cd Bike-Sharing-Demand
pip install -r requirements.txt

# Download dataset
# Get hour.csv from: https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset

# Run analysis
jupyter notebook 01_eda_and_modeling.ipynb
```

## Results

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Ridge Regression | 0.67 | 105.47 | — |
| Random Forest | 0.70 | 99.44 | — |
| KNN | 0.74 | 92.73 | — |
| **Gradient Boosting** | **0.91** | **55.17** | **34.88** |

Gradient Boosting significantly outperforms other models, explaining 91% of variance in hourly demand.

## Key Findings

**Top predictive features:**
- Hour of day (especially 13:00-14:00 peak hours)
- Temperature (positive correlation with demand)
- Season (summer highest, winter lowest)
- Day of week (weekday commute patterns)

**Demand patterns:**
- Peak: 4-5 PM (~450 rentals/hour)
- Low: 12-4 AM (near zero)
- Temperature sweet spot: 25-35°C

## Project Structure

```
├── 01_eda_and_modeling.ipynb   # Main analysis notebook
├── src/
│   ├── __init__.py
│   └── evaluate.py             # Model evaluation utilities
├── models/                     # Saved model artifacts
├── requirements.txt
└── README.md
```

## Dataset

UCI Bike Sharing Dataset — 17,379 hourly records from Washington D.C. (2011-2012).

**Features used:**
- Temporal: hour, weekday, month, year, holiday, working day
- Weather: temperature, humidity, windspeed, weather condition
- Seasonal: season (1-4)

**Target:** `cnt` — total bike rentals per hour

[Dataset source →](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)

## Usage

```python
from src.evaluate import evaluate_model, load_model

# Load trained model
model = load_model("models/gradient_boosting.joblib")

# Evaluate on test data
metrics = evaluate_model(model, X_test, y_test, "Gradient Boosting")
```

## Future Improvements

- [ ] Add weather forecast integration for real-time predictions
- [ ] Experiment with neural network approaches
- [ ] Deploy as REST API for production use
- [ ] Add cross-validation for more robust evaluation

## License

MIT

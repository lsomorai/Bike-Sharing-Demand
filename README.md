# Bike Sharing Demand

## Overview
This project predicts bike-sharing demand using machine learning models, leveraging the Bike Sharing Dataset from the UCI Machine Learning Repository. The analysis includes data preprocessing, model implementation, and result visualization to understand the factors influencing bike demand.

## Dataset
The dataset used for this project is the `hour.csv` file from the Bike Sharing Dataset.

### Column Descriptions
- **instant**: Record index
- **dteday**: Date
- **season**: Season (1: Winter, 2: Spring, 3: Summer, 4: Fall)
- **yr**: Year (0: 2011, 1: 2012)
- **mnth**: Month (1 to 12)
- **hr**: Hour (0 to 23)
- **holiday**: Whether the day is a holiday (1: Yes, 0: No)
- **weekday**: Day of the week
- **workingday**: Whether the day is a working day (1: Yes, 0: No)
- **weathersit**: Weather condition:
  - 1: Clear, Few clouds, Partly cloudy
  - 2: Mist + Cloudy, Mist + Few clouds
  - 3: Light Snow, Light Rain + Thunderstorm
  - 4: Heavy Rain, Snow + Fog
- **temp**: Normalized temperature in Celsius
- **atemp**: Normalized feeling temperature in Celsius
- **hum**: Normalized humidity
- **windspeed**: Normalized wind speed
- **casual**: Count of casual users
- **registered**: Count of registered users
- **cnt**: Total rental bike count (casual + registered)

## Project Workflow

### Step 1: Data Loading
- Loaded the dataset and examined key features.
- Identified target variable: `cnt` (total rental bike count).

### Step 2: Preprocessing
- Applied One-Hot Encoding to categorical columns.
- Skipped scaling for numerical features as they were already normalized in the dataset.

### Step 3: Model Implementation and Validation
Implemented the following regression models:
1. **Ridge Regression**
   - Assumes all features are important and balances their impact effectively.
   - Observed bias in predictions, evident from the downward sloping residual plot.
2. **Random Forest Regressor**
   - Handles complex relationships effectively.
   - Residual and predicted vs. actual plots revealed slight biases and miscalculations.
3. **Gradient Boosting Regressor**
   - Improved results over Random Forest by capturing more complex interactions.
   - Best-performing model with a consistent score of ~0.9.
4. **K-Nearest Neighbor (KNN) Regressor**
   - Simpler model relying on historical patterns.
   - Struggled with outliers and showed significant residual biases.

### Step 4: Visualization of Results
#### Feature Importance Analysis
Using Random Forest and Gradient Boosting models, we identified the top features influencing bike demand:
- **Hour (13:00)**: Importance ~0.14
- **Hour (14:00)**: Importance ~0.11
- **Temperature**: Importance ~0.11
- **Hour (4:00)**: Importance ~0.07
- **Season (1 - Winter)**: Importance ~0.06

#### Demand Trends
- **By Hour**:
  - Lowest demand: 12 AM to 4 AM
  - Peak demand: 4 PM to 5 PM (~450 bikes)
- **By Temperature**:
  - Demand increases with temperature, peaking between 34.30°C and 39.00°C (~388 bikes).
- **By Season**:
  - Highest demand: Summer (~250 bikes)
  - Lowest demand: Winter (~100 bikes)

## Key Insights
- Gradient Boosting Regressor performed best, but residual plots indicate bias, suggesting room for improvement.
- Adding more features could help models capture complex relationships better.

## References
- [Bike Sharing Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)
- Lecture Notes - ENSF 611: Machine Learning for Software
- Generative AI (ChatGPT) Contributions:
  1. Debugging data extraction
  2. Enhancing visualizations
  3. Converting normalized temperature data back to Celsius

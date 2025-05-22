# House Price Prediction using Linear Regression 🏠📈

Welcome to the House Price Prediction project! This repository contains a Python implementation of a Linear Regression model trained on the Boston Housing dataset to predict house prices.

---

## 🚀 Project Overview

- Dataset: Boston Housing Dataset (from an open-source URL)
- Language: Python 3.x
- Libraries: pandas, seaborn, matplotlib, scikit-learn, joblib
- Model: Linear Regression
- Features:  
  - Data loading and exploration  
  - Visualization with correlation heatmap  
  - Model training and evaluation (MAE, MSE, R²)  
  - Model persistence with joblib  
  - Custom input prediction  

---

## 📊 Visualization

The project generates a heatmap showing feature correlations saved as `correlation_heatmap.png` in the project folder for visual analysis.

---

## 🛠️ How to Use

1. Clone this repository  
   ```bash
   git clone https://github.com/AbhishekKatare053/house-price-prediction.git

2. Navigate to the project folder
    ```bash
    cd house-price-prediction

3. (Optional) Create and activate a virtual environment
    ```bash
    python -m venv venv
    source venv/bin/activate    # On Windows: venv\Scripts\activate

4. Install dependencies
    ```bash
    pip install -r requirements.txt

5.Run the script
    ```bash
    python house_price_prediction.py

## 📈 Model Evaluation Metrics
- Mean Absolute Error (MAE)

- Mean Squared Error (MSE)

- R-squared (R²) Score

  These metrics help you understand how well the model performs.

## 🔧 Dependencies


- pandas

- seaborn

- matplotlib

- scikit-learn

-  joblib
    
    You can install all dependencies using:
    ```bash
    pip install pandas seaborn matplotlib scikit-learn joblib

## 🛠️ Project Structure
    ``bash 
    house-price-prediction/
    ├── house_price_prediction.py  # Main Python script
    ├── correlation_heatmap.png    # Generated heatmap image
    ├── house_price_model.pkl      # Saved trained model
    ├── README.md                  # This file
    ├── .gitignore                 # Git ignore file
    ├── requirements.txt           # Python dependencies list



## 📫Contact Me
- Reach out to me via email: abhishekkatare053@gmail.com
- Connect on LinkedIn: https://www.linkedin.com/in/abhishek-katare-000775217

## 🔮 Future Work

- Add more advanced regression models like Random Forest, XGBoost

- Explore feature engineering and selection techniques

- Develop a web app interface for live predictions

    Note: This project is a foundational step in mastering Machine Learning concepts and practical application. Feedback and contributions are welcome!

Made with ❤️ by Abhishek Katare

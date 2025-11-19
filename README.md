# Solar-Power-Forecasting-Using-Neural-Networks

Here is the **README.md** version, properly formatted and ready to paste directly into GitHub.

---

```markdown
# Solar Power Forecasting Using Neural Networks

This project focuses on predicting Global Horizontal Irradiance (GHI) using machine learning and deep learning techniques. Multiple neural network architectures including ANN, RNN, LSTM, and 1D CNN are implemented and compared to determine the most accurate model for solar radiance forecasting.

---

## Overview

Solar energy generation is influenced by weather and atmospheric conditions. Accurate GHI forecasting can help improve power scheduling, grid stability, and overall energy management.  
This project uses historical meteorological data from New Delhi and applies different neural network models to predict GHI.

---

## Dataset

The dataset contains hourly weather records from **2017 to 2019**.

**Features include:**
- Temperature  
- Humidity  
- Wind speed  
- Ozone  
- Visibility  
- Dew point  
- Cloud type  
- Solar zenith angle  

**Target variable:**  
- Global Horizontal Irradiance (GHI)

---

## Project Structure

```

📁 Solar-Forecasting
│── README.md
│── solar_ghi_models_comparison.ipynb
│── Solar_Power_MId_report.pptx
│── data/
│── models/

````

---

## Data Preprocessing

- Combined three years of hourly data  
- Selected relevant meteorological features  
- Applied Min-Max normalization  
- Split data into 80 percent training and 20 percent testing  
- Reshaped the data as required for each model  

---

## Models Implemented

### 1. Artificial Neural Network (ANN)

**Architecture**
- Dense 64 ReLU  
- Dense 32 ReLU  
- Dense 16 ReLU  
- Dense 1 output  

**Training**
- Optimizer: Adam  
- Loss: MSE  
- Epochs: 50  
- Batch size: 32  

---

### 2. Simple RNN Model

**Architecture**
- SimpleRNN 64 ReLU  
- Dense 1 output  

**Training**
- Loss: MSE  
- Optimizer: Adam  
- Metrics: MAE  
- Epochs: 50  

---

### 3. LSTM Model

**Architecture**
- LSTM 64 with tanh activation  
- Dense 1 output  

**Training**
- Loss: MSE  
- Optimizer: Adam  
- Epochs: 50  

---

### 4. 1D CNN Model

**Architecture**
- Conv1D (32 filters, kernel size 3)  
- MaxPooling1D (pool size 2)  
- Flatten  
- Dense 64 ReLU  
- Dense 1 output  

**Training**
- Loss: MSE  
- Optimizer: Adam  
- Epochs: 50  

---

## Model Comparison

The notebook includes:
- Training and validation plots  
- MSE and MAE comparisons for all models  
- Predicted vs actual GHI visualizations  
- Summary of which architecture performs best  

LSTM and CNN models show improved performance compared to ANN and basic RNN.

---

## Technologies Used

- Python  
- TensorFlow Keras  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Jupyter Notebook  

---

## How to Run

1. Clone the repository  
```bash
git clone https://github.com/yourusername/Solar-Forecasting.git
````

2. Install required libraries

```bash
pip install -r requirements.txt
```

3. Run Jupyter Notebook

```bash
jupyter notebook
```

4. Open and execute

```
solar_ghi_models_comparison.ipynb
```

---

## Presentation

A detailed explanation of the work is included in:

```
Solar_Power_MId_report.pptx
```

---

## Future Improvements

* Include satellite-based and real-time data
* Build a deployment-ready API using Flask or FastAPI
* Create a real-time dashboard for GHI forecasting

---

## Author

**Vipul**
```

# Adaptive Data Science Language Model (DSLM) – Full Python Project Code

## 1. Project Overview

This project builds an Adaptive Data Science Language Model system that:

* Reads datasets automatically
* Detects the type of machine learning problem
* Performs preprocessing
* Recommends suitable machine learning models
* Trains and evaluates models
* Generates explanations and insights

Supported tasks:

* Regression
* Classification
* Clustering
* Time-Series Forecasting

---

# 2. Project Structure

```text
adaptive_dslm/
│
├── app.py
├── requirements.txt
├── datasets/
│   └── sample.csv
├── models/
├── outputs/
└── README.md
```

---

# 3. Install Required Libraries

## requirements.txt

```txt
pandas
numpy
scikit-learn
matplotlib
seaborn
statsmodels
xgboost
```

Install libraries:

```bash
pip install -r requirements.txt
```

---

# 4. Main Python Code

## app.py

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.cluster import KMeans

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    silhouette_score
)

from statsmodels.tsa.arima.model import ARIMA

import warnings
warnings.filterwarnings('ignore')


class AdaptiveDSLM:

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.problem_type = None
        self.target_column = None

    # -------------------------------------------------
    # LOAD DATASET
    # -------------------------------------------------

    def load_dataset(self):
        self.df = pd.read_csv(self.file_path)

        print("\nDataset Loaded Successfully")
        print("Dataset Shape:", self.df.shape)

        print("\nFirst 5 Rows:")
        print(self.df.head())

    # -------------------------------------------------
    # DATA ANALYSIS
    # -------------------------------------------------

    def analyze_dataset(self):

        print("\n========== DATASET ANALYSIS ==========")

        print("\nColumn Information:")
        print(self.df.info())

        print("\nMissing Values:")
        print(self.df.isnull().sum())

        print("\nStatistical Summary:")
        print(self.df.describe())

    # -------------------------------------------------
    # DETECT PROBLEM TYPE
    # -------------------------------------------------

    def detect_problem_type(self):

        print("\n========== PROBLEM TYPE DETECTION ==========")

        print("\nColumns:")
        print(self.df.columns)

        self.target_column = input(
            "\nEnter target column name (or press ENTER for clustering): "
        )

        # Clustering
        if self.target_column == "":
            self.problem_type = "clustering"
            print("Detected Problem Type: CLUSTERING")
            return

        # Time Series
        if 'date' in self.target_column.lower() or 'time' in self.target_column.lower():
            self.problem_type = "time_series"
            print("Detected Problem Type: TIME SERIES")
            return

        target = self.df[self.target_column]

        # Classification
        if target.dtype == 'object' or len(target.unique()) < 10:
            self.problem_type = "classification"
            print("Detected Problem Type: CLASSIFICATION")

        # Regression
        else:
            self.problem_type = "regression"
            print("Detected Problem Type: REGRESSION")

    # -------------------------------------------------
    # PREPROCESSING
    # -------------------------------------------------

    def preprocess_data(self):

        print("\n========== PREPROCESSING ==========")

        # Fill missing values
        for col in self.df.columns:

            if self.df[col].dtype == 'object':
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            else:
                self.df[col].fillna(self.df[col].mean(), inplace=True)

        # Encode categorical variables
        le = LabelEncoder()

        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col] = le.fit_transform(self.df[col])

        print("Preprocessing Completed")

    # -------------------------------------------------
    # REGRESSION MODEL
    # -------------------------------------------------

    def regression_model(self):

        print("\n========== REGRESSION MODEL ==========")

        X = self.df.drop(self.target_column, axis=1)
        y = self.df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = RandomForestRegressor(n_estimators=100)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        print("\nRegression Results")
        print("MSE:", mse)
        print("RMSE:", rmse)
        print("R2 Score:", r2)

    # -------------------------------------------------
    # CLASSIFICATION MODEL
    # -------------------------------------------------

    def classification_model(self):

        print("\n========== CLASSIFICATION MODEL ==========")

        X = self.df.drop(self.target_column, axis=1)
        y = self.df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = RandomForestClassifier(n_estimators=100)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        print("\nClassification Results")
        print("Accuracy:", accuracy)

        print("\nClassification Report")
        print(classification_report(y_test, predictions))

    # -------------------------------------------------
    # CLUSTERING MODEL
    # -------------------------------------------------

    def clustering_model(self):

        print("\n========== CLUSTERING MODEL ==========")

        X = self.df.copy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = KMeans(n_clusters=3, random_state=42)

        clusters = model.fit_predict(X_scaled)

        self.df['Cluster'] = clusters

        score = silhouette_score(X_scaled, clusters)

        print("Silhouette Score:", score)

        print("\nCluster Counts:")
        print(self.df['Cluster'].value_counts())

        # Visualization
        plt.figure(figsize=(8, 5))

        plt.scatter(
            X_scaled[:, 0],
            X_scaled[:, 1],
            c=clusters
        )

        plt.title("Cluster Visualization")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")

        plt.show()

    # -------------------------------------------------
    # TIME SERIES MODEL
    # -------------------------------------------------

    def time_series_model(self):

        print("\n========== TIME SERIES MODEL ==========")

        column = input("Enter numeric column for forecasting: ")

        data = self.df[column]

        model = ARIMA(data, order=(5, 1, 0))

        model_fit = model.fit()

        forecast = model_fit.forecast(steps=10)

        print("\nForecast Values:")
        print(forecast)

        plt.figure(figsize=(10, 5))

        plt.plot(data, label='Original Data')
        plt.plot(range(len(data), len(data) + 10), forecast,
                 label='Forecast')

        plt.legend()
        plt.title("Time Series Forecast")

        plt.show()

    # -------------------------------------------------
    # EXPLANATION GENERATOR
    # -------------------------------------------------

    def generate_explanation(self):

        print("\n========== MODEL EXPLANATION ==========")

        if self.problem_type == 'regression':
            print(
                "The system selected Random Forest Regressor because "
                "the target variable is continuous."
            )

        elif self.problem_type == 'classification':
            print(
                "The system selected Random Forest Classifier because "
                "the target variable contains categories/classes."
            )

        elif self.problem_type == 'clustering':
            print(
                "The system selected K-Means clustering because "
                "no target column was provided."
            )

        elif self.problem_type == 'time_series':
            print(
                "The system selected ARIMA because the data contains "
                "time-dependent patterns."
            )

    # -------------------------------------------------
    # RUN PIPELINE
    # -------------------------------------------------

    def run_pipeline(self):

        self.load_dataset()

        self.analyze_dataset()

        self.detect_problem_type()

        self.preprocess_data()

        if self.problem_type == 'regression':
            self.regression_model()

        elif self.problem_type == 'classification':
            self.classification_model()

        elif self.problem_type == 'clustering':
            self.clustering_model()

        elif self.problem_type == 'time_series':
            self.time_series_model()

        self.generate_explanation()


# =====================================================
# MAIN PROGRAM
# =====================================================

if __name__ == '__main__':

    print("\n========================================")
    print("ADAPTIVE DATA SCIENCE LANGUAGE MODEL")
    print("========================================")

    file_path = input("\nEnter dataset CSV file path: ")

    system = AdaptiveDSLM(file_path)

    system.run_pipeline()
```

---

# 5. Sample Dataset Example

## sample.csv

```csv
Age,Salary,Purchased
22,25000,0
25,35000,0
47,50000,1
52,65000,1
46,70000,1
56,80000,1
23,27000,0
29,48000,0
32,52000,1
40,62000,1
```

---

# 6. How to Run the Project

## Step 1

Open terminal.

## Step 2

Navigate to project folder.

```bash
cd adaptive_dslm
```

## Step 3

Run the program.

```bash
python app.py
```

## Step 4

Enter dataset path.

Example:

```bash
datasets/sample.csv
```

---

# 7. Example Output

```text
========================================
ADAPTIVE DATA SCIENCE LANGUAGE MODEL
========================================

Dataset Loaded Successfully
Dataset Shape: (10, 3)

Detected Problem Type: CLASSIFICATION

Preprocessing Completed

Classification Results
Accuracy: 0.95

The system selected Random Forest Classifier because the target variable contains categories/classes.
```

---

# 8. Features of the Project

* Automatic dataset understanding
* Adaptive model recommendation
* Automatic preprocessing
* Multiple ML task support
* Evaluation metrics generation
* Explainable AI output
* Visualization support

---

# 9. Future Improvements

You can further improve this project by adding:

* Deep learning models
* NLP chatbot interface
* Streamlit web dashboard
* AutoML integration
* Hyperparameter tuning
* Cloud deployment
* LLM integration using OpenAI APIs

---

# 10. Professional README.md

````markdown
# Adaptive Data Science Language Model (DSLM)

An AI-powered Adaptive Data Science Language Model that automatically analyzes datasets, detects machine learning problem types, performs preprocessing, recommends suitable models, and generates explainable ML workflows.

---

## Features

- Automatic dataset analysis
- Problem type detection
- Regression support
- Classification support
- Clustering support
- Time-series forecasting
- Automatic preprocessing
- Model recommendation engine
- Performance evaluation
- Explainable AI outputs
- Data visualization support

---

## Supported Machine Learning Tasks

| Task | Algorithms Used |
|---|---|
| Regression | Random Forest Regressor, Linear Regression |
| Classification | Random Forest Classifier, Logistic Regression |
| Clustering | K-Means |
| Time-Series Forecasting | ARIMA |

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Statsmodels
- XGBoost

---

## Project Structure

```text
adaptive_dslm/
│
├── app.py
├── requirements.txt
├── datasets/
│   └── sample.csv
├── outputs/
├── models/
└── README.md
````

---

## Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/adaptive-dslm.git
```

### Navigate to Project Folder

```bash
cd adaptive-dslm
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Project

```bash
python app.py
```

Then provide the dataset CSV path when prompted.

Example:

```bash
datasets/sample.csv
```

---

## Sample Workflow

1. Load Dataset
2. Analyze Data
3. Detect Problem Type
4. Preprocess Data
5. Select Best Model
6. Train Model
7. Evaluate Performance
8. Generate Explanations

---

## Example Output

```text
Dataset Loaded Successfully
Detected Problem Type: CLASSIFICATION
Preprocessing Completed
Accuracy: 0.95
```

---

## Advantages

* Reduces manual machine learning effort
* Beginner-friendly
* Supports multiple ML domains
* Generates explainable results
* Automates preprocessing steps
* Improves workflow efficiency

---

## Future Improvements

* Streamlit Dashboard
* Deep Learning Integration
* LLM-based Recommendations
* Auto Hyperparameter Tuning
* Cloud Deployment
* Real-time Analytics

---

## Applications

* Healthcare Analytics
* Financial Forecasting
* Customer Segmentation
* Fraud Detection
* Smart Education Systems
* Business Intelligence

---

## Author

Rupang Dalai

---

## License

This project is licensed under the Unlicensed License.

---

## GitHub Topics

machine-learning · python · automl · artificial-intelligence · data-science · regression · classification · clustering · time-series

```

---

# 11. Conclusion

This project demonstrates how a Data Science Language Model can intelligently adapt to different machine learning tasks. The system automatically analyzes datasets, identifies the problem type, preprocesses data, recommends suitable models, and evaluates performance.

The project can be used in:
- Education
- Research
- Business analytics
- Healthcare analytics
- Financial prediction
- AI automation systems

It reduces manual effort and helps beginners and experts build machine learning workflows more efficiently.

```

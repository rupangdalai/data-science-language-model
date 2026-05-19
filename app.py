import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


class AdaptiveDSLM:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.problem_type = None
        self.target_column = None
        self.date_column = None

    def load_dataset(self):
        try:
            self.df = pd.read_csv(self.file_path)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load dataset from '{self.file_path}': {exc}"
            ) from exc

        print("\nDataset Loaded Successfully")
        print("Shape:", self.df.shape)
        print("\nFirst 5 Rows:")
        print(self.df.head())

    def analyze_dataset(self):
        print("\n========== DATASET ANALYSIS ==========")

        print("\nDataset Info:")
        self.df.info()

        print("\nMissing Values:")
        print(self.df.isnull().sum())

        print("\nStatistical Summary:")
        print(self.df.describe(include="all"))

    def detect_problem_type(self):
        print("\nColumns:")
        print(list(self.df.columns))

        print("\nSelect a task:")
        print("1 - Regression")
        print("2 - Classification")
        print("3 - Clustering")
        print("4 - Time Series Forecasting")

        choice = input(
            "\nEnter choice [1-4] (press ENTER to auto-detect): "
        ).strip()

        if choice == "3":
            self.problem_type = "clustering"
            print("\nDetected Problem Type: CLUSTERING")
            return

        if choice == "4":
            self.problem_type = "time_series"
            self.target_column = input(
                "\nEnter numeric column to forecast: "
            ).strip()
            self.date_column = input(
                "Enter optional date/time column for ordering (ENTER to skip): "
            ).strip()
            print("\nDetected Problem Type: TIME SERIES")
            return

        self.target_column = input(
            "\nEnter target column name (press ENTER for clustering): "
        ).strip()

        if self.target_column == "":
            self.problem_type = "clustering"
            print("\nDetected Problem Type: CLUSTERING")
            return

        if self.target_column not in self.df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' does not exist."
            )

        if choice == "1":
            self.problem_type = "regression"
        elif choice == "2":
            self.problem_type = "classification"
        else:
            target = self.df[self.target_column]
            if target.dtype == object or len(target.unique()) < 10:
                self.problem_type = "classification"
            else:
                self.problem_type = "regression"

        print(f"\nDetected Problem Type: {self.problem_type.upper()}")

    def preprocess_data(self):
        if self.date_column:
            self.df[self.date_column] = pd.to_datetime(
                self.df[self.date_column], errors="coerce"
            )

        for col in self.df.columns:
            if col == self.date_column:
                continue

            if self.df[col].dtype == object:
                if self.df[col].mode().empty:
                    self.df[col].fillna("missing", inplace=True)
                else:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            else:
                self.df[col].fillna(self.df[col].mean(), inplace=True)

        for col in self.df.select_dtypes(include=[object]).columns:
            encoder = LabelEncoder()
            self.df[col] = encoder.fit_transform(self.df[col].astype(str))

        print("\nPreprocessing Completed")

    def regression_model(self):
        X = self.df.drop(self.target_column, axis=1)
        y = self.df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        print("\n========== REGRESSION RESULTS ==========")
        print("MSE:", mse)
        print("RMSE:", rmse)
        print("R2 Score:", r2)

    def classification_model(self):
        X = self.df.drop(self.target_column, axis=1)
        y = self.df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print("\n========== CLASSIFICATION RESULTS ==========")
        print("Accuracy:", accuracy)
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))

    def clustering_model(self):
        X = self.df.select_dtypes(include=[np.number]).copy()

        if X.shape[1] == 0:
            raise RuntimeError("Clustering requires at least one numeric feature.")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = KMeans(n_clusters=3, random_state=42)
        clusters = model.fit_predict(X_scaled)
        self.df["Cluster"] = clusters

        score = silhouette_score(X_scaled, clusters)

        print("\n========== CLUSTERING RESULTS ==========")
        print("Silhouette Score:", score)
        print("\nCluster Counts:")
        print(self.df["Cluster"].value_counts())

        plt.figure(figsize=(8, 5))
        if X_scaled.shape[1] == 1:
            plt.scatter(X_scaled[:, 0], np.zeros_like(X_scaled[:, 0]), c=clusters)
        else:
            plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters)

        plt.title("Cluster Visualization")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2" if X_scaled.shape[1] > 1 else "")
        plt.show()

    def time_series_model(self):
        if self.date_column:
            self.df.sort_values(by=self.date_column, inplace=True)

        data = self.df[self.target_column].astype(float)

        model = ARIMA(data, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=10)

        print("\nForecast Values:")
        print(forecast)

        plt.figure(figsize=(10, 5))
        plt.plot(data.reset_index(drop=True), label="Original")
        plt.plot(
            range(len(data), len(data) + len(forecast)),
            forecast,
            label="Forecast",
        )
        plt.legend()
        plt.title("Time Series Forecast")
        plt.show()

    def generate_explanation(self):
        print("\n========== MODEL EXPLANATION ==========")
        if self.problem_type == "regression":
            print(
                "Random Forest Regressor was selected because the target values are continuous."
            )
        elif self.problem_type == "classification":
            print(
                "Random Forest Classifier was selected because the target values are categorical."
            )
        elif self.problem_type == "clustering":
            print("K-Means was selected because no target column was provided.")
        elif self.problem_type == "time_series":
            print("ARIMA was selected for time series forecasting.")

    def run_pipeline(self):
        self.load_dataset()
        self.analyze_dataset()
        self.detect_problem_type()
        self.preprocess_data()

        if self.problem_type == "regression":
            self.regression_model()
        elif self.problem_type == "classification":
            self.classification_model()
        elif self.problem_type == "clustering":
            self.clustering_model()
        elif self.problem_type == "time_series":
            self.time_series_model()

        self.generate_explanation()


if __name__ == "__main__":
    print("\n================================")
    print("ADAPTIVE DATA SCIENCE MODEL")
    print("================================")

    file_path = input("\nEnter CSV dataset path: ")
    system = AdaptiveDSLM(file_path)
    system.run_pipeline()

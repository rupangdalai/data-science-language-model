# Adaptive Data Science Language Model (DSLM)

## Project Overview

This repository contains a simple adaptive data science pipeline that:

- Loads CSV datasets
- Detects the ML task
- Preprocesses data
- Trains and evaluates models
- Displays task-specific results
- Generates a short explanation of the selected model

Supported tasks:

- Regression
- Classification
- Clustering
- Time Series Forecasting

## Project Structure

```text
/data-science-language-model
├── app.py
├── requirements.txt
└── README.md
```

## Install Required Libraries

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Run the main script from the repository root:

```bash
python app.py
```

For a quick start, use the sample dataset at `datasets/sample.csv`.

The script prompts for the CSV dataset path and for task or target information.

## Notes

- Regression and classification auto-detection works when a target column is provided.
- Clustering runs when no target column is supplied.
- Time series forecasting can be selected explicitly and uses ARIMA.

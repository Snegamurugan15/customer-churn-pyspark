# Customer Churn Prediction Using PySpark

This project uses Apache Spark ML to predict customer churn from telecom customer behavior data. It started from the BDM Term 3 Python coursework notebook and is organized here as a clean portfolio repository with a reproducible training script.

## What It Covers

- Spark DataFrame loading and schema inference
- Telecom churn exploratory analysis in the original notebook
- Categorical feature indexing and one-hot encoding
- Vector assembly for Spark ML models
- Logistic Regression, Random Forest, and Gradient-Boosted Tree classifiers
- Accuracy, F1, weighted precision/recall, and area-under-ROC comparison

## Source Materials

- `notebooks/customer_churn_prediction_using_pyspark.ipynb` - original coursework notebook
- `docs/customer-churn-pyspark-report.docx` - project report
- `docs/project-plan.pdf` - project planning document
- `src/train_churn_pyspark.py` - cleaned runnable Spark ML pipeline

## Dataset

The project uses the public Orange Telecom churn dataset hosted by BigML:

- `churn-bigml-80.csv`
- `churn-bigml-20.csv`

The CSV files are not committed because the training script can download them.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python src/train_churn_pyspark.py --download
```

If you already have the CSV files locally:

```powershell
python src/train_churn_pyspark.py --train-csv data/raw/churn-bigml-80.csv --test-csv data/raw/churn-bigml-20.csv
```

## Notes

This is a coursework analytics project intended to demonstrate PySpark ML workflow design. It is not a production churn system, but it is structured so it can be extended with model persistence, batch scoring, and dashboard reporting.

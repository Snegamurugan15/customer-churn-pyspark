from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when


TRAIN_URL = "https://bml-data.s3.amazonaws.com/churn-bigml-80.csv"
TEST_URL = "https://bml-data.s3.amazonaws.com/churn-bigml-20.csv"


def download_if_missing(path: Path, url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {url} -> {path}")
        urlretrieve(url, path)


def build_preprocessing_pipeline(frame):
    categorical_cols = [field.name for field in frame.schema.fields if field.dataType.simpleString() == "string"]
    categorical_cols = [column for column in categorical_cols if column != "Churn"]
    numeric_cols = [field.name for field in frame.schema.fields if field.name not in categorical_cols + ["Churn"]]

    indexers = [
        StringIndexer(inputCol=column, outputCol=f"{column}_idx", handleInvalid="keep")
        for column in categorical_cols
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{column}_idx", outputCol=f"{column}_vec")
        for column in categorical_cols
    ]
    assembler = VectorAssembler(
        inputCols=numeric_cols + [f"{column}_vec" for column in categorical_cols],
        outputCol="features",
    )
    return indexers + encoders + [assembler]


def evaluate(model_name: str, predictions) -> dict:
    binary = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
    multi = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    return {
        "model": model_name,
        "area_under_roc": round(binary.evaluate(predictions), 4),
        "accuracy": round(multi.evaluate(predictions, {multi.metricName: "accuracy"}), 4),
        "f1": round(multi.evaluate(predictions, {multi.metricName: "f1"}), 4),
        "weighted_precision": round(multi.evaluate(predictions, {multi.metricName: "weightedPrecision"}), 4),
        "weighted_recall": round(multi.evaluate(predictions, {multi.metricName: "weightedRecall"}), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Spark ML churn classifiers on the Orange Telecom dataset.")
    parser.add_argument("--train-csv", default="data/raw/churn-bigml-80.csv")
    parser.add_argument("--test-csv", default="data/raw/churn-bigml-20.csv")
    parser.add_argument("--download", action="store_true", help="Download the public BigML churn CSV files if missing.")
    args = parser.parse_args()

    train_path = Path(args.train_csv)
    test_path = Path(args.test_csv)
    if args.download:
        download_if_missing(train_path, TRAIN_URL)
        download_if_missing(test_path, TEST_URL)

    spark = SparkSession.builder.appName("CustomerChurnPredictionPySpark").getOrCreate()
    train = spark.read.csv(str(train_path), header=True, inferSchema=True)
    test = spark.read.csv(str(test_path), header=True, inferSchema=True)

    train = train.withColumn("label", when(col("Churn") == True, 1.0).otherwise(0.0)).drop("Churn")
    test = test.withColumn("label", when(col("Churn") == True, 1.0).otherwise(0.0)).drop("Churn")

    preprocessing = build_preprocessing_pipeline(train)
    classifiers = {
        "logistic_regression": LogisticRegression(maxIter=30),
        "random_forest": RandomForestClassifier(numTrees=80, seed=42),
        "gradient_boosted_trees": GBTClassifier(maxIter=40, seed=42),
    }

    results = []
    for name, classifier in classifiers.items():
        pipeline = Pipeline(stages=preprocessing + [classifier])
        model = pipeline.fit(train)
        predictions = model.transform(test)
        results.append(evaluate(name, predictions))

    for row in sorted(results, key=lambda item: item["area_under_roc"], reverse=True):
        print(row)

    spark.stop()


if __name__ == "__main__":
    main()

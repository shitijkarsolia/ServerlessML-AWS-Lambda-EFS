import json
import time
import os
import pandas as pd
from pickle import load
import boto3
from sklearn.externals import joblib

start1 = time.time()
s3_client = boto3.client("s3")

preprocess = load(open("/mnt/inference/sagemaker_model/preprocessor.pkl", "rb"))
print("Preprocessor Loaded")

columns = [
    "age",
    "education",
    "major industry code",
    "class of worker",
    "num persons worked for employer",
    "capital gains",
    "capital losses",
    "dividends from stocks",
    "income",
]
class_labels = [" - 50000.", " 50000+."]

model = joblib.load("/mnt/inference/sagemaker_model/model.joblib")
print("Model Loaded!")


def lambda_handler(event, context):
    url = event["queryStringParameters"]["url"]
    _, path = url.split(":", 1)
    path = path.lstrip("/")
    bucket, path = path.split("/", 1)
    download_path = "/tmp/data.csv"
    s3_client.download_file(bucket, path, download_path)
    print("Test data downloaded from S3")

    input_data_path = os.path.join("/tmp", "data.csv")

    df = pd.read_csv(input_data_path, engine="python")
    df = pd.DataFrame(data=df, columns=columns)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.replace(class_labels, [0, 1], inplace=True)
    print("Test data shape before preprocessing: {}".format(df.shape))

    X_test = df.drop("income", axis=1)
    y_test = df["income"]

    print("Running preprocessing and feature engineering transformations")
    test_features = preprocess.transform(X_test)

    print("Test data shape after preprocessing: {}".format(test_features.shape))

    test_features_output_path = os.path.join("/tmp", "test_features.csv")
    test_labels_output_path = os.path.join("/tmp", "test_labels.csv")

    print("Saving test features to {}".format(test_features_output_path))
    pd.DataFrame(test_features).to_csv(
        test_features_output_path, header=False, index=False
    )

    X_test = pd.read_csv(test_features_output_path, header=None)

    predictions = model.predict(X_test)
    val = []
    print(predictions)
    for i in predictions:
        if i == 0:
            val.append("Less than 50K")
        else:
            val.append("Greater than 50K")
    print(val)
    end1 = time.time()
    print("Request Time Taken:{}".format(end1 - start1))

    return json.dumps({"statusCode": 200, "predictions": json.dumps(val)})

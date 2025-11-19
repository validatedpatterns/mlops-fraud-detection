import os

from kfp import compiler, dsl, kubernetes
from kfp.dsl import InputPath, OutputPath


@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-pytorch-ubi9-python-3.11-20250501-8e41d5c"
)
def get_data(
    train_data_output_path: OutputPath(), validate_data_output_path: OutputPath()
):
    import urllib.request

    print("starting download...")
    print("downloading training data")
    url = "https://raw.githubusercontent.com/rh-aiservices-bu/fraud-detection/main/data/train.csv"
    urllib.request.urlretrieve(url, train_data_output_path)
    print("train data downloaded")
    print("downloading validation data")
    url = "https://raw.githubusercontent.com/rh-aiservices-bu/fraud-detection/main/data/validate.csv"
    urllib.request.urlretrieve(url, validate_data_output_path)
    print("validation data downloaded")


@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-pytorch-ubi9-python-3.11-20250501-8e41d5c",
)
def train_model(
    train_data_input_path: InputPath(),
    validate_data_input_path: InputPath(),
    model_output_path: OutputPath(),
):
    import pickle
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils import class_weight

    torch.set_default_dtype(torch.float32)

    feature_cols = list(range(7))
    label_col = 7

    df_train = pd.read_csv(train_data_input_path)
    df_val = pd.read_csv(validate_data_input_path)

    X_train = df_train.iloc[:, feature_cols].values
    y_train = df_train.iloc[:, label_col].values.reshape(-1, 1)

    X_val = df_val.iloc[:, feature_cols].values
    y_val = df_val.iloc[:, label_col].values.reshape(-1, 1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_val = scaler.transform(X_val).astype("float32")
    y_train = y_train.astype("float32")
    y_val = y_val.astype("float32")

    Path("artifact").mkdir(parents=True, exist_ok=True)
    pickle.dump(scaler, open("artifact/scaler.pkl", "wb"))

    cw = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train.ravel()
    )
    pos_weight = torch.tensor([cw[1] / cw[0]], dtype=torch.float32)

    class FraudNetMedium(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FraudNetMedium(len(feature_cols)).to(device)

    X_train_t = torch.tensor(X_train, device=device)
    y_train_t = torch.tensor(y_train, device=device)
    X_val_t = torch.tensor(X_val, device=device)
    y_val_t = torch.tensor(y_val, device=device)

    sample_weights = (y_train_t * (pos_weight[0] - 1) + 1).flatten()
    criterion = nn.BCELoss(weight=sample_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    y_train_flat = y_train_t.flatten()

    for epoch in range(3):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t).flatten()
        loss = criterion(preds, y_train_flat)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t).flatten()
            val_loss = nn.BCELoss()(val_preds, y_val_t.flatten())
            val_acc = ((val_preds > 0.5).float() == y_val_t.flatten()).float().mean()

        print(
            f"Epoch {epoch + 1}: train loss {loss.item():.4f} | val loss {val_loss.item():.4f} | val acc {val_acc.item():.4f}"
        )

    dummy = torch.randn(1, len(feature_cols), dtype=torch.float32)
    torch.onnx.export(
        model.cpu(),
        dummy,
        model_output_path,
        input_names=["dense_input"],
        output_names=["output"],
        dynamic_axes={"dense_input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
    )


@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-pytorch-ubi9-python-3.11-20250501-8e41d5c",
)
def upload_model(input_model_path: InputPath()):
    import os

    import boto3
    import botocore

    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    endpoint_url = os.environ.get("AWS_S3_ENDPOINT")
    region_name = os.environ.get("AWS_DEFAULT_REGION")
    bucket_name = os.environ.get("AWS_S3_BUCKET")

    s3_key = os.environ.get("S3_KEY")

    session = boto3.session.Session(
        aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key
    )

    s3_resource = session.resource(
        "s3",
        config=botocore.client.Config(signature_version="s3v4"),
        endpoint_url=endpoint_url,
        region_name=region_name,
    )

    bucket = s3_resource.Bucket(bucket_name)

    print(f"Uploading {s3_key}")
    bucket.upload_file(input_model_path, s3_key)


@dsl.pipeline(name=os.path.basename(__file__).replace(".py", ""))
def pipeline():
    get_data_task = get_data()
    train_data_csv_file = get_data_task.outputs["train_data_output_path"]
    validate_data_csv_file = get_data_task.outputs["validate_data_output_path"]

    train_model_task = train_model(
        train_data_input_path=train_data_csv_file,
        validate_data_input_path=validate_data_csv_file,
    )
    onnx_file = train_model_task.outputs["model_output_path"]

    upload_model_task = upload_model(input_model_path=onnx_file)

    upload_model_task.set_env_variable(name="S3_KEY", value="models/fraud/1/model.onnx")

    kubernetes.use_secret_as_env(
        task=upload_model_task,
        secret_name="aws-connection-my-storage",
        secret_key_to_env={
            "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
            "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
            "AWS_S3_BUCKET": "AWS_S3_BUCKET",
            "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",
        },
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pipeline, package_path=__file__.replace(".py", ".yaml")
    )

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
    # 1. Imports ------------------------------------------------------
    import pickle
    import time
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils import class_weight

    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")  # keep memory in RSS

    # 2. Load and scale data -----------------------------------------
    cols = list(range(7))
    lbl = 7
    df_tr = pd.read_csv(train_data_input_path)
    df_va = pd.read_csv(validate_data_input_path)

    X_tr = df_tr.iloc[:, cols].values.astype("float32")
    y_tr = df_tr.iloc[:, lbl].values.reshape(-1, 1).astype("float32")
    X_va = df_va.iloc[:, cols].values.astype("float32")
    y_va = df_va.iloc[:, lbl].values.reshape(-1, 1).astype("float32")

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype("float32")
    X_va = scaler.transform(X_va).astype("float32")

    Path("artifact").mkdir(exist_ok=True)
    with open("artifact/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # 3. Balanced loss weight (capped) -------------------------------
    cw = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_tr), y=y_tr.ravel()
    )
    pos_w_val = min(cw[1] / cw[0], 5.0)  # cap at 5 to avoid over-bias
    pos_w = torch.tensor([pos_w_val], dtype=torch.float32)

    # 4. Network (logits out) ----------------------------------------
    class FraudNet(nn.Module):
        def __init__(self, inp: int):
            super().__init__()
            hid = 4096
            self.layers = nn.Sequential(
                nn.Linear(inp, hid),
                nn.BatchNorm1d(hid),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hid, hid),
                nn.BatchNorm1d(hid),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hid, hid),
                nn.BatchNorm1d(hid),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hid, 1),
            )

        def forward(self, x):
            return self.layers(x)

    model = FraudNet(len(cols)).to(device)

    # 5. Prepare tensors and loaders ---------------------------------
    t_tr = torch.tensor(np.hstack([X_tr, y_tr]), device=device)
    t_va = torch.tensor(np.hstack([X_va, y_va]), device=device)

    def make_loader(t: torch.Tensor, bs: int):
        ds = torch.utils.data.TensorDataset(t[:, : len(cols)], t[:, len(cols) :])
        return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True)

    loss_fn_template = lambda w: nn.BCEWithLogitsLoss(pos_weight=w)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    batch_sizes = [
        2048,
        2048,
        2048,
        2048,
        2048,
        2048,
        2048,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
        524288,
    ]

    # 6. Training loop ----------------------------------------------
    for ep, bs in enumerate(batch_sizes, 1):
        tic = time.perf_counter()
        loader = make_loader(t_tr, bs)
        val_load = make_loader(t_va, bs)

        model.train()
        for xb, yb in loader:  # one huge batch => one iteration
            samp_w = yb * (pos_w[0] - 1) + 1
            loss_f = loss_fn_template(samp_w)
            optim.zero_grad()
            logits = model(xb)
            loss = loss_f(logits, yb)
            loss.backward()
            optim.step()
            break  # process exactly one batch per epoch

        model.eval()
        with torch.no_grad():
            xv, yv = next(iter(val_load))
            v_logit = model(xv)
            v_loss = nn.BCEWithLogitsLoss()(v_logit, yv)
            v_prob = torch.sigmoid(v_logit)
            v_acc = ((v_prob > 0.5).float() == yv).float().mean()

        dt = time.perf_counter() - tic
        print(
            f"[epoch {ep}] bs={bs:<6} dur={dt:6.1f}s "
            f"| train_loss={loss.item():.4f} "
            f"| val_loss={v_loss.item():.4f} "
            f"| val_acc={v_acc.item():.4f}"
        )

    # 7. Threshold calibration --------------------------------------
    model.eval()
    with torch.no_grad():
        full_logits = model(t_va[:, : len(cols)])
        probs = torch.sigmoid(full_logits).cpu().numpy().ravel()
        labels = t_va[:, len(cols) :].cpu().numpy().ravel()

    best_f1 = 0.0
    best_thr = 0.5
    for thr in np.linspace(0.05, 0.95, 19):
        preds = (probs > thr).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        denom = 2 * tp + fp + fn
        if denom == 0:
            continue
        f1 = 2 * tp / denom
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    with open("artifact/threshold.txt", "w") as f:
        f.write(f"{best_thr}\n")
    print(f"Chosen threshold {best_thr:.2f} with F1 {best_f1:.3f}")

    # 8. Export ONNX (attach Sigmoid) -------------------------------
    print("Exporting ONNX model.")
    export_model = nn.Sequential(model.cpu(), nn.Sigmoid())
    dummy = torch.randn(1, len(cols), dtype=torch.float32)
    torch.onnx.export(
        export_model,
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

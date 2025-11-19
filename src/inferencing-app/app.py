import os

import gradio as gr
import requests
import urllib3

# Disable warnings for self‑signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Environment variables
URL = os.getenv("INFERENCE_ENDPOINT")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")

THRESHOLD = 0.95  # flag as fraud if prob >= 0.95


# ------------------------------------------------------------------
# Inference call
# ------------------------------------------------------------------
def predict(
    distance_from_home,
    distance_from_last_transaction,
    ratio_to_median_purchase_price,
    repeat_retailer,
    used_chip,
    used_pin_number,
    online_order,
):
    payload = {
        "inputs": [
            {
                "name": "dense_input",
                "shape": [1, 7],
                "datatype": "FP32",
                "data": [
                    [
                        distance_from_home,
                        distance_from_last_transaction,
                        ratio_to_median_purchase_price,
                        repeat_retailer,
                        used_chip,
                        used_pin_number,
                        online_order,
                    ]
                ],
            }
        ]
    }

    response = requests.post(
        URL, json=payload, headers={"Content-Type": "application/json"}, verify=False
    )
    prob = response.json()["outputs"][0]["data"][0]
    return "Fraud" if prob >= THRESHOLD else "Not fraud"


# ------------------------------------------------------------------
# Gradio interface
# ------------------------------------------------------------------
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Distance from Home"),
        gr.Number(label="Distance from Last Transaction"),
        gr.Number(label="Ratio to Median Purchase Price"),
        gr.Number(label="Repeat Retailer"),
        gr.Number(label="Used Chip"),
        gr.Number(label="Used PIN Number"),
        gr.Number(label="Online Order"),
    ],
    outputs="textbox",
    examples=[
        [
            15.694985541059943,
            175.98918151972342,
            0.8556228290724207,
            1,
            0,
            0,
            1,
        ],  # fraud
        [
            57.87785658389723,
            0.3111400080477545,
            1.9459399775518593,
            10,
            1,
            100,
            0,
        ],  # not fraud
    ],
    title="Predict Credit Card Fraud",
    allow_flagging="never",
)

demo.launch(server_name=GRADIO_SERVER_NAME, server_port=GRADIO_SERVER_PORT)

# MLOps Fraud Detection Pattern

## Installation Instructions

1. Create an OpenShift 4 cluster.
2. Log into the cluster using `oc login` or by exporting your `KUBECONFIG`.
3. Clone this repository and `cd` into the root folder (where this README is located).
4. Run `./pattern.sh make install` to deploy the pattern.
5. Wait for all components to finish deploying via Argo CD. You’ll know the installation is complete when the `Hub ArgoCD` instance (available via the 9-dots menu) shows all applications as **Healthy** and **Synced**.

## Notable Links

- **Inferencing App**
  In the 9-dots menu, you'll find a link to **Inferencing App**, a small Gradio-based web app for testing the model. It includes two example inputs: one non-fraudulent, and one fraudulent.
  Source: [src/inferencing-app/app.py](./src/inferencing-app/app.py)

- **Red Hat OpenShift AI**
  Also in the 9-dots menu is a link to **Red Hat OpenShift AI**. Most resources are deployed in the `fraud-detection` project (namespace). Notable components:
  - A pipeline named `fraud-detection` will have already completed an initial run after installation.
  - Under `Models → Model deployments`, you’ll find a `fraud-detection` model actively serving predictions from that initial run.

## Updating the Model

After installation, a Kubeflow pipeline runs automatically to train and deploy the initial model. The job that triggers this pipeline is located at:
[charts/fraud-detection/templates/job-create-fraud-detection-pipeline.yaml](./charts/fraud-detection/templates/job-create-fraud-detection-pipeline.yaml)

While it's possible to modify that job directly to change the initial pipeline, it’s often easier to simply upload a new pipeline version manually once installation is complete.

### Uploading a New Pipeline Version

Once the pattern is fully installed (all applications in Argo are healthy and synced), follow these steps to upload and run a new pipeline:

1. Open **Red Hat OpenShift AI** from the 9-dots menu.
2. In the left-hand navigation, go to **Data science pipelines → Pipelines**.
3. On the Pipelines page, locate the existing `fraud-detection` pipeline. Click the three-dot menu on the far right and select **Upload new version**.
4. On the upload screen:
   - Choose whether to upload the pipeline YAML directly or provide a URL.
   - For example, you can upload the [small model pipeline](./src/kubeflow-pipelines/small-model/train_upload_model.yaml) from this repository.
5. After the upload completes:
   - Click the `>` arrow next to the `fraud-detection` pipeline to expand the list of available versions.
   - Use the three-dot menu next to your new version and select **Create run**.
   - Give the run a name (e.g., `small-model` or `v1`) and click **Create run** at the bottom of the form.

You’ll be redirected to the run details page, where you can monitor its progress.

> **Note:** After the run completes, you'll need to manually restart the `fraud-detection-predictor-*` pod. This is because the predictor uses an init container to preload the model from MinIO. Restarting the pod ensures it picks up the newly trained model, assuming it was saved to the same path used in the deployment.

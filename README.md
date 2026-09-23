# An Evidence-Augmented Explainable Transformer Framework for Early Depression Risk Detection from Social Media Text

This is an academic research project/paper for early depression risk detection using evidence-augmented explainable transformers.

## Reproducibility

To ensure reproducibility, please follow these guidelines:

* **Python Version:** 3.10+
* **Package Installation:** Run `pip install -r requirements.txt`
* **Random Seeds:** All random seeds are fixed (e.g., 42) to ensure deterministic results where possible.
* **Dataset Placement:** Place the datasets in the `data/` directory. Do not commit data to version control.
* **Model Configuration:** Configurable parameters are located in `configs/experiment.yaml`.
* **GPU Usage:** The code automatically detects and uses CUDA if available.
* **Execution Order:** Execute the notebooks in the `notebooks/` directory in numerical order (01 to 08).

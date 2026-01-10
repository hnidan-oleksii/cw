import numpy as np


def calculate_metrics(results_df):
    """
    Computes MAE metrics for each model.

    Args:
        results_df (pd.DataFrame): Results dataframe.

    Returns:
        dict: MAE metrics per model.
    """
    metrics = {}

    for model in ["gru", "lgbm", "ensemble"]:
        pred_col = f"{model}_prediction"
        metrics[f"{model}_mae"] = np.mean(
            np.abs(results_df[pred_col] - results_df["actual"])
        )

    return metrics

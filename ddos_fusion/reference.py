"""Original four-model diversity calculations; research reference API."""

import pandas as pd


def cognitiveD(raw):
    """
    This function calculate the cognitive diversities table between models from raw
    """
    df = pd.DataFrame(
        {
            "Conv1d": [
                abs(raw.Conv1d.sum() - raw.Conv1d.sum()),
                abs(raw.GRU.sum() - raw.Conv1d.sum()),
                abs(raw.MLP.sum() - raw.Conv1d.sum()),
                abs(raw.ResNets.sum() - raw.Conv1d.sum()),
            ],
            "GRU": [
                abs(raw.Conv1d.sum() - raw.GRU.sum()),
                abs(raw.GRU.sum() - raw.GRU.sum()),
                abs(raw.MLP.sum() - raw.GRU.sum()),
                abs(raw.ResNets.sum() - raw.GRU.sum()),
            ],
            "MLP": [
                abs(raw.Conv1d.sum() - raw.MLP.sum()),
                abs(raw.GRU.sum() - raw.MLP.sum()),
                abs(raw.MLP.sum() - raw.MLP.sum()),
                abs(raw.ResNets.sum() - raw.MLP.sum()),
            ],
            "ResNets": [
                abs(raw.Conv1d.sum() - raw.ResNets.sum()),
                abs(raw.GRU.sum() - raw.ResNets.sum()),
                abs(raw.MLP.sum() - raw.ResNets.sum()),
                abs(raw.ResNets.sum() - raw.ResNets.sum()),
            ],
        },
        index=["Conv1d", "GRU", "MLP", "ResNets"],
    )
    df = df[["Conv1d", "GRU", "MLP", "ResNets"]]
    return df


def diversityS(raw):
    """
    This function calculate the diversity strength of four single model
    input raw should contain four models' normalized sorce
    """
    strength_A = (
        abs(raw.ResNets.sum() - raw.Conv1d.sum())
        + abs(raw.GRU.sum() - raw.Conv1d.sum())
        + abs(raw.MLP.sum() - raw.Conv1d.sum())
    ) / 3
    strength_B = (
        abs(raw.ResNets.sum() - raw.GRU.sum())
        + abs(raw.MLP.sum() - raw.GRU.sum())
        + abs(raw.Conv1d.sum() - raw.GRU.sum())
    ) / 3
    strength_C = (
        abs(raw.ResNets.sum() - raw.MLP.sum())
        + abs(raw.GRU.sum() - raw.MLP.sum())
        + abs(raw.Conv1d.sum() - raw.MLP.sum())
    ) / 3
    strength_D = (
        abs(raw.MLP.sum() - raw.ResNets.sum())
        + abs(raw.GRU.sum() - raw.ResNets.sum())
        + abs(raw.Conv1d.sum() - raw.ResNets.sum())
    ) / 3
    return strength_A, strength_B, strength_C, strength_D

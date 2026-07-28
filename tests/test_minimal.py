import torch

from data.datasets import build_dataset
from models import ContinuousARModel, DirectForecastModel, TextARModel
from tokenizers.numeric import FixedWidthTextNumberTokenizer


def test_text_roundtrip():
    tokenizer = FixedWidthTextNumberTokenizer()
    values = torch.tensor([[-3.21, -0.01, 0.0, 1.23, 4.56]])
    assert torch.equal(tokenizer.decode(tokenizer.encode(values)), values)


def test_all_models_share_forecast_contract():
    context_len, horizon = 12, 4
    context = torch.randn(2, context_len)
    future = torch.randn(2, horizon)
    models = [
        DirectForecastModel(context_len, horizon, d_model=16, n_heads=4, n_layers=1, dropout=0.0),
        ContinuousARModel(context_len, horizon, d_model=16, n_heads=4, n_layers=1, dropout=0.0),
        TextARModel(context_len, horizon, d_model=16, n_heads=4, n_layers=1, dropout=0.0),
    ]
    for model in models:
        loss = model.training_loss(context, future)
        prediction = model.predict(context)
        assert loss.ndim == 0 and torch.isfinite(loss)
        assert prediction.shape == future.shape
        assert torch.isfinite(prediction).all()


def test_synthetic_windows_are_deterministic_and_leakage_safe():
    first = build_dataset(
        "synthetic_ar", context_len=16, horizon=4, seed=11,
        total_length=400, max_train_windows=8, max_val_windows=8, max_test_windows=8,
    )
    second = build_dataset(
        "synthetic_ar", context_len=16, horizon=4, seed=11,
        total_length=400, max_train_windows=8, max_val_windows=8, max_test_windows=8,
    )
    for first_split, second_split in zip(first[:3], second[:3]):
        assert torch.equal(first_split[0][0], second_split[0][0])
        assert torch.equal(first_split[0][1], second_split[0][1])

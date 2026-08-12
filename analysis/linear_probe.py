"""Shared linear probe / softmax regression utilities (numpy only)."""

from __future__ import annotations

import numpy as np


def softmax_regression(
    X: np.ndarray, Y: np.ndarray, n_iter: int = 4000, lr: float = 0.5, l2: float = 1e-3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multinomial logistic regression in pure numpy; returns (W, mean, std)."""
    num, dim = X.shape
    num_classes = int(Y.max()) + 1
    weights = np.zeros((dim, num_classes))
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-9
    Xs = (X - mean) / std
    onehot = np.eye(num_classes)[Y]
    for _ in range(n_iter):
        scores = Xs @ weights
        scores -= scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        prob = exp / exp.sum(axis=1, keepdims=True)
        grad = Xs.T @ (prob - onehot) / num + l2 * weights
        weights -= lr * grad
    return weights, mean, std


def softmax_predict(
    X: np.ndarray, weights: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probabilities, argmax class)."""
    Xs = (X - mean) / std
    scores = Xs @ weights
    scores -= scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    prob = exp / exp.sum(axis=1, keepdims=True)
    return prob, prob.argmax(axis=1)


def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(pred == target))

"""Recurrent, convolutional, attention and basis-expansion sequence forecasters."""

from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQUENCE_LENGTH = 24
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
PATIENCE = 10
GRADIENT_CLIP = 1.0


class VanillaLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :])).squeeze(-1)


class CNNLSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        filters: int = 64,
        hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv = nn.Conv1d(n_features, filters, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(filters, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = torch.relu(self.conv(x.permute(0, 2, 1)))
        c = self.pool(c).permute(0, 2, 1)
        out, _ = self.lstm(c)
        return self.head(self.dropout(out[:, -1, :])).squeeze(-1)


class AdditiveAttention(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.project = nn.Linear(hidden, hidden)
        self.score = nn.Linear(hidden, 1, bias=False)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(torch.tanh(self.project(states))), dim=1)
        return (weights * states).sum(dim=1)


class AttentionLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.attention = AdditiveAttention(hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states, _ = self.lstm(x)
        return self.head(self.dropout(self.attention(states))).squeeze(-1)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.projection = nn.Linear(n_features, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.projection(x))
        return self.head(encoded[:, -1, :]).squeeze(-1)


class NBEATSBlock(nn.Module):
    def __init__(self, input_size: int, hidden: int = 128, n_layers: int = 4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_size, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        self.stack = nn.Sequential(*layers)
        self.backcast = nn.Linear(hidden, input_size)
        self.forecast = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stack(x)
        return self.backcast(h), self.forecast(h).squeeze(-1)


class NBEATS(nn.Module):
    def __init__(
        self,
        n_features: int,
        seq_len: int = SEQUENCE_LENGTH,
        n_stacks: int = 2,
        blocks_per_stack: int = 3,
        hidden: int = 128,
    ):
        super().__init__()
        self.projection = nn.Linear(n_features, 1)
        self.blocks = nn.ModuleList(
            [NBEATSBlock(seq_len, hidden) for _ in range(n_stacks * blocks_per_stack)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.projection(x).squeeze(-1)
        forecast = torch.zeros(x.shape[0], device=x.device)
        for block in self.blocks:
            backcast, partial = block(residual)
            residual = residual - backcast
            forecast = forecast + partial
        return forecast


class EarlyStopping:
    def __init__(self, patience: int = PATIENCE, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_state: dict | None = None
        self.waited = 0

    def step(self, loss: float, model: nn.Module) -> bool:
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.waited = 0
        else:
            self.waited += 1

        if self.waited >= self.patience:
            if self.best_state is not None:
                model.load_state_dict(self.best_state)
            return True
        return False


def sequences(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_windows = len(X) - seq_len + 1
    if n_windows <= 0:
        raise ValueError(f"need more than {seq_len - 1} rows, got {len(X)}")

    rows = np.arange(n_windows)[:, None]
    columns = np.arange(seq_len)[None, :]
    return (
        X[rows + columns].astype(np.float32),
        y[seq_len - 1 :].astype(np.float32),
        w[seq_len - 1 :].astype(np.float32),
    )


def train(model: nn.Module, loader: DataLoader, epochs: int = EPOCHS) -> nn.Module:
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    stopper = EarlyStopping()

    for _ in range(epochs):
        model.train()
        total = 0.0
        for X_batch, y_batch, w_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            w_batch = w_batch.to(DEVICE)

            optimizer.zero_grad()
            loss = (w_batch * (model(X_batch) - y_batch) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            total += loss.item()

        if stopper.step(total / max(len(loader), 1), model):
            break

    return model


def predictor(model: nn.Module, context: np.ndarray, seq_len: int) -> Callable:
    def predict_fn(X_test: np.ndarray, timestamps=None) -> np.ndarray:
        padded = np.concatenate([context, X_test], axis=0).astype(np.float32)
        rows = np.arange(len(X_test))[:, None]
        columns = np.arange(seq_len)[None, :]
        windows = torch.tensor(padded[rows + columns], dtype=torch.float32).to(DEVICE)

        model.eval()
        with torch.no_grad():
            predictions = model(windows).cpu().numpy().astype(np.float64)
        return np.clip(predictions, 0.0, 1.0)

    return predict_fn


def _sequence_model(
    build: Callable[[int], nn.Module],
    seq_len: int,
    epochs: int,
    carry_state: bool = False,
) -> Callable:
    state: list = [None]

    def model_fn(X_train, y_train, w_train):
        X_seq, y_seq, w_seq = sequences(X_train, y_train, w_train, seq_len)
        loader = DataLoader(
            TensorDataset(
                torch.tensor(X_seq), torch.tensor(y_seq), torch.tensor(w_seq)
            ),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        model = build(X_train.shape[1])
        if carry_state and state[0] is not None:
            try:
                model.load_state_dict(state[0])
            except RuntimeError:
                pass

        model = train(model, loader, epochs=epochs)
        if carry_state:
            state[0] = copy.deepcopy(model.state_dict())

        return predictor(model, X_train[-(seq_len - 1) :], seq_len)

    return model_fn


def vanilla_lstm(hidden: int = 64, seq_len: int = SEQUENCE_LENGTH, epochs: int = EPOCHS):
    return _sequence_model(lambda n: VanillaLSTM(n, hidden), seq_len, epochs)


def cnn_lstm(
    filters: int = 64,
    hidden: int = 64,
    seq_len: int = SEQUENCE_LENGTH,
    epochs: int = EPOCHS,
):
    return _sequence_model(lambda n: CNNLSTM(n, filters, hidden), seq_len, epochs)


def attention_lstm(hidden: int = 64, seq_len: int = SEQUENCE_LENGTH, epochs: int = EPOCHS):
    return _sequence_model(lambda n: AttentionLSTM(n, hidden), seq_len, epochs)


def transformer(d_model: int = 64, seq_len: int = SEQUENCE_LENGTH, epochs: int = EPOCHS):
    return _sequence_model(lambda n: TransformerEncoder(n, d_model), seq_len, epochs)


def transformer_warm(d_model: int = 64, seq_len: int = SEQUENCE_LENGTH, epochs: int = EPOCHS):
    return _sequence_model(
        lambda n: TransformerEncoder(n, d_model), seq_len, epochs, carry_state=True
    )


def nbeats(
    seq_len: int = SEQUENCE_LENGTH,
    n_stacks: int = 2,
    blocks_per_stack: int = 3,
    hidden: int = 128,
    epochs: int = EPOCHS,
):
    return _sequence_model(
        lambda n: NBEATS(n, seq_len, n_stacks, blocks_per_stack, hidden), seq_len, epochs
    )


FACTORIES: dict[str, Callable[..., Callable]] = {
    "vanilla_lstm": vanilla_lstm,
    "cnn_lstm": cnn_lstm,
    "attention_lstm": attention_lstm,
    "transformer": transformer,
    "transformer_warm": transformer_warm,
    "nbeats": nbeats,
}


def get_model(name: str, **kwargs: Any) -> Callable:
    if name not in FACTORIES:
        raise KeyError(f"unknown sequence model '{name}'")
    return FACTORIES[name](**kwargs)

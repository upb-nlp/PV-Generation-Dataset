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
DEFAULT_SEED = 42


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
            dim_feedforward=2 * d_model,
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


def seed_everything(seed: int) -> torch.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def sequences(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding windows ending at each row; windows touching a zero-weight (missing) row are dropped."""
    n_windows = len(X) - seq_len + 1
    if n_windows <= 0:
        raise ValueError(f"need more than {seq_len - 1} rows, got {len(X)}")

    rows = np.arange(n_windows)[:, None]
    columns = np.arange(seq_len)[None, :]
    present = np.asarray(w) > 0
    complete = present[rows + columns].all(axis=1)
    if not complete.any():
        raise ValueError("no complete input window in the training data")

    index = (rows + columns)[complete]
    return (
        X[index].astype(np.float32),
        y[seq_len - 1 :][complete].astype(np.float32),
        w[seq_len - 1 :][complete].astype(np.float32),
    )


def windows(context: np.ndarray, X: np.ndarray, seq_len: int) -> torch.Tensor:
    padded = np.concatenate([context, X], axis=0).astype(np.float32)
    rows = np.arange(len(X))[:, None]
    columns = np.arange(seq_len)[None, :]
    return torch.tensor(padded[rows + columns], dtype=torch.float32).to(DEVICE)


def _predict(model: nn.Module, inputs: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(inputs).cpu().numpy().astype(np.float64)


def train(
    model: nn.Module,
    loader: DataLoader,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    validation: Callable[[nn.Module], float] | None = None,
    patience: int = PATIENCE,
) -> tuple[nn.Module, int]:
    """Fixed-length training, or early stopping on ``validation`` loss when one is given.

    Returns the model (best validation state when validating) and the epoch count it reflects.
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss, best_state, best_epoch, waited = float("inf"), None, epochs, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for X_batch, y_batch, w_batch in loader:
            X_batch, y_batch, w_batch = X_batch.to(DEVICE), y_batch.to(DEVICE), w_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = (w_batch * (model(X_batch) - y_batch) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()

        if validation is None:
            continue
        current = validation(model)
        if current < best_loss - 1e-7:
            best_loss, best_state, best_epoch, waited = current, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            waited += 1
            if waited >= patience:
                break

    if validation is not None and best_state is not None:
        model.load_state_dict(best_state)
        return model, best_epoch
    return model, epochs


def loader_for(X_seq, y_seq, w_seq, generator: torch.Generator) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.tensor(X_seq), torch.tensor(y_seq), torch.tensor(w_seq)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )


def _sequence_model(
    build: Callable[[int], nn.Module],
    seq_len: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    carry_state: bool = False,
) -> Callable:
    state: list = [None]

    def model_fn(X_train, y_train, w_train):
        generator = seed_everything(seed)
        X_seq, y_seq, w_seq = sequences(X_train, y_train, w_train, seq_len)
        model = build(X_train.shape[1])
        if carry_state and state[0] is not None:
            try:
                model.load_state_dict(state[0])
            except RuntimeError:
                pass

        model, _ = train(model, loader_for(X_seq, y_seq, w_seq, generator), epochs, learning_rate)
        if carry_state:
            state[0] = copy.deepcopy(model.state_dict())

        context = X_train[-(seq_len - 1) :]

        def predict_fn(X_test: np.ndarray, timestamps=None) -> np.ndarray:
            return np.clip(_predict(model, windows(context, X_test, seq_len)), 0.0, 1.0)

        return predict_fn

    return model_fn


ARCHITECTURES: dict[str, Callable[[int, int], nn.Module]] = {
    "vanilla_lstm": lambda n, width: VanillaLSTM(n, width),
    "cnn_lstm": lambda n, width: CNNLSTM(n, width, width),
    "attention_lstm": lambda n, width: AttentionLSTM(n, width),
    "transformer": lambda n, width: TransformerEncoder(n, width),
    "nbeats": lambda n, width: NBEATS(n, SEQUENCE_LENGTH, 2, 3, width),
}
DEFAULT_WIDTH = {
    "vanilla_lstm": 64,
    "cnn_lstm": 64,
    "attention_lstm": 64,
    "transformer": 64,
    "nbeats": 128,
}


def build_model(name: str, n_features: int, width: int | None = None) -> nn.Module:
    base = name.removesuffix("_warm")
    return ARCHITECTURES[base](n_features, width or DEFAULT_WIDTH[base])


def get_model(
    name: str,
    width: int | None = None,
    learning_rate: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    seed: int = DEFAULT_SEED,
    seq_len: int = SEQUENCE_LENGTH,
    **unused: Any,
) -> Callable:
    base = name.removesuffix("_warm")
    if base not in ARCHITECTURES:
        raise KeyError(f"unknown sequence model '{name}'")
    return _sequence_model(
        lambda n: build_model(base, n, width),
        seq_len,
        epochs,
        learning_rate,
        seed,
        carry_state=name.endswith("_warm"),
    )

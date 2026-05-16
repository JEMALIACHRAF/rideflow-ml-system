"""
Neural network models: LSTM (bidirectional) and TabNet.
LSTM captures long-range temporal dependencies.
TabNet uses attention for feature selection on tabular data.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
from models.base_model import BaseModel


# ─── LSTM ─────────────────────────────────────────────────────────────────────

class BiLSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)
        return self.fc(context).squeeze(-1)


class LSTMDemandModel(BaseModel):
    """
    Bidirectional LSTM with attention pooling.
    Expects sequences of shape (batch, seq_len, n_features).
    """
    def __init__(self, seq_len: int = 24, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2,
                 lr: float = 1e-3, epochs: int = 50, batch_size: int = 256):
        params = dict(seq_len=seq_len, hidden_size=hidden_size,
                      num_layers=num_layers, dropout=dropout,
                      lr=lr, epochs=epochs, batch_size=batch_size)
        super().__init__("LSTM", params)
        self.seq_len = seq_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scaler_mean_: float = 0.0
        self.scaler_std_: float = 1.0

    def _build_model(self) -> BiLSTMNet:
        return BiLSTMNet(
            input_size=len(self.feature_names_),
            hidden_size=self.params["hidden_size"],
            num_layers=self.params["num_layers"],
            dropout=self.params["dropout"],
        ).to(self.device)

    def _make_sequences(self, X: np.ndarray, y: np.ndarray | None = None):
        """Convert tabular data to overlapping sequences."""
        X_seq, y_seq = [], []
        for i in range(self.seq_len, len(X)):
            X_seq.append(X[i - self.seq_len:i])
            if y is not None:
                y_seq.append(y[i])
        X_seq = np.array(X_seq, dtype=np.float32)
        if y is not None:
            return X_seq, np.array(y_seq, dtype=np.float32)
        return X_seq

    def fit(self, X: pd.DataFrame, y: pd.Series,
            eval_set: tuple | None = None) -> "LSTMDemandModel":
        self.feature_names_ = list(X.columns)
        # Normalize
        X_arr = X.values.astype(np.float32)
        self.scaler_mean_ = X_arr.mean(axis=0)
        self.scaler_std_  = X_arr.std(axis=0) + 1e-8
        X_norm = (X_arr - self.scaler_mean_) / self.scaler_std_

        X_seq, y_seq = self._make_sequences(X_norm, y.values)
        dataset = TensorDataset(torch.from_numpy(X_seq), torch.from_numpy(y_seq))
        loader  = DataLoader(dataset, batch_size=self.params["batch_size"], shuffle=True)

        self.model = self._build_model()
        optimizer  = torch.optim.AdamW(self.model.parameters(), lr=self.params["lr"])
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.params["epochs"]
        )
        criterion  = nn.HuberLoss()

        best_loss = float("inf")
        for epoch in range(self.params["epochs"]):
            self.model.train()
            epoch_loss = 0.0
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(X_b), y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            avg_loss = epoch_loss / len(loader)
            if avg_loss < best_loss:
                best_loss = avg_loss
            if (epoch + 1) % 10 == 0:
                logger.debug(f"LSTM epoch {epoch+1}/{self.params['epochs']} loss={avg_loss:.4f}")

        self.is_fitted = True
        logger.info(f"LSTM trained — best loss: {best_loss:.4f}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_arr  = X[self.feature_names_].values.astype(np.float32)
        X_norm = (X_arr - self.scaler_mean_) / self.scaler_std_
        X_seq  = self._make_sequences(X_norm)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_seq), 512):
                batch = torch.from_numpy(X_seq[i:i+512]).to(self.device)
                preds.append(self.model(batch).cpu().numpy())
        # Pad the first seq_len predictions with zeros
        result = np.concatenate(preds)
        result = np.concatenate([np.zeros(self.seq_len), result])
        return np.maximum(result, 0)


# ─── TabNet ───────────────────────────────────────────────────────────────────

class TabNetAttention(nn.Module):
    """
    Simplified TabNet — sequential attention steps over features.
    Each step selects a subset of features; outputs are summed.
    """
    def __init__(self, input_dim: int, n_steps: int = 3,
                 n_a: int = 64, n_d: int = 64, gamma: float = 1.3):
        super().__init__()
        self.input_dim = input_dim
        self.n_steps   = n_steps
        self.gamma     = gamma
        self.bn        = nn.BatchNorm1d(input_dim)
        self.initial_bn = nn.BatchNorm1d(input_dim)

        self.attention_transformers = nn.ModuleList([
            nn.Sequential(nn.Linear(n_d, input_dim), nn.Softmax(dim=-1))
            for _ in range(n_steps)
        ])
        self.feature_transformers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, n_d * 2),
                nn.GLU(dim=-1),
                nn.BatchNorm1d(n_d),
            )
            for _ in range(n_steps)
        ])
        self.final_fc = nn.Linear(n_d, 1)

    def forward(self, x: torch.Tensor):
        x = self.initial_bn(x)
        prior_scale = torch.ones_like(x)
        h = torch.zeros(x.size(0), self.attention_transformers[0][0].out_features // 2
                        if hasattr(self.attention_transformers[0][0], "out_features")
                        else 64).to(x.device)
        total_entropy = 0.0
        step_outputs = []

        for step in range(self.n_steps):
            # Compute attention mask
            mask = self.attention_transformers[step](h)
            mask = mask * prior_scale
            mask = mask / (mask.sum(dim=-1, keepdim=True) + 1e-8)
            prior_scale = prior_scale * (self.gamma - mask)
            total_entropy += (-mask * torch.log(mask + 1e-8)).sum(dim=-1).mean()

            masked_x = mask * x
            h = self.feature_transformers[step](masked_x)
            step_outputs.append(h)

        out = torch.stack(step_outputs, dim=0).sum(dim=0)
        return self.final_fc(out).squeeze(-1), total_entropy


class TabNetDemandModel(BaseModel):
    def __init__(self, n_steps: int = 3, n_a: int = 64, n_d: int = 64,
                 lr: float = 2e-3, epochs: int = 100, batch_size: int = 1024,
                 lambda_sparse: float = 1e-4):
        params = dict(n_steps=n_steps, n_a=n_a, n_d=n_d, lr=lr,
                      epochs=epochs, batch_size=batch_size, lambda_sparse=lambda_sparse)
        super().__init__("TabNet", params)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _build_model(self) -> TabNetAttention:
        return TabNetAttention(
            input_dim=len(self.feature_names_),
            n_steps=self.params["n_steps"],
            n_a=self.params["n_a"],
            n_d=self.params["n_d"],
        ).to(self.device)

    def fit(self, X: pd.DataFrame, y: pd.Series,
            eval_set: tuple | None = None) -> "TabNetDemandModel":
        self.feature_names_ = list(X.columns)
        X_arr = torch.tensor(X.values.astype(np.float32))
        y_arr = torch.tensor(y.values.astype(np.float32))
        dataset = TensorDataset(X_arr, y_arr)
        loader  = DataLoader(dataset, batch_size=self.params["batch_size"], shuffle=True)

        self.model = self._build_model()
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"])

        for epoch in range(self.params["epochs"]):
            self.model.train()
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                pred, sparse_loss = self.model(X_b)
                loss = nn.HuberLoss()(pred, y_b) + self.params["lambda_sparse"] * sparse_loss
                loss.backward()
                optimizer.step()

        self.is_fitted = True
        logger.info("TabNet trained")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_t = torch.tensor(X[self.feature_names_].values.astype(np.float32)).to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds, _ = self.model(X_t)
        return np.maximum(preds.cpu().numpy(), 0)

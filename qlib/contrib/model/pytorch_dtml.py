# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from typing import Text, Union
import copy

from ...data.dataset.weight import Reweighter

from ...data.dataset import Dataset
from ...utils import get_or_create_path
from ...log import get_module_logger

import torch
import torch.nn as nn
import torch.optim as optim

from .pytorch_utils import count_parameters
from ...model.base import Model
from ...data.dataset import TSDatasetH
from ...data.dataset.handler import DataHandlerLP
from torch.utils.data import DataLoader
from ...model.utils import ConcatDataset
from typing import Callable, Union, List, Tuple, Dict, Text, Optional
from ...data.dataset import TSDataSampler
from datetime import date


def filter_group_data(group_df: pd.DataFrame, data_arr: np.ndarray) -> Tuple[List[str], np.ndarray]:
    group_data_list = []
    instrument_list = []
    nan_idx = len(data_arr) - 1
    for instrument, instrument_index in group_df.items():
        indices = np.nan_to_num(instrument_index.astype(
            np.float64), nan=nan_idx).astype(int)
        data = data_arr[indices[0]: indices[-1] + 1]
        instrument_list.append(instrument)
        group_data_list.append(data)
    group_data = np.stack(group_data_list)
    return instrument_list, group_data


def create_data(dl_train: TSDataSampler, step_len: int) -> Dict[date, Tuple[List[str], np.ndarray]]:
    print(len(dl_train.idx_df.index))
    train_dict = {}
    for idx in range(len(dl_train.idx_df.index)):
        if idx < step_len:
            continue
        today = dl_train.idx_df.index[idx].date()
        dates = dl_train.idx_df.index[max(idx - step_len + 1, 0): idx + 1]
        group_df = dl_train.idx_df.loc[dates]
        group_df = group_df.dropna(axis=1)
        instruments, group_data = filter_group_data(
            group_df, dl_train.data_arr)
        train_dict[today] = (instruments, group_data)
        print(
            f"{type(today)}, {today}, instruments={len(instruments)}, data = {group_data.shape}")
    return train_dict


class DTML(Model):

    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 64,
        num_layers: int = 1,
        n_heads: int = 8,
        beta=0.1,
        drop_rate=0.15,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=256,
        early_stop=20,
        loss="mse",
        optimizer="adam",
        reg=1e-3,
        GPU=0,
        seed=None,
    ):
        # Set logger.
        self.logger = get_module_logger("DTML")
        self.logger.info("DTML pytorch version...")

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.loss = loss
        self.device = torch.device(
            "cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu"
        )

        self.model = DTMLModel(
            input_size, hidden_size, num_layers, n_heads, beta, drop_rate
        )

        self.logger.info("model:\n{:}".format(self.model))
        self.logger.info("model size: {:.4f} MB".format(
            count_parameters(self.model)))

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.reg
            )
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(
                self.model.parameters(), lr=self.lr, weight_decay=self.reg
            )
        else:
            raise NotImplementedError(
                "optimizer {} is not supported!".format(optimizer)
            )

        self.fitted = False
        self.model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label, weight):
        loss = weight * (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label, weight=None):
        mask = ~torch.isnan(label)

        if weight is None:
            weight = torch.ones_like(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask], weight[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])
        elif self.metric == "mse":
            mask = ~torch.isnan(label)
            weight = torch.ones_like(label)
            return -self.mse(pred[mask], label[mask], weight[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, data_loader):
        self.model.train()

        for data, weight in data_loader:
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label, weight.to(self.device))

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):
        self.model.eval()

        scores = []
        losses = []

        for data, weight in data_loader:
            feature = data[:, :, 0:-1].to(self.device)
            # feature[torch.isnan(feature)] = 0
            label = data[:, -1, -1].to(self.device)

            with torch.no_grad():
                pred = self.model(feature.float())
                loss = self.loss_fn(pred, label, weight.to(self.device))
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset: TSDatasetH,
        evals_result=dict(),
        save_path=None,
        reweighter=None,
    ):
        dl_train = dataset.prepare(
            "train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare(
            "valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError(
                "Empty data from dataset, please check your dataset config.")

        # process nan brought by dataloader
        dl_train.config(fillna_type="ffill+bfill")
        # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")

        if reweighter is None:
            wl_train = np.ones(len(dl_train))
            wl_valid = np.ones(len(dl_valid))
        elif isinstance(reweighter, Reweighter):
            wl_train = reweighter.reweight(dl_train)
            wl_valid = reweighter.reweight(dl_valid)
        else:
            raise ValueError("Unsupported reweighter type.")

        train_loader = DataLoader(
            ConcatDataset(dl_train, wl_train),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=20,
            drop_last=True,
        )
        valid_loader = DataLoader(
            ConcatDataset(dl_valid, wl_valid),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=20,
            drop_last=True,
        )

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" %
                             (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: TSDatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare(
            segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        test_loader = DataLoader(
            dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        self.model.eval()
        preds = []

        for data in test_loader:
            feature = data[:, :, 0:-1].to(self.device)

            with torch.no_grad():
                pred = self.model(feature.float()).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=dl_test.get_index())


class TimeAxisAttention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=False
        )
        self.lnorm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.tensor, rt_attn=False):
        # x: (D, W, L)
        o, (h, _) = self.lstm(x)  # o: (D, W, H) / h: (1, D, H)
        score = torch.bmm(o, h.permute(1, 2, 0))  # (D, W, H) x (D, H, 1)
        tx_attn = torch.softmax(score, 1).squeeze(-1)  # (D, W)
        context = torch.bmm(tx_attn.unsqueeze(1), o).squeeze(
            1)  # (D, 1, W) x (D, W, H)
        normed_context = self.lnorm(context)
        if rt_attn:
            return normed_context, tx_attn
        else:
            return normed_context, None


class DataAxisAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, drop_rate=0.1):
        super().__init__()
        self.multi_attn = nn.MultiheadAttention(
            hidden_size, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.lnorm1 = nn.LayerNorm(hidden_size)
        self.lnorm2 = nn.LayerNorm(hidden_size)
        self.drop_out = nn.Dropout(drop_rate)

    def forward(self, hm: torch.tensor, rt_attn=False):
        # Forward Multi-head Attention
        residual = hm
        # hm_hat: (D, H), dx_attn: (D, D)
        hm_hat, dx_attn = self.multi_attn(hm, hm, hm)
        hm_hat = self.lnorm1(residual + self.drop_out(hm_hat))

        # Forward FFN
        residual = hm_hat
        # hp: (D, H)
        hp = torch.tanh(hm + hm_hat + self.mlp(hm + hm_hat))
        hp = self.lnorm2(residual + self.drop_out(hp))

        if rt_attn:
            return hp, dx_attn
        else:
            return hp, None


class DTMLModel(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, n_heads, beta=0.1, drop_rate=0.1
    ):
        super().__init__()
        self.beta = beta
        self.txattention = TimeAxisAttention(
            input_size, hidden_size, num_layers)
        self.dxattention = DataAxisAttention(hidden_size, n_heads, drop_rate)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, stocks, index, rt_attn=False):
        # stocks: (W, D, L) for a single time stamp
        # index: (W, 1, L) for a single time stamp
        # W: length of observations
        # D: number of stocks
        # L: number of features

        # Time-Axis Attention
        # c_stocks: (D, H) / tx_attn_stocks: (D, W)
        c_stocks, tx_attn_stocks = self.txattention(
            stocks.transpose(1, 0), rt_attn=rt_attn
        )
        # c_index: (1, H) / tx_attn_index: (1, W)
        c_index, tx_attn_index = self.txattention(
            index.transpose(1, 0), rt_attn=rt_attn
        )

        # Context Aggregation
        # Multi-level Context
        # hm: (D, H)
        hm = c_stocks + self.beta * c_index
        # The Effect of Global Contexts
        # effect: (D, D)
        effect = (
            c_stocks.mm(c_stocks.transpose(0, 1))
            + self.beta * c_index.mm(c_stocks.transpose(1, 0))
            + self.beta**2 * torch.mm(c_index, c_index.transpose(0, 1))
        )

        # Data-Axis Attention
        # hp: (D, H) / dx_attn: (D, D)
        hp, dx_attn_stocks = self.dxattention(hm, rt_attn=rt_attn)
        # output: (D, 1)
        output = self.linear(hp)

        return {
            "output": output,
            "tx_attn_stocks": tx_attn_stocks,
            "tx_attn_index": tx_attn_index,
            "dx_attn_stocks": dx_attn_stocks,
            "effect": effect,
        }

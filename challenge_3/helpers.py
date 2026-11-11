from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import time
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
)
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from sklearn.model_selection import train_test_split


ZO = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=3)

class Model:
    def __init__(self, qc: QuantumCircuit, x_var: ParameterVector, th_var: ParameterVector):
        self.qc = qc
        self.x_var = x_var
        self.th_var = th_var
        self.name = 'Model'
        
    def add_name(self, name: str) -> Self:
        self.name = name
        return self
        
    def measure(self, x: np.ndarray, th: np.ndarray) -> float:
        assert x.shape == (len(self.x_var),), f'Invalid shape {x.shape} for x (should be ({len(self.x_var)},))'
        assert th.shape == (len(self.th_var),), f'Invalid shape {th.shape} for th (should be ({len(self.x_var)},))'
        
        bound = self.qc.assign_parameters({**dict(zip(self.x_var, x)), **dict(zip(self.th_var, th))})
        job = StatevectorEstimator().run([(bound, ZO)])
        return float(job.result()[0].data['evs'])
    
    def measure_batch(self, x: np.ndarray, th: np.ndarray) -> np.ndarray:
        if th.ndim == 1:
            return self._measure_z_batch(x, th)
        
        return self._measure_z_batch_th_batch(x, th)
    
    def _measure_z_batch(self, X, th) -> np.ndarray:
        assert X.ndim == 2
        assert X.shape[1] == len(self.x_var)
        assert len(th) == len(self.th_var)

        x_parameter_values = {
            param: X[:, i]
            for i, param in enumerate(self.x_var)
        }
        
        bound = self.qc.assign_parameters(dict(zip(self.th_var, th)))

        job = StatevectorEstimator().run([
            (bound, ZO, x_parameter_values)
        ])
        
        assert job.result()[0].data['evs'].size == X.shape[0], (
            f"Expected {X.shape[0]} outputs, got {job.result()[0].data['evs'].shape}"
        )

        return np.asarray(job.result()[0].data['evs'])
    
    def _measure_z_batch_th_batch(self, X, TH) -> np.ndarray:
        assert X.ndim == 2 and TH.ndim == 2
        assert X.shape[1] == len(self.x_var)
        assert TH.shape[1] == len(self.th_var)
        assert X.shape[0] == TH.shape[0]

        params = np.concatenate([self.x_var, self.th_var])
        values = np.concatenate([X, TH], axis=1)
        
        parameter_values = {
            param: values[:, i]
            for i, param in enumerate(params)
        }

        job = StatevectorEstimator().run([
            (self.qc, ZO, parameter_values)
        ])
        
        assert job.result()[0].data['evs'].size == X.shape[0], (
            f"Expected {X.shape[0]} outputs, got {job.result()[0].data['evs'].shape}"
        )

        return np.asarray(job.result()[0].data['evs'])


@dataclass
class Metrics:
    mse: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    
    mean: float
    std: float
    min: float
    max: float
    
    gradient_norm: float
    
    @classmethod
    def calc(cls, y_true: np.ndarray, outputs: np.ndarray, grad_L: np.ndarray) -> Self:
        predictions = np.where(outputs >= 0, 1, -1)
                
        mse = mean_squared_error(y_true, outputs)
        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
                
        return cls(
            float(mse), float(accuracy), float(precision), float(recall), float(f1),
            outputs.mean(), outputs.std(), outputs.min(), outputs.max(),
            np.linalg.norm(grad_L)
            )


@dataclass
class HistoryEntry:
    epoch: int
    metrics_train: Metrics
    metrics_val: Metrics
    train_time: float
    params: dict
    
    def __str__(self) -> str:
        val_str = (
            f"Val> | MSE: {self.metrics_val.mse:.4f} | "
            f"Accuracy: {self.metrics_val.accuracy:.2%} | "
            f"Precision: {self.metrics_val.precision:.2%} | "
            f"Recall: {self.metrics_val.recall:.2%} | "
            f"F1: {self.metrics_val.f1:.4f} |\n"
        )
        
        train_str = (
            f"Train> | MSE: {self.metrics_train.mse:.4f} | "
            f"Accuracy: {self.metrics_train.accuracy:.2%} | "
            f"Precision: {self.metrics_train.precision:.2%} | "
            f"Recall: {self.metrics_train.recall:.2%} | "
            f"F1: {self.metrics_train.f1:.4f} |\n"
        )
        
        val_stats_str = (
            f"Val Preds Stats> | Mean: {self.metrics_val.mean:.3f} | "
            f"Std: {self.metrics_val.std:.3f} | "
            f"Min: {self.metrics_val.min:.3f} | "
            f"Max: {self.metrics_val.max:.3f} | "
            f"Gradient Norm: {self.metrics_val.gradient_norm:.3f} |\n"
        )
        
        return (
            f"-- Epoch {self.epoch + 1:02d} --\n" +
            val_str +
            train_str +
            val_stats_str +
            f"Train Time: {int(self.train_time) // 60} min {int(self.train_time) % 60} s"
        )


@dataclass
class TrainData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    
    @classmethod
    def make_split(cls, X, y, val_size):
        train_idx, val_idx = train_test_split(
            np.arange(len(X)),
            test_size=val_size,
            random_state=42,
            stratify=y
        )
        
        return cls(
            X[train_idx], y[train_idx],
            X[val_idx], y[val_idx]
        )


@dataclass
class TrainParameters:
    epochs: int
    batch_size: list[int]
    init_th_range: tuple[float, float]
    lr: list[float]


class Judge:
    def __init__(self, model: Model, th: np.ndarray, data: TrainData, max_batch: int) -> None:
        self.model = model
        self.th = th
        self.data = data
        self.max_batch = max_batch
        
        self.train_start_time = 0
        self.history: list[HistoryEntry] = []
        
    def train_start(self) -> Self:
        self.train_start_time = time.time()
        return self
    
    def epoch_end(self, epoch: int, grad_L: np.ndarray, print_info: bool = True, **params) -> Self:
        metrics_train = self._calc_metrics(self.data.X_train, self.data.y_train, grad_L)
        metrics_val = self._calc_metrics(self.data.X_val, self.data.y_val, grad_L)
        
        train_time_s = round(time.time() - self.train_start_time)
        
        self.history.append(HistoryEntry(
            epoch=epoch,
            metrics_train=metrics_train,
            metrics_val=metrics_val,
            train_time=train_time_s,
            params=params,
        ))
        
        if print_info:
            print(self.history[-1])
            
        return self
    
    def _calc_metrics(self, X, y, grad_L) -> Metrics:
        indices = np.random.permutation(len(X))[:self.max_batch]
        
        outputs: np.ndarray = self.model.measure_batch(X[indices], self.th)
        
        return Metrics.calc(y[indices], outputs, grad_L)


def compare_models(y_true, *models: tuple[str, np.ndarray, float]):
    """
    Compare multiple binary classification models.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels (-1 or 1).
    *models : tuple
        Triples of (model_name, output, treshold), where output contains
        continuous predictions in the range [-1, 1].
    """

    results = []

    for name, output, treshold in models:
        output = np.asarray(output)
        predictions = np.where(output >= treshold, 1, -1)

        mse = mean_squared_error(y_true, output)
        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(
            y_true, predictions, pos_label=1, zero_division=0
        )
        recall = recall_score(
            y_true, predictions, pos_label=1, zero_division=0
        )
        f1 = f1_score(
            y_true, predictions, pos_label=1, zero_division=0
        )
        balanced_acc = balanced_accuracy_score(
            y_true, predictions
        )
        mcc = matthews_corrcoef(y_true, predictions)

        tn, fp, fn, tp = confusion_matrix(
            y_true, predictions, labels=[-1, 1]
        ).ravel()

        specificity = tn / (tn + fp) if tn + fp else 0

        results.append({
            "Model": name,
            "MSE": mse,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "Specificity": specificity,
            "F1 Score": f1,
            "Balanced Accuracy": balanced_acc,
            "MCC": mcc,
        })

    df = pd.DataFrame(results).set_index("Model")

    return (
        df.style
        .format("{:.4f}")
        .highlight_max(
            subset=df.columns.drop("MSE"),
            color="#00ff0022"
        )
        .highlight_min(
            subset=["MSE"],
            color="#00ff0022"
        )
        .set_caption("Model Performance Comparison")
    )



def compare_models_md(y_true, *models: tuple[str, np.ndarray, float]) -> str:
    """
    Compare multiple binary classification models and generate
    a Markdown table suitable for README.md.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels (-1 or 1).
    *models : tuple
        Triples of (model_name, output, threshold).

    Returns
    -------
    str
        GitHub-compatible Markdown table.
    """

    results: list[dict[str, Any]] = []

    for name, output, threshold in models:
        output = np.asarray(output)
        predictions = np.where(output >= threshold, 1, -1)

        tn, fp, fn, tp = confusion_matrix(
            y_true, predictions, labels=[-1, 1]
        ).ravel()

        results.append({
            "Model": name,
            "MSE": mean_squared_error(y_true, output),
            "Accuracy": accuracy_score(y_true, predictions),
            "Precision": precision_score(
                y_true, predictions, pos_label=1, zero_division=0
            ),
            "Recall": recall_score(
                y_true, predictions, pos_label=1, zero_division=0
            ),
            "Specificity": tn / (tn + fp) if tn + fp else 0,
            "F1 Score": f1_score(
                y_true, predictions, pos_label=1, zero_division=0
            ),
            "Balanced Accuracy": balanced_accuracy_score(
                y_true, predictions
            ),
            "MCC": matthews_corrcoef(y_true, predictions),
        })

    df = pd.DataFrame(results).set_index("Model")

    # Generate Markdown table
    columns = ["Model"] + list(df.columns)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows = []

    for model, row in df.iterrows():
        values = [model]

        for col in df.columns:
            value = f"{row[col]:.4f}"

            values.append(value)

        rows.append("| " + " | ".join(map(str, values)) + " |")

    return "\n".join([header, separator] + rows)


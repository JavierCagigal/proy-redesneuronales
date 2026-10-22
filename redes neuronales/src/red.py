"""Funciones de la red, compartidas por 02-modelo.qmd y 03-experimentos.qmd."""

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score


def preparar_datos(ruta):
    """Separa nutrientes (X) y letra (y), divide 70/15/15, escala y pasa a tensores."""
    df = pd.read_csv(ruta)
    X = df.drop(columns=["nutriscore_grade"]).values
    y = df["nutriscore_grade"].map({"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}).values

    X_ent, X_resto, y_ent, y_resto = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_resto, y_resto, test_size=0.5, stratify=y_resto, random_state=42)

    escalador = StandardScaler().fit(X_ent)
    X_ent = escalador.transform(X_ent)
    X_val = escalador.transform(X_val)
    X_test = escalador.transform(X_test)

    X_ent = torch.tensor(X_ent, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_ent = torch.tensor(y_ent)
    y_val = torch.tensor(y_val)
    y_test = torch.tensor(y_test)

    return X_ent, y_ent, X_val, y_val, X_test, y_test


def crear_red(capas_ocultas, neuronas):
    """Crea una red nueva: 8 entradas, capas ocultas con ReLU y 5 salidas (una por letra)."""
    torch.manual_seed(42)
    capas = []
    entradas = 8
    for _ in range(capas_ocultas):
        capas.append(nn.Linear(entradas, neuronas))
        capas.append(nn.ReLU())
        entradas = neuronas
    capas.append(nn.Linear(entradas, 5))
    return nn.Sequential(*capas)


def entrenar(red, X_ent, y_ent, X_val, y_val, batch_size, lr, epocas=30):
    """Ajusta los pesos lote a lote y devuelve la pérdida de cada época."""
    cargador = DataLoader(TensorDataset(X_ent, y_ent), batch_size=batch_size, shuffle=True)
    optimizador = torch.optim.Adam(red.parameters(), lr=lr)
    funcion_perdida = nn.CrossEntropyLoss()
    historial = []

    for epoca in range(epocas):
        for X_lote, y_lote in cargador:
            optimizador.zero_grad()
            perdida = funcion_perdida(red(X_lote), y_lote)
            perdida.backward()
            optimizador.step()

        with torch.no_grad():
            historial.append({
                "epoca": epoca + 1,
                "perdida_ent": funcion_perdida(red(X_ent), y_ent).item(),
                "perdida_val": funcion_perdida(red(X_val), y_val).item(),
            })

    return pd.DataFrame(historial)


def evaluar(red, X, y):
    """Predice la letra de cada producto y devuelve accuracy, F1 macro y predicciones."""
    with torch.no_grad():
        predicciones = red(X).argmax(dim=1)
    accuracy = accuracy_score(y, predicciones)
    f1 = f1_score(y, predicciones, average="macro")
    return accuracy, f1, predicciones


# Comprobación rápida desde la carpeta "redes neuronales": python src/red.py
if __name__ == "__main__":
    X_ent, y_ent, X_val, y_val, X_test, y_test = preparar_datos("data/processed/productos_limpios.csv")
    red = crear_red(capas_ocultas=2, neuronas=64)
    entrenar(red, X_ent, y_ent, X_val, y_val, batch_size=64, lr=0.001)
    accuracy, f1, _ = evaluar(red, X_val, y_val)
    print(f"Accuracy: {accuracy:.3f}  F1 macro: {f1:.3f}")

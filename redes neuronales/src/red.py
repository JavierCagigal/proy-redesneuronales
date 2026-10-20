"""Funciones compartidas por 02-modelo.qmd y 03-experimentos.qmd.

Se importan desde los .qmd así:

    from src.red import preparar_datos, crear_red, entrenar, evaluar

Si cambiáis este archivo con VS Code abierto, reiniciad el kernel:
Python no vuelve a leer un módulo que ya ha importado.
"""

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score


def preparar_datos(ruta):
    """Lee el CSV limpio y devuelve los tres conjuntos listos para PyTorch.

    Devuelve: X_ent, y_ent, X_val, y_val, X_test, y_test (todos tensores).
    """
    # 1. Leer el CSV con pd.read_csv(ruta).
    df = pd.read_csv(ruta)

    # 2. X = todas las columnas menos "nutriscore_grade" (pista: .drop(columns=...).values).
    #    y = la letra convertida a número con .map({"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}).values
    X = df.drop(columns=["nutriscore_grade"]).values
    y = df["nutriscore_grade"].map({"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}).values

    # 3. Primera división: 70 % entrenamiento y 30 % resto.
    #    train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    #    stratify mantiene la misma proporción de letras en cada parte.
    X_ent, X_resto, y_ent, y_resto = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
   
    # 4. Segunda división: el resto, mitad validación y mitad test (test_size=0.5).
    #    Ojo: aquí stratify es la y del resto, no la y completa.
    X_val, X_test, y_val, y_test = train_test_split(X_resto, y_resto, test_size=0.5, stratify=y_resto, random_state=42)


    # 5. Escalar: StandardScaler().fit(...) SOLO con entrenamiento
    #    y después .transform(...) en los tres conjuntos.
    #    Si el escalador "viera" validación o test, habría fuga de datos.
    escalador = StandardScaler().fit(X_ent)
    X_ent = escalador.transform(X_ent)
    X_val = escalador.transform(X_val)
    X_test = escalador.transform(X_test)


    # 6. Pasar a tensores:
    #    X → torch.tensor(..., dtype=torch.float32)
    #    y → torch.tensor(...)   (enteros, que es lo que necesita CrossEntropyLoss)
    X_ent = torch.tensor(X_ent, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_ent = torch.tensor(y_ent)
    y_val = torch.tensor(y_val)
    y_test = torch.tensor(y_test)

    # 7. return X_ent, y_ent, X_val, y_val, X_test, y_test
    return X_ent, y_ent, X_val, y_val, X_test, y_test

    pass


def crear_red(capas_ocultas, neuronas):
    """Crea una red con `capas_ocultas` capas de `neuronas` neuronas.

    Con capas_ocultas=0 queda una sola capa Linear(8, 5): es el modelo base
    (equivale a una regresión logística).
    """
    # 1. torch.manual_seed(42) para que todas las redes empiecen con los mismos
    #    pesos y la comparación entre experimentos sea justa.
    torch.manual_seed(42)

    # 2. Una lista vacía de capas y una variable entradas = 8 (los 8 nutrientes).
    capas = []
    entradas = 8

    # 3. Un bucle que se repite capas_ocultas veces y en cada vuelta:
    #    - añade nn.Linear(entradas, neuronas)
    #    - añade nn.ReLU()  (sin ella, apilar capas seguiría siendo un modelo lineal)
    #    - actualiza entradas = neuronas (la siguiente capa recibe lo que sale de esta)
    for _ in range(capas_ocultas):
        capas.append(nn.Linear(entradas, neuronas))
        capas.append(nn.ReLU())
        entradas = neuronas

    # 4. Después del bucle, la capa de salida: nn.Linear(entradas, 5), una neurona por letra.
    #    No hace falta Softmax: CrossEntropyLoss ya lo aplica por dentro.
    capas.append(nn.Linear(entradas, 5))

    # 5. return nn.Sequential(*capas)   (el * "desempaqueta" la lista)
    return nn.Sequential(*capas)

    pass


def entrenar(red, X_ent, y_ent, X_val, y_val, batch_size, lr, epocas=30):
    """Entrena la red y devuelve una tabla con la pérdida de cada época.

    Devuelve un DataFrame con las columnas epoca, perdida_ent y perdida_val.
    """
    # 1. Preparar:
    #    - cargador = DataLoader(TensorDataset(X_ent, y_ent), batch_size=batch_size, shuffle=True)
    #    - optimizador = torch.optim.Adam(red.parameters(), lr=lr)
    #    - funcion_perdida = nn.CrossEntropyLoss()
    #    - historial = []
    cargador = DataLoader(TensorDataset(X_ent, y_ent), batch_size=batch_size, shuffle=True)
    optimizador = torch.optim.Adam(red.parameters(), lr=lr)
    funcion_perdida = nn.CrossEntropyLoss()
    historial = []

    # 2. Un bucle por épocas y, dentro, otro por los lotes del cargador:
    #        for X_lote, y_lote in cargador:
    #    Dentro, los 4 pasos del entrenamiento (tenéis que saber explicarlos):
    #    - optimizador.zero_grad()      → borra los gradientes del lote anterior
    #    - perdida = funcion_perdida(red(X_lote), y_lote)   → predice y mide el error
    #    - perdida.backward()           → calcula cómo cambiar cada peso
    #    - optimizador.step()           → cambia los pesos un poco (según lr)
    for epoca in range(epocas):
        for X_lote, y_lote in cargador:
            optimizador.zero_grad()
            perdida = funcion_perdida(red(X_lote), y_lote)
            perdida.backward()
            optimizador.step()

    # 3. Al acabar cada época, dentro de "with torch.no_grad():" (medir no es entrenar),
    #    añadir a historial un diccionario con:
    #    - "epoca": epoca + 1
    #    - "perdida_ent": funcion_perdida(red(X_ent), y_ent).item()
    #    - "perdida_val": funcion_perdida(red(X_val), y_val).item()
    with torch.no_grad():
            historial.append({
                "epoca": epoca + 1,
                "perdida_ent": funcion_perdida(red(X_ent), y_ent).item(),
                "perdida_val": funcion_perdida(red(X_val), y_val).item(),
            })

    # 4. return pd.DataFrame(historial)
    return pd.DataFrame(historial)

    pass


def evaluar(red, X, y):
    """Mide la red en un conjunto.

    Devuelve: accuracy, f1 macro y las predicciones (para la matriz de confusión).
    """
    # 1. Dentro de "with torch.no_grad():"
    #    predicciones = red(X).argmax(dim=1)
    #    La red da 5 puntuaciones por producto; argmax se queda con la letra que más puntúa.
    with torch.no_grad():
        predicciones = red(X).argmax(dim=1)

    # 2. accuracy = accuracy_score(y, predicciones)
    #    f1 = f1_score(y, predicciones, average="macro")
    #    "macro" hace la media de las 5 letras por igual, aunque haya más productos de unas que de otras.
    accuracy = accuracy_score(y, predicciones)
    f1 = f1_score(y, predicciones, average="macro")

    # 3. return accuracy, f1, predicciones
    return accuracy, f1, predicciones
    pass


# Comprobación rápida. Desde la carpeta "redes neuronales":
#     python src/red.py
# Este bloque solo se ejecuta así; al importar el archivo desde un .qmd se ignora.
if __name__ == "__main__":
    X_ent, y_ent, X_val, y_val, X_test, y_test = preparar_datos("data/processed/productos_limpios.csv")
    print("Tamaños:", len(X_ent), len(X_val), len(X_test))          # unos 4447 953 954

    red = crear_red(capas_ocultas=2, neuronas=64)
    print(red)

    historial = entrenar(red, X_ent, y_ent, X_val, y_val, batch_size=64, lr=0.001)
    print(historial.tail())                                           # las pérdidas deben bajar

    accuracy, f1, predicciones = evaluar(red, X_val, y_val)
    print(f"Accuracy: {accuracy:.3f}  F1 macro: {f1:.3f}")           # en torno a 0.75 y 0.73

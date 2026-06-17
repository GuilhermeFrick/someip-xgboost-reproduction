"""Carga do dataset KIM (SOME/IP XGBoost) — features já extraídas.

O dataset fornece os vetores de features prontos (12 features comportamentais por pacote)
e os rótulos binários (0 = normal, 1 = ataque), em arquivos .npy:
    X_train (7.116.674, 12)  X_test (7.116.673, 12)  y_train/y_test (N,)

Como os .npy são grandes (~326 MB cada), o caminho dos dados é configurável via a variável
de ambiente KIM_DATA (padrão: C:/Mestrado/KIM).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Por padrão, usa os .npz (comprimidos, versionados via Git LFS) dentro do próprio repo.
# Pode-se apontar para os .npy originais via a variável de ambiente KIM_DATA.
REPO_DATA = Path(__file__).resolve().parents[1] / "data"
DATA_DIR = Path(os.environ.get("KIM_DATA", str(REPO_DATA)))


def _load_array(name: str) -> np.ndarray:
    """Carrega `name` aceitando .npz (chave 'a') ou .npy, do diretório de dados."""
    npz = DATA_DIR / f"{name}.npz"
    npy = DATA_DIR / f"{name}.npy"
    if npz.exists():
        return np.load(npz)["a"]
    return np.load(npy)

# As 12 features são comportamentais (não usam campos de header), agrupadas em 5 classes
# (Tabela 1 do artigo). Mapeamento exato coluna->nome não é dado; nomes ilustrativos.
FEATURE_GROUPS = [
    "Time interval (IP)",
    "Payload likelihood (SOME/IP-SD, TCP/UDP)",
    "Payload entropy/cross-entropy (SOME/IP-SD, TCP/UDP)",
    "Payload changes / Hamming (SOME/IP-SD, TCP/UDP)",
    "Length changes (IP, TCP/UDP)",
]


def load(mmap: bool = True):
    """Carrega (X_train, y_train, X_test, y_test).

    Aceita os .npz comprimidos (padrão, versionados no repo) ou os .npy originais
    (se KIM_DATA apontar para a pasta do dataset). O parâmetro `mmap` só tem efeito
    para .npy (o .npz é descomprimido em memória).

    Returns:
        X_train (N,12) float32, y_train (N,) int, X_test, y_test.
    """
    _ = mmap  # mantido por compatibilidade; .npz não suporta memory-map
    X_train = _load_array("X_train")
    X_test = _load_array("X_test")
    y_train = _load_array("y_train")
    y_test = _load_array("y_test")
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    Xtr, ytr, Xte, yte = load()
    print("X_train:", Xtr.shape, "| X_test:", Xte.shape)
    for split, y in [("train", ytr), ("test", yte)]:
        u, c = np.unique(y, return_counts=True)
        d = {int(k): int(v) for k, v in zip(u, c)}
        print(f"{split}: {d}  (ataque% = {100*d.get(1,0)/len(y):.2f})")

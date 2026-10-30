"""Metricas compartilhadas pelas analises.

Tudo implementado aqui, sobre numpy, para que cada numero do artigo tenha uma definicao
legivel num so lugar. Convencoes:

  - classes na ordem de treino.LABELS: 0 Negativo, 1 Neutro, 2 Positivo;
  - "adequacao" (A) e a leitura que a interface exibe: P(Positivo) + 0.5 P(Neutro);
    a adequacao "verdadeira" de uma frase e 1, 0.5 ou 0 conforme o rotulo humano;
  - "erro de fronteira" e o erro que cruza a fronteira de risco: Negativo tomado por
    nao-Negativo ou o inverso. E o unico tipo de erro que muda o que o sistema rebaixa.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

NEG = 0
ADEQ_ROTULO = np.array([0.0, 0.5, 1.0])


def softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = np.asarray(z, np.float64) / t
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def ensemble(logits_folds: np.ndarray, t: float = 1.0) -> np.ndarray:
    """Soft voting: media das probabilidades dos modelos. logits_folds: [k, n, 3]."""
    return softmax(logits_folds, t).mean(0)


def adequacao(p: np.ndarray) -> np.ndarray:
    return p[..., 2] + 0.5 * p[..., 1]


# ---------------------------------------------------------------- classificacao


def por_classe(y: np.ndarray, pred: np.ndarray) -> dict:
    out = {}
    for c, nome in enumerate(("neg", "neu", "pos")):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out[nome] = {"precisao": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "n": int((y == c).sum())}
    out["acuracia"] = float((pred == y).mean()) if len(y) else float("nan")
    out["f1_macro"] = float(np.mean([out[n]["f1"] for n in ("neg", "neu", "pos")]))
    return out


def f1_macro(y, pred) -> float:
    return por_classe(y, pred)["f1_macro"]


def erro_fronteira(y, pred) -> np.ndarray:
    return (y == NEG) != (pred == NEG)


# ------------------------------------------------------------------- calibracao


def ece(conf: np.ndarray, acerto: np.ndarray, bins: int = 15) -> float:
    """Expected Calibration Error, bins de largura igual sobre a confianca."""
    conf, acerto = np.asarray(conf, float), np.asarray(acerto, float)
    lim = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, lim[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(conf[m].mean() - acerto[m].mean())
    return float(total)


def ece_topo(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """ECE da confianca da classe predita (o que a literatura chama de top-label)."""
    return ece(p.max(1), p.argmax(1) == y, bins)


def ece_classe(p: np.ndarray, y: np.ndarray, c: int, bins: int = 15) -> float:
    """Calibracao de P(classe c) como probabilidade de o rotulo ser c."""
    return ece(p[:, c], y == c, bins)


def ece_adequacao(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Distancia media entre a adequacao exibida e a adequacao do rotulo, por faixa."""
    a, s = adequacao(p), ADEQ_ROTULO[y]
    return ece(a, s, bins)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    oh = np.eye(p.shape[1])[y]
    return float(((p - oh) ** 2).sum(1).mean())


def nll(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def ajusta_temperatura(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperatura unica que minimiza a NLL (Guo et al., 2017). Busca em log T."""
    res = minimize_scalar(
        lambda lt: nll(softmax(logits, np.exp(lt)), y), bounds=(-3, 3), method="bounded"
    )
    return float(np.exp(res.x))


def curva_confiabilidade(conf, alvo, bins=10) -> list[dict]:
    lim = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, lim[1:-1], right=True), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        out.append(
            {
                "faixa": [float(lim[b]), float(lim[b + 1])],
                "n": int(m.sum()),
                "conf_media": float(conf[m].mean()) if m.any() else None,
                "alvo_medio": float(np.asarray(alvo, float)[m].mean()) if m.any() else None,
            }
        )
    return out


# -------------------------------------------------------------------- incerteza


def entropia(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(-1)


def auroc(score: np.ndarray, positivo: np.ndarray) -> float:
    """P(score de um positivo > score de um negativo), empates valendo 1/2."""
    score, positivo = np.asarray(score, float), np.asarray(positivo, bool)
    pos, neg = score[positivo], score[~positivo]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ordem = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    postos = np.empty(len(ordem))
    todos = np.concatenate([pos, neg])[ordem]
    i = 0
    while i < len(todos):
        j = i
        while j + 1 < len(todos) and todos[j + 1] == todos[i]:
            j += 1
        postos[ordem[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((postos[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def captura_por_orcamento(incerteza: np.ndarray, alvo: np.ndarray, orcamentos) -> dict:
    """Fracao dos itens-alvo que caem entre os q% mais incertos, para cada orcamento q."""
    ordem = np.argsort(-incerteza, kind="mergesort")
    alvo = np.asarray(alvo, bool)[ordem]
    out = {}
    for q in orcamentos:
        k = int(round(q * len(ordem)))
        out[f"{q:.2f}"] = float(alvo[:k].sum() / max(alvo.sum(), 1))
    return out


def risco_cobertura(incerteza: np.ndarray, erro: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Curva risco-cobertura e sua area (AURC). Retem primeiro os mais confiantes."""
    ordem = np.argsort(incerteza, kind="mergesort")
    e = np.asarray(erro, float)[ordem]
    n = np.arange(1, len(e) + 1)
    risco = np.cumsum(e) / n
    cobertura = n / len(e)
    return cobertura, risco, float(risco.mean())


# -------------------------------------------------------------------- bootstrap


def bootstrap_agrupado(grupos: np.ndarray, estatistica, n: int = 2000, seed: int = 7):
    """IC de 95% reamostrando GRUPOS (videos), nao frases.

    Frases do mesmo video nao sao independentes; reamostra-las uma a uma estreitaria o
    intervalo artificialmente. `estatistica(indices)` recebe os indices da reamostra.
    """
    rng = np.random.default_rng(seed)
    unicos, inv = np.unique(grupos, return_inverse=True)
    membros = [np.where(inv == g)[0] for g in range(len(unicos))]
    vals = []
    for _ in range(n):
        escolha = rng.integers(0, len(unicos), len(unicos))
        idx = np.concatenate([membros[g] for g in escolha])
        vals.append(estatistica(idx))
    vals = np.array(vals, float)
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]

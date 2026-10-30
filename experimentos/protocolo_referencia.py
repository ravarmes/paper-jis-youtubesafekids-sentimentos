"""Protocolo R: o ensemble do sistema, sobre a particao de referencia.

O QUE FAZ
---------
Retreina os cinco modelos do ensemble do YouTube Safe Kids sobre a mesma particao do
classificador de referencia (545 frases de teste retido, cinco folds estratificados sobre
as 2.177 restantes) e guarda os LOGITS de tres conjuntos:

  validacao  cada frase dos 80% recebe o logit do modelo do fold em que ficou de fora.
             E a amostra que ajusta a temperatura (analise_calibracao.py) e escolhe os
             limiares de deferencia (analise_deferencia.py) -- o teste retido nunca e
             usado para escolher nada.
  teste      as 545 frases, pelos cinco modelos; o ensemble e a media das probabilidades.
  ruido      as mesmas 545 frases em versoes degradadas que imitam o que a plataforma
             entrega de fato: legenda automatica, sem caixa nem pontuacao, com palavras e
             letras perdidas (analise_robustez.py).

VARIANTES DE RUIDO
------------------
  legenda          minusculas, sem pontuacao -- o formato da legenda automatica
  legenda+palN     legenda + cada palavra apagada com probabilidade N%
  legenda+letN     legenda + cada letra apagada ou trocada com probabilidade N%

As variantes estocasticas sao sorteadas com um gerador PROPRIO, antes do treino, em
SEMENTES_RUIDO sementes. O gerador da particao e do oversampling nao e tocado: o treino
e identico ao que seria sem nenhuma variante. O ruido e sintetico e nao modela o erro de
um reconhecedor de fala especifico; o que mede e a sensibilidade a degradacoes do tipo
que a transcricao automatica produz.

Saida: resultados/logits_referencia.npz
Uso:   python protocolo_referencia.py [--epocas N]   (--epocas so para teste rapido)
Custo: ~15 min numa GPU de notebook (RTX 4050); horas em CPU.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

import treino
from caminhos import RESULTADOS, prepara_saidas
from treino import LABELS, SEED

SEMENTES_RUIDO = [101, 202, 303]
TAXAS_PALAVRA = [0.10, 0.20]
TAXAS_LETRA = [0.05, 0.10]
ALFABETO = "abcdefghijklmnopqrstuvwxyzáàâãéêíóôõúç"

# Conferencia opcional: se a pasta do classificador de referencia estiver ao lado, a
# particao sorteada aqui e comparada com a dele, indice a indice.
REFERENCIA = (
    Path(__file__).resolve().parents[3]
    / "2026--lre--youtubesafekids-sentimentos/src/resultados/predicoes_teste.npz"
)


def legenda(txt: str) -> str:
    txt = re.sub(r"[^\w\s]", " ", txt.lower())
    return re.sub(r"\s+", " ", txt).strip()


def apaga_palavras(txt: str, p: float, rng: random.Random) -> str:
    pal = txt.split()
    fica = [w for w in pal if rng.random() >= p]
    return " ".join(fica) if fica else (pal[0] if pal else txt)


def degrada_letras(txt: str, p: float, rng: random.Random) -> str:
    saida = []
    for ch in txt:
        if ch.isalpha() and rng.random() < p:
            if rng.random() < 0.5:
                continue  # apagada
            ch = rng.choice(ALFABETO)  # trocada
        saida.append(ch)
    return "".join(saida)


def variantes(textos: list[str]) -> dict[str, list[list[str]]]:
    """{nome: [textos da semente 1, textos da semente 2, ...]}."""
    base = [legenda(t) for t in textos]
    out = {"limpo": [textos], "legenda": [base]}
    for p in TAXAS_PALAVRA:
        nome = f"legenda+pal{int(p*100):02d}"
        out[nome] = []
        for s in SEMENTES_RUIDO:
            g = random.Random(s)
            out[nome].append([apaga_palavras(t, p, g) for t in base])
    for p in TAXAS_LETRA:
        nome = f"legenda+let{int(p*100):02d}"
        out[nome] = []
        for s in SEMENTES_RUIDO:
            g = random.Random(s)
            out[nome].append([degrada_letras(t, p, g) for t in base])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=treino.EPOCAS)
    args = ap.parse_args()
    treino.EPOCAS = args.epocas
    prepara_saidas()

    # mesma sequencia de semeadura do codigo de referencia
    rng = np.random.default_rng(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    itens = treino.carrega_corpus()
    textos = [x["texto"] for x in itens]
    rotulos = np.array([x["rotulo"] for x in itens])
    print(f"corpus apos exclusao: {len(textos)} frases", flush=True)

    idx_teste, idx_resto, folds = treino.particao_referencia(rotulos, rng)
    print(f"teste retido: {len(idx_teste)} | treino+validacao: {len(idx_resto)}", flush=True)

    mesma_particao = None
    if REFERENCIA.exists():
        ref = np.load(REFERENCIA, allow_pickle=True)
        mesma_particao = bool(np.array_equal(ref["idx"], idx_teste))
        print(f"particao identica a de referencia: {mesma_particao}", flush=True)

    txt_teste = [textos[i] for i in idx_teste]
    var = variantes(txt_teste)
    nomes_var = list(var)

    tok = treino.tokenizador()
    logits_val = np.full((len(textos), 3), np.nan, np.float32)
    fold_val = np.full(len(textos), -1, np.int8)
    logits_teste = []  # [fold][n_teste, 3]
    logits_ruido = {n: [] for n in nomes_var}  # [nome][fold][semente][n_teste, 3]

    t0 = time.time()
    for k in range(treino.N_FOLDS):
        treino_ids = np.array([i for j in range(treino.N_FOLDS) if j != k for i in folds[j]])
        print(f"\n[fold {k+1}/{treino.N_FOLDS}] treino={len(treino_ids)}", flush=True)
        modelo = treino.treina_fold(
            [textos[i] for i in treino_ids], rotulos[treino_ids], tok, rng, f"fold {k+1}"
        )
        val_ids = np.array(folds[k])
        logits_val[val_ids] = treino.logits(modelo, [textos[i] for i in val_ids], tok)
        fold_val[val_ids] = k
        logits_teste.append(treino.logits(modelo, txt_teste, tok))
        for n in nomes_var:
            logits_ruido[n].append([treino.logits(modelo, ts, tok) for ts in var[n]])
        treino.libera(modelo)

    lt = np.stack(logits_teste)
    p_ens = np.mean(np.exp(lt) / np.exp(lt).sum(-1, keepdims=True), axis=0)
    acc = float((p_ens.argmax(1) == rotulos[idx_teste]).mean())
    print(f"\nensemble no teste retido: acuracia {acc:.4f}", flush=True)

    meta = {
        "gerado_em": time.strftime("%Y-%m-%d %H:%M"),
        "dispositivo": str(treino.DEV),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "epocas": treino.EPOCAS,
        "minutos": round((time.time() - t0) / 60, 1),
        "particao_identica_referencia": mesma_particao,
        "variantes": nomes_var,
        "sementes_ruido": SEMENTES_RUIDO,
    }
    np.savez_compressed(
        RESULTADOS / "logits_referencia.npz",
        logits_val=logits_val,
        fold_val=fold_val,
        idx_teste=idx_teste,
        idx_resto=idx_resto,
        logits_teste=lt,
        **{f"ruido__{n}": np.array(logits_ruido[n], np.float32) for n in nomes_var},
        exemplos_ruido=np.array(
            [[var[n][0][i] for n in nomes_var] for i in range(10)], dtype=object
        ),
        nomes_variantes=np.array(nomes_var),
        labels=np.array(LABELS),
        meta=np.array(json.dumps(meta)),
    )
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()

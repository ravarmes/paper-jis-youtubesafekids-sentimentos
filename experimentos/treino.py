"""Treino e inferencia do classificador, compartilhados pelos dois protocolos.

O classificador e o do sistema YouTube Safe Kids, sem nenhuma alteracao de receita:
BERTimbau Base (cased) ajustado, 5 epocas, lote 8, lr 3e-5, 100 passos de aquecimento,
comprimento maximo de 64 subpalavras, Random Oversampling apenas no treino de cada fold.
Este artigo NAO propoe um classificador: estuda o que a interface mostra a partir dele.
Por isso a receita e copiada, e nao ajustada.

Uma diferenca em relacao ao codigo de referencia, deliberada: aqui a inferencia devolve
LOGITS, e nao probabilidades. A calibracao por temperatura (analise_calibracao.py) opera
sobre logits; guardar so probabilidades obrigaria a inverter um softmax com perda de
precisao nas classes de probabilidade quase nula.

Pre-processamento: nenhum alem da tokenizacao WordPiece. A marcacao de negacao (NEG_)
que a plataforma aplica NAO e aplicada no codigo de experimento de referencia, e
portanto tambem nao aqui -- o objetivo e reproduzir o classificador avaliado.
"""

from __future__ import annotations

import csv
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from caminhos import CORPUS
from fontes import atribui_grupos, fonte, grupo_serie, normaliza

MODELO = "neuralmind/bert-base-portuguese-cased"
LABELS = ["Negativo", "Neutro", "Positivo"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}

SEED = 42
EPOCAS = 5
BATCH = 8
LR = 3e-5
WARMUP = 100
MAX_LEN = 64
TESTE_FRAC = 0.20
N_FOLDS = 5

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Os dois videos que o classificador de referencia excluiu antes de qualquer particao,
# por aparecerem no estudo de caso daquele trabalho. A exclusao e mantida para que a
# particao de teste seja exatamente a mesma (545 frases), e o classificador avaliado
# aqui seja o mesmo que o sistema usa.
TITULOS_EXCLUIDOS = {
    normaliza("Rick And Morty Dublado - Completo"),
    normaliza(
        "As 5 melhores músicas infantis! Cante com Léo, o caminhão curioso! "
        "Desenho animado para crianças"
    ),
}


def carrega_corpus() -> list[dict]:
    """Frases do corpus, na ordem do arquivo, sem os dois videos excluidos.

    A ordem importa: a particao de referencia e sorteada sobre os indices desta lista.
    """
    with open(CORPUS, encoding="utf-8-sig") as fh:
        linhas = list(csv.DictReader(fh, delimiter=";"))
    # o video e conferido sobre o arquivo inteiro, antes de qualquer exclusao
    grupos = atribui_grupos(linhas)
    itens = []
    for r, grupo in zip(linhas, grupos):
        rot = (r.get("AS") or "").strip()
        frase = (r.get("FRASE") or "").strip().strip('"').strip()
        titulo = (r.get("TITULO") or "").strip()
        if not frase or rot not in LABEL_TO_ID:
            continue
        if normaliza(titulo) in TITULOS_EXCLUIDOS:
            continue
        franquia, categoria = fonte(titulo)
        itens.append(
            {
                "texto": frase,
                "rotulo": LABEL_TO_ID[rot],
                "titulo": titulo,
                "franquia": franquia,
                "categoria": categoria,
                "grupo_video": grupo,
                "grupo_serie": grupo_serie(titulo),
                "anotacoes": [
                    (r.get(c) or "").strip() for c in ("P1", "A1", "A2", "A3")
                ],
            }
        )
    return itens


def particao_referencia(rotulos: np.ndarray, rng: np.random.Generator):
    """Teste retido de 20% e cinco folds estratificados sobre o resto.

    Replica, sorteio a sorteio, a particao do classificador de referencia: mesma
    semente, mesma ordem de chamadas ao gerador. Qualquer chamada extra ao `rng` antes
    deste ponto mudaria o teste retido.
    """
    idx_teste, idx_resto = [], []
    for c in range(3):
        ids = np.where(rotulos == c)[0]
        rng.shuffle(ids)
        corte = int(round(len(ids) * TESTE_FRAC))
        idx_teste.extend(ids[:corte].tolist())
        idx_resto.extend(ids[corte:].tolist())
    idx_teste, idx_resto = np.array(sorted(idx_teste)), np.array(sorted(idx_resto))

    folds = [[] for _ in range(N_FOLDS)]
    for c in range(3):
        ids = idx_resto[rotulos[idx_resto] == c]
        rng.shuffle(ids)
        for k, i in enumerate(ids):
            folds[k % N_FOLDS].append(int(i))
    return idx_teste, idx_resto, folds


def oversample(textos, rotulos, rng):
    """Random Oversampling: iguala as classes a maior, so no treino."""
    por_classe = defaultdict(list)
    for i, r in enumerate(rotulos):
        por_classe[int(r)].append(i)
    alvo = max(len(v) for v in por_classe.values())
    idx = []
    for c, ids in por_classe.items():
        idx.extend(ids)
        if len(ids) < alvo:
            idx.extend(rng.choice(ids, size=alvo - len(ids), replace=True).tolist())
    rng.shuffle(idx)
    return [textos[i] for i in idx], rotulos[idx]


def tokenizador():
    return AutoTokenizer.from_pretrained(MODELO, local_files_only=True)


def _colador(tok):
    def _f(lote):
        textos = [t for t, _ in lote]
        enc = tok(textos, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        enc["labels"] = torch.tensor([r for _, r in lote])
        return enc

    return _f


def treina_fold(textos, rotulos, tok, rng, etiqueta=""):
    modelo = AutoModelForSequenceClassification.from_pretrained(
        MODELO, num_labels=3, local_files_only=True
    ).to(DEV)
    tx, ry = oversample(textos, rotulos, rng)
    ds = list(zip(tx, [int(r) for r in ry]))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=_colador(tok))

    opt = torch.optim.AdamW(modelo.parameters(), lr=LR)
    total = len(dl) * EPOCAS
    sched = get_linear_schedule_with_warmup(opt, WARMUP, total)

    modelo.train()
    t0 = time.time()
    for _ in range(EPOCAS):
        for lote in dl:
            lote = {k: v.to(DEV) for k, v in lote.items()}
            modelo(**lote).loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
    print(f"    {etiqueta} treinado em {(time.time()-t0)/60:.1f} min ({total} passos)", flush=True)
    modelo.eval()
    return modelo


@torch.no_grad()
def logits(modelo, textos, tok, batch=32) -> np.ndarray:
    saidas = []
    for i in range(0, len(textos), batch):
        enc = tok(
            textos[i : i + batch],
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        enc = {k: v.to(DEV) for k, v in enc.items()}
        saidas.append(modelo(**enc).logits.float().cpu().numpy())
    return np.vstack(saidas) if saidas else np.zeros((0, 3), np.float32)


def libera(modelo) -> None:
    del modelo
    if DEV.type == "cuda":
        torch.cuda.empty_cache()

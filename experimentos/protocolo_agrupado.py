"""Protocolo G: o mesmo classificador avaliado sobre videos e series nunca vistos.

POR QUE
-------
A particao de referencia sorteia FRASES. Frases do mesmo video -- mesmos personagens,
mesmo narrador, mesmos bordoes -- caem dos dois lados, e o teste mede em parte a
capacidade de reconhecer um video ja visto. Em operacao isso nunca acontece: cada busca
devolve videos que o modelo nao viu, e com frequencia de series que ele nao viu.

Este script estima o desempenho nessas duas condicoes, com validacao cruzada de cinco
folds sobre as 2.722 frases, em tres esquemas de particao:

  frase  folds estratificados por frase (a condicao da particao de referencia)
  video  folds agrupados por video: nenhum video tem frases dos dois lados
  serie  folds agrupados por franquia: nenhuma serie tem frases dos dois lados
         (videos avulsos sao a propria franquia -- ver fontes.grupo_serie)

Cada frase recebe o logit do unico modelo que nao a viu (um modelo por fold, sem
ensemble: o que se compara aqui sao particoes, nao o sistema). A receita de treino e a
mesma do protocolo R.

Os folds agrupados usam StratifiedGroupKFold, que equilibra as classes ate onde os
grupos permitem. Com franquias grandes e quase monoclasse (South Park, Rick and Morty),
o equilibrio por fold e aproximado; a distribuicao efetiva de cada fold vai no .npz.

Saida: resultados/logits_agrupado.npz
Uso:   python protocolo_agrupado.py [--esquemas frase video serie] [--epocas N]
Custo: ~15 treinos de BERTimbau; ~40 min numa GPU de notebook.
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

import treino
from caminhos import RESULTADOS, prepara_saidas
from treino import LABELS, SEED

ESQUEMAS = ["frase", "video", "serie"]


def folds_do_esquema(esquema: str, rotulos: np.ndarray, itens: list[dict]) -> np.ndarray:
    """Vetor com o fold de cada frase."""
    fold = np.full(len(rotulos), -1, np.int8)
    if esquema == "frase":
        div = StratifiedKFold(n_splits=treino.N_FOLDS, shuffle=True, random_state=SEED)
        partes = div.split(np.zeros(len(rotulos)), rotulos)
    else:
        chave = "grupo_video" if esquema == "video" else "grupo_serie"
        grupos = np.array([x[chave] for x in itens])
        div = StratifiedGroupKFold(n_splits=treino.N_FOLDS, shuffle=True, random_state=SEED)
        partes = div.split(np.zeros(len(rotulos)), rotulos, grupos)
    for k, (_, teste) in enumerate(partes):
        fold[teste] = k
    assert (fold >= 0).all()
    return fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esquemas", nargs="+", default=ESQUEMAS, choices=ESQUEMAS)
    ap.add_argument("--epocas", type=int, default=treino.EPOCAS)
    args = ap.parse_args()
    treino.EPOCAS = args.epocas
    prepara_saidas()

    itens = treino.carrega_corpus()
    textos = [x["texto"] for x in itens]
    rotulos = np.array([x["rotulo"] for x in itens])
    tok = treino.tokenizador()

    saida = {}
    meta = {"gerado_em": time.strftime("%Y-%m-%d %H:%M"), "epocas": treino.EPOCAS, "esquemas": {}}
    t0 = time.time()
    for esquema in args.esquemas:
        # cada esquema recomeca da mesma semente: as diferencas entre esquemas vem da
        # particao, nao de sorteios de inicializacao e oversampling acumulados
        rng = np.random.default_rng(SEED)
        random.seed(SEED)
        torch.manual_seed(SEED)

        fold = folds_do_esquema(esquema, rotulos, itens)
        oof = np.full((len(textos), 3), np.nan, np.float32)
        print(f"\n=== esquema {esquema} ===", flush=True)
        dist = []
        for k in range(treino.N_FOLDS):
            tr = np.where(fold != k)[0]
            te = np.where(fold == k)[0]
            dist.append({"n_teste": int(len(te)), "classes_teste": np.bincount(rotulos[te], minlength=3).tolist()})
            print(f"[{esquema} fold {k+1}] treino={len(tr)} teste={len(te)} classes={dist[-1]['classes_teste']}", flush=True)
            modelo = treino.treina_fold([textos[i] for i in tr], rotulos[tr], tok, rng, f"{esquema} {k+1}")
            oof[te] = treino.logits(modelo, [textos[i] for i in te], tok)
            treino.libera(modelo)
        acc = float((oof.argmax(1) == rotulos).mean())
        print(f"  acuracia OOF ({esquema}): {acc:.4f}", flush=True)
        saida[f"logits__{esquema}"] = oof
        saida[f"fold__{esquema}"] = fold
        meta["esquemas"][esquema] = {"folds": dist, "acuracia_oof": acc}

    meta["minutos"] = round((time.time() - t0) / 60, 1)

    # Rodar so parte dos esquemas preserva os demais ja salvos: o treino de cada esquema
    # recomeca da semente, entao os esquemas nao dependem uns dos outros.
    destino = RESULTADOS / "logits_agrupado.npz"
    if destino.exists() and set(args.esquemas) != set(ESQUEMAS):
        antigo = np.load(destino, allow_pickle=True)
        assert np.array_equal(antigo["rotulos"], rotulos), "corpus mudou: rode todos os esquemas"
        meta_antiga = json.loads(str(antigo["meta"]))
        for e in ESQUEMAS:
            if e not in args.esquemas and f"logits__{e}" in antigo.files:
                saida[f"logits__{e}"] = antigo[f"logits__{e}"]
                saida[f"fold__{e}"] = antigo[f"fold__{e}"]
                meta["esquemas"][e] = meta_antiga["esquemas"][e]

    np.savez_compressed(
        RESULTADOS / "logits_agrupado.npz",
        rotulos=rotulos,
        categoria=np.array([x["categoria"] for x in itens]),
        franquia=np.array([x["franquia"] for x in itens]),
        grupo_video=np.array([x["grupo_video"] for x in itens]),
        grupo_serie=np.array([x["grupo_serie"] for x in itens]),
        labels=np.array(LABELS),
        meta=np.array(json.dumps(meta)),
        **saida,
    )
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()

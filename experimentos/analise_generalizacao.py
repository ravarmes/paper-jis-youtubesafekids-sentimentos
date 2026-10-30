"""Generalizacao a videos e series nao vistos, e quem o filtro rebaixa.

Le resultados/logits_agrupado.npz (protocolo_agrupado.py) e responde a duas perguntas.

1. QUANTO DO DESEMPENHO SOBREVIVE A UM VIDEO NOVO?
   Macro F1, acuracia e recall/F1 da classe Negativo em cada esquema de particao
   (frase, video, serie). Intervalos de 95% por bootstrap AGRUPADO por video: frases do
   mesmo video nao sao amostras independentes. As diferencas entre esquemas sao pareadas
   (mesmas frases, predicoes diferentes) e reamostradas pelos mesmos grupos.

2. DE QUEM E O CONTEUDO QUE O FILTRO REBAIXA?
   Por categoria (adulta, classica, narrativa, educativa):
     - taxa de frases preditas Negativo (o que o filtro rebaixaria),
     - recall de Negativo dentro da categoria (onde ha rotulos Negativo suficientes),
     - taxa de falso Negativo sobre frases nao-Negativas (rebaixamento indevido).
   E no nivel do VIDEO, que e o que a interface ordena: cada video recebe a media das
   probabilidades das suas frases (agregacao por media de frases) e e rebaixado se a
   classe resultante for Negativo. O mesmo criterio aplicado aos rotulos humanos (a
   proporcao de cada rotulo entre as frases do video) da o rebaixamento "oraculo": o que
   um classificador perfeito de sentimento faria. A diferenca entre os dois separa o que
   e erro do modelo do que e consequencia de usar sentimento como sinal de risco.

Saidas: resultados/generalizacao.json
"""

from __future__ import annotations

import json

import numpy as np

import metricas as M
from caminhos import RESULTADOS, prepara_saidas

ESQUEMAS = ["frase", "video", "serie"]
CATEGORIAS = ["adulta", "classica", "narrativa", "educativa"]
MIN_NEG_RECALL = 10  # abaixo disso o recall de Negativo por categoria nao e reportado
N_BOOT = 2000


def resumo(y, pred, grupos, idx=None, com_ic=True) -> dict:
    if idx is not None:
        y, pred, grupos = y[idx], pred[idx], grupos[idx]
    pc = M.por_classe(y, pred)
    out = {
        "n": int(len(y)),
        "acuracia": pc["acuracia"],
        "f1_macro": pc["f1_macro"],
        "f1_neg": pc["neg"]["f1"],
        "recall_neg": pc["neg"]["recall"],
        "precisao_neg": pc["neg"]["precisao"],
        "f1_neu": pc["neu"]["f1"],
        "f1_pos": pc["pos"]["f1"],
        "erro_fronteira": float(M.erro_fronteira(y, pred).mean()),
    }
    if com_ic:
        out["ic_f1_macro"] = M.bootstrap_agrupado(grupos, lambda i: M.f1_macro(y[i], pred[i]), N_BOOT)
        out["ic_recall_neg"] = M.bootstrap_agrupado(
            grupos, lambda i: M.por_classe(y[i], pred[i])["neg"]["recall"], N_BOOT
        )
    return out


def por_categoria(y, pred, cat) -> dict:
    out = {}
    for c in CATEGORIAS:
        m = cat == c
        yc, pc = y[m], pred[m]
        n_neg = int((yc == M.NEG).sum())
        nao_neg = yc != M.NEG
        out[c] = {
            "n": int(m.sum()),
            "rotulos": np.bincount(yc, minlength=3).tolist(),
            "preditos": np.bincount(pc, minlength=3).tolist(),
            "taxa_pred_neg": float((pc == M.NEG).mean()),
            "taxa_rotulo_neg": float((yc == M.NEG).mean()),
            "n_neg": n_neg,
            "recall_neg": float((pc[yc == M.NEG] == M.NEG).mean()) if n_neg >= MIN_NEG_RECALL else None,
            "precisao_neg": float((yc[pc == M.NEG] == M.NEG).mean()) if (pc == M.NEG).sum() >= MIN_NEG_RECALL else None,
            "n_pred_neg": int((pc == M.NEG).sum()),
            "n_acerto_neg": int(((pc == M.NEG) & (yc == M.NEG)).sum()),
            "falso_neg": float((pc[nao_neg] == M.NEG).mean()) if nao_neg.any() else None,
            "n_falso_neg": int((pc[nao_neg] == M.NEG).sum()),
            "acuracia": float((pc == yc).mean()),
        }
    return out


def rebaixamento_videos(y, probs, grupos, cat) -> dict:
    """Rebaixamento no nivel do video: modelo (media das probabilidades) e oraculo."""
    oh = np.eye(3)[y]
    unicos, inv = np.unique(grupos, return_inverse=True)
    linhas = []
    for g in range(len(unicos)):
        m = inv == g
        p_mod = probs[m].mean(0)
        p_hum = oh[m].mean(0)
        # empate no oraculo: decide-se contra o rebaixamento (o criterio conservador)
        neg_hum = p_hum[M.NEG] > max(p_hum[1], p_hum[2])
        linhas.append(
            {
                "video": str(unicos[g]),
                "categoria": cat[m][0],
                "n_frases": int(m.sum()),
                "rotulos": np.bincount(y[m], minlength=3).tolist(),
                "rebaixa_modelo": bool(p_mod.argmax() == M.NEG),
                "rebaixa_oraculo": bool(neg_hum),
            }
        )
    out = {}
    for c in CATEGORIAS:
        vs = [l for l in linhas if l["categoria"] == c]
        n = len(vs)
        out[c] = {
            "n_videos": n,
            "rebaixa_oraculo": sum(l["rebaixa_oraculo"] for l in vs),
            "rebaixa_modelo": sum(l["rebaixa_modelo"] for l in vs),
            "rebaixa_modelo_nao_oraculo": sum(l["rebaixa_modelo"] and not l["rebaixa_oraculo"] for l in vs),
            "oraculo_nao_modelo": sum(l["rebaixa_oraculo"] and not l["rebaixa_modelo"] for l in vs),
        }
        # nos conteudos infantis e classicos os casos sao poucos: registram-se um a um,
        # para que o texto possa nomea-los e o leitor conferir
        if c != "adulta":
            out[c]["videos_rebaixados"] = [
                {k: l[k] for k in ("video", "rotulos", "rebaixa_oraculo", "rebaixa_modelo")}
                for l in vs
                if l["rebaixa_oraculo"] or l["rebaixa_modelo"]
            ]
    out["_n_videos"] = len(linhas)
    return out


def main():
    prepara_saidas()
    d = np.load(RESULTADOS / "logits_agrupado.npz", allow_pickle=True)
    y = d["rotulos"]
    cat = d["categoria"]
    gv = d["grupo_video"]
    gs = d["grupo_serie"]
    # unidade de reamostragem: o video, salvo no esquema por serie, onde os erros sao
    # correlacionados por franquia inteira (cada franquia cai num fold so)
    cluster = {"frase": gv, "video": gv, "serie": gs}

    res = {"esquemas": {}, "diferencas": {}, "rebaixamento_videos": {}}
    # contagens do protocolo G: grupos e tamanho dos folds por serie
    series_multi = [g for g in np.unique(gs) if len(np.unique(gv[gs == g])) > 1]
    res["protocolo"] = {
        "n_frases": int(len(y)),
        "n_grupos_video": int(len(np.unique(gv))),
        "n_grupos_serie": int(len(np.unique(gs))),
        "n_series_multivideo": len(series_multi),
        "frases_series_multivideo": int(np.isin(gs, series_multi).sum()),
    }
    if "fold__serie" in d.files:
        fs = d["fold__serie"]
        tam = [int((fs == k).sum()) for k in np.unique(fs)]
        neg = [float((y[fs == k] == M.NEG).mean()) for k in np.unique(fs)]
        res["protocolo"].update(
            serie_fold_min=min(tam), serie_fold_max=max(tam),
            serie_fold_neg_min=min(neg), serie_fold_neg_max=max(neg),
            serie_treino_min=int(len(y) - max(tam)),
        )
    preds = {}
    for e in ESQUEMAS:
        if f"logits__{e}" not in d.files:
            continue
        lg = d[f"logits__{e}"]
        probs = M.softmax(lg)
        pred = probs.argmax(1)
        preds[e] = pred
        bloco = resumo(y, pred, cluster[e])
        bloco["unidade_bootstrap"] = "serie" if e == "serie" else "video"
        bloco["por_categoria"] = por_categoria(y, pred, cat)
        res["esquemas"][e] = bloco
        res["rebaixamento_videos"][e] = rebaixamento_videos(y, probs, gv, cat)
        print(f"{e:6s} F1-macro {bloco['f1_macro']:.4f} {bloco['ic_f1_macro']} "
              f"recall-neg {bloco['recall_neg']:.4f}")

    for e in ("video", "serie"):
        if e in preds and "frase" in preds:
            a, b = preds["frase"], preds[e]
            delta = M.f1_macro(y, a) - M.f1_macro(y, b)
            ic = M.bootstrap_agrupado(cluster[e], lambda i: M.f1_macro(y[i], a[i]) - M.f1_macro(y[i], b[i]), N_BOOT)
            dneg = M.por_classe(y, a)["neg"]["recall"] - M.por_classe(y, b)["neg"]["recall"]
            res["diferencas"][f"frase-{e}"] = {"delta_f1_macro": delta, "ic": ic, "delta_recall_neg": dneg}
            print(f"frase - {e}: {delta:.4f} {ic}")

    with open(RESULTADOS / "generalizacao.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

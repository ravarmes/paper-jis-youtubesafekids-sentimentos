"""Calibracao do que a interface exibe.

A interface do YouTube Safe Kids mostra, em cada resultado, uma cor (a classe predita) e
um percentual (a adequacao predita A = P(Positivo) + 0.5 P(Neutro)). Um responsavel le
esses numeros como frequencias: "90% adequado" deveria querer dizer algo verificavel.
Este script mede se querem, e se uma correcao barata -- escala de temperatura (Guo et al.,
2017), um unico parametro -- resolve.

PROTOCOLO R (o ensemble do sistema, teste retido de 545 frases)
  - A temperatura T e ajustada por NLL nos logits de VALIDACAO fora-da-amostra (cada
    frase dos 80% pelo modelo que nao a viu). O teste retido nao participa do ajuste.
  - O ensemble calibrado aplica T aos logits de cada modelo antes da media.
  - Metricas, antes e depois: ECE da confianca da classe predita, ECE de P(Negativo)
    (a probabilidade que decide o rebaixamento), ECE da adequacao exibida (A contra a
    adequacao do rotulo: 1, 0.5, 0), Brier e NLL. ICs por bootstrap de frases.
  - Quanto a interface "promete": fracao de itens exibidos com confianca >= 0.99 e a
    acuracia efetiva entre eles.

PROTOCOLO G (particoes por video e por serie)
  - Calibracao sob mudanca de distribuicao: T de cada fold e ajustada nas predicoes
    fora-da-amostra dos OUTROS quatro folds (ajuste cruzado: o rotulo do fold avaliado
    nunca entra no ajuste da sua temperatura), e aplicada ao fold.

Saidas: resultados/calibracao.json, figuras/fig-confiabilidade.pdf
"""

from __future__ import annotations

import json

import numpy as np

import metricas as M
from caminhos import FIGURAS, FIGURAS_PT, RESULTADOS, prepara_saidas

N_BOOT = 2000
LIMIAR_ALTO = 0.99


def bloco_metricas(p, y) -> dict:
    return {
        "acuracia": float((p.argmax(1) == y).mean()),
        "ece_topo": M.ece_topo(p, y),
        "ece_neg": M.ece_classe(p, y, M.NEG),
        "ece_adequacao": M.ece_adequacao(p, y),
        "brier": M.brier(p, y),
        "nll": M.nll(p, y),
        "conf_media": float(p.max(1).mean()),
        "frac_conf_alta": float((p.max(1) >= LIMIAR_ALTO).mean()),
        "acuracia_conf_alta": float((p.argmax(1) == y)[p.max(1) >= LIMIAR_ALTO].mean())
        if (p.max(1) >= LIMIAR_ALTO).any()
        else None,
        "n_conf_alta": int((p.max(1) >= LIMIAR_ALTO).sum()),
        "acuracia_conf_baixa": float((p.argmax(1) == y)[p.max(1) < LIMIAR_ALTO].mean())
        if (p.max(1) < LIMIAR_ALTO).any()
        else None,
    }


def ic_bootstrap(p, y, grupos, chave_fn, n=N_BOOT, seed=11):
    """IC de 95% reamostrando VIDEOS: frases do mesmo video nao sao independentes."""
    return M.bootstrap_agrupado(grupos, lambda i: chave_fn(p[i], y[i]), n=n, seed=seed)


def protocolo_r() -> tuple[dict, dict]:
    d = np.load(RESULTADOS / "logits_referencia.npz", allow_pickle=True)
    from treino import carrega_corpus

    itens = carrega_corpus()
    rot = np.array([x["rotulo"] for x in itens])
    idx_resto, idx_teste = d["idx_resto"], d["idx_teste"]
    lv, yv = d["logits_val"][idx_resto], rot[idx_resto]
    lt, yt = d["logits_teste"], rot[idx_teste]

    t = M.ajusta_temperatura(lv, yv)
    p0, p1 = M.ensemble(lt, 1.0), M.ensemble(lt, t)
    gt = np.array([itens[i]["grupo_video"] for i in idx_teste])
    # a temperatura, aplicada antes da media, muda algumas classes do ensemble
    muda_classe = p0.argmax(1) != p1.argmax(1)
    muda_rebaixa = (p0.argmax(1) == M.NEG) != (p1.argmax(1) == M.NEG)
    pv0, pv1 = M.softmax(lv), M.softmax(lv, t)

    res = {
        "temperatura": t,
        "n_validacao": int(len(yv)),
        "n_teste": int(len(yt)),
        "n_videos_teste": int(len(set(gt))),
        "temperatura_muda_classe": int(muda_classe.sum()),
        "temperatura_muda_rebaixamento": int(muda_rebaixa.sum()),
        "validacao_antes": bloco_metricas(pv0, yv),
        "validacao_depois": bloco_metricas(pv1, yv),
        "teste_antes": bloco_metricas(p0, yt),
        "teste_depois": bloco_metricas(p1, yt),
        "ic": {
            "ece_topo_antes": ic_bootstrap(p0, yt, gt, M.ece_topo),
            "ece_topo_depois": ic_bootstrap(p1, yt, gt, M.ece_topo),
            "ece_adequacao_antes": ic_bootstrap(p0, yt, gt, M.ece_adequacao),
            "ece_adequacao_depois": ic_bootstrap(p1, yt, gt, M.ece_adequacao),
        },
        "confiabilidade": {
            "topo_antes": M.curva_confiabilidade(p0.max(1), p0.argmax(1) == yt),
            "topo_depois": M.curva_confiabilidade(p1.max(1), p1.argmax(1) == yt),
            "adequacao_antes": M.curva_confiabilidade(M.adequacao(p0), M.ADEQ_ROTULO[yt]),
            "adequacao_depois": M.curva_confiabilidade(M.adequacao(p1), M.ADEQ_ROTULO[yt]),
        },
    }
    print(f"T = {t:.3f} | ECE topo {res['teste_antes']['ece_topo']:.4f} -> {res['teste_depois']['ece_topo']:.4f}"
          f" | ECE adequacao {res['teste_antes']['ece_adequacao']:.4f} -> {res['teste_depois']['ece_adequacao']:.4f}")
    curvas = {"p0": p0, "p1": p1, "y": yt}
    return res, curvas


def protocolo_g(t_ref: float) -> dict:
    """Calibracao sob particao agrupada.

    Alem da temperatura de ajuste cruzado, aplica a temperatura do protocolo R (`t_ref`,
    ajustada como se faz de costume, numa validacao por frase) as predicoes de videos e
    series nao vistos: mede quanto a correcao feita do jeito usual subcorrige em operacao.
    """
    caminho = RESULTADOS / "logits_agrupado.npz"
    if not caminho.exists():
        return {}
    d = np.load(caminho, allow_pickle=True)
    y = d["rotulos"]
    out = {}
    for e in ("frase", "video", "serie"):
        if f"logits__{e}" not in d.files:
            continue
        lg, fold = d[f"logits__{e}"], d[f"fold__{e}"]
        p_cal = np.zeros_like(lg, dtype=np.float64)
        temps = []
        for k in np.unique(fold):
            fora = fold != k
            tk = M.ajusta_temperatura(lg[fora], y[fora])
            temps.append(tk)
            p_cal[fold == k] = M.softmax(lg[fold == k], tk)
        p0 = M.softmax(lg)
        out[e] = {
            "temperaturas": temps,
            "antes": bloco_metricas(p0, y),
            "depois": bloco_metricas(p_cal, y),
            "com_t_referencia": bloco_metricas(M.softmax(lg, t_ref), y),
        }
        print(f"G/{e}: ECE topo {out[e]['antes']['ece_topo']:.4f} -> {out[e]['depois']['ece_topo']:.4f}; "
              f"T medio {np.mean(temps):.2f}")
    return out


TEXTO = {
    "en": {
        "conf": "Confidence of the predicted class", "acc": "Observed accuracy",
        "adeq": "Displayed suitability $A$", "adeq_y": "Mean label suitability",
        "x": "Mean displayed value (bin)", "cru": "Uncalibrated", "temp": "Temperature-scaled",
    },
    "pt": {
        "conf": "Confiança da classe predita", "acc": "Acurácia observada",
        "adeq": "Adequação exibida $A$", "adeq_y": "Adequação média do rótulo",
        "x": "Valor médio exibido (faixa)", "cru": "Sem calibração", "temp": "Com temperatura",
    },
}


def figura(curvas: dict, idioma: str = "en") -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType: o PDF nao leva fonte Type 3
    import matplotlib.pyplot as plt

    from matplotlib.ticker import FuncFormatter

    T = TEXTO[idioma]
    p0, p1, y = curvas["p0"], curvas["p1"], curvas["y"]
    fig, eixos = plt.subplots(1, 2, figsize=(7.0, 3.1))
    paineis = [
        (T["conf"], lambda p: (p.max(1), p.argmax(1) == y), T["acc"]),
        (T["adeq"], lambda p: (M.adequacao(p), M.ADEQ_ROTULO[y]), T["adeq_y"]),
    ]
    for ax, (titulo, fn, ylab) in zip(eixos, paineis):
        ax.plot([0, 1], [0, 1], ls=":", c="0.5", lw=1)
        for p, nome, cor, mk in ((p0, T["cru"], "#b2182b", "o"), (p1, T["temp"], "#2166ac", "s")):
            c, a = fn(p)
            curva = M.curva_confiabilidade(c, a, bins=10)
            xs = [b["conf_media"] for b in curva if b["n"] >= 5]
            ys = [b["alvo_medio"] for b in curva if b["n"] >= 5]
            ns = [b["n"] for b in curva if b["n"] >= 5]
            ax.plot(xs, ys, marker=mk, c=cor, lw=1.2, ms=3.5, label=nome)
            for x_, y_, n_ in zip(xs, ys, ns):
                if nome == T["cru"]:
                    ax.annotate(str(n_), (x_, y_), textcoords="offset points", xytext=(3, -9), fontsize=6, color=cor)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_title(titulo, fontsize=8.5)
        ax.set_xlabel(T["x"], fontsize=8)
        if idioma == "pt":
            virgula = FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))
            ax.xaxis.set_major_formatter(virgula)
            ax.yaxis.set_major_formatter(virgula)
        ax.set_ylabel(ylab, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)
    eixos[0].legend(fontsize=7, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig((FIGURAS_PT if idioma == "pt" else FIGURAS) / "fig-confiabilidade.pdf")
    plt.close(fig)


def main():
    prepara_saidas()
    res_r, curvas = protocolo_r()
    res = {"referencia": res_r, "agrupado": protocolo_g(res_r["temperatura"])}
    with open(RESULTADOS / "calibracao.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    figura(curvas, "en")
    figura(curvas, "pt")


if __name__ == "__main__":
    main()

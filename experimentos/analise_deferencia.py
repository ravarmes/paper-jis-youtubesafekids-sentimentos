"""Deferencia ao responsavel: que itens a interface deveria marcar para conferencia.

A interface nao precisa fingir certeza uniforme. Se marcar uma fracao pequena dos
resultados como "confira este", o responsavel concentra a atencao onde o filtro mais
erra. A pergunta e empirica: que sinal de incerteza separa melhor os erros, e quantos
erros CAROS (os que cruzam a fronteira de risco -- Negativo tomado por nao-Negativo, ou
o inverso) cabem num orcamento de conferencia de 5, 10, 20 ou 30% dos itens.

Sinais comparados, sobre o ensemble calibrado do protocolo R (T de
analise_calibracao.py, ajustada na validacao):

  confianca    1 - max_c P(c)                       (a confianca da classe predita)
  margem       -(P(1a classe) - P(2a classe))
  entropia     H(media das probabilidades)
  desacordo    informacao mutua entre os cinco modelos: H(media) - media(H(modelo k)).
               So existe porque a configuracao de referencia roda um ensemble
  meio         -|P(Negativo) - 1/2|: proximidade de P(Negativo) a meio
  decisao      -|P(Negativo) - max(P(Neutro), P(Positivo))|: distancia ate a decisao de
               rebaixar, que e um argmax entre tres classes. E o sinal dirigido ao erro
               caro, na forma exata da regra que a interface aplica

Para cada sinal: AUROC na deteccao de erro (qualquer) e de erro de fronteira, AURC
(area sob a curva risco-cobertura) e a fracao de erros capturada em cada orcamento. Para
os sinais principais, IC de 95% da captura a 10% e da diferenca para a confianca, por
bootstrap de VIDEOS.

O mesmo quadro e repetido nas particoes agrupadas (protocolo G), com modelo unico por
frase e temperatura de ajuste cruzado (a de cada fold vem dos outros quatro); ali nao ha
desacordo entre modelos.

Saidas: resultados/deferencia.json, figuras/fig-deferencia.pdf
"""

from __future__ import annotations

import json

import numpy as np

import metricas as M
from caminhos import FIGURAS, FIGURAS_PT, RESULTADOS, prepara_saidas

ORCAMENTOS = [0.05, 0.10, 0.20, 0.30]
SINAIS = ["confianca", "margem", "entropia", "desacordo", "meio", "decisao"]
N_BOOT = 2000


def sinais_de(p: np.ndarray, pk: np.ndarray | None = None) -> dict:
    ordenado = np.sort(p, 1)
    s = {
        "confianca": 1 - p.max(1),
        "margem": -(ordenado[:, -1] - ordenado[:, -2]),
        "entropia": M.entropia(p),
        "meio": -np.abs(p[:, M.NEG] - 0.5),
        "decisao": -np.abs(p[:, M.NEG] - np.maximum(p[:, 1], p[:, 2])),
    }
    if pk is not None:
        s["desacordo"] = M.entropia(p) - M.entropia(pk).mean(0)
    return s


def captura(u, alvo, q):
    return M.captura_por_orcamento(u, alvo, [q])[f"{q:.2f}"]


def bloco(s: dict, erro: np.ndarray, fronteira: np.ndarray) -> dict:
    out = {}
    for nome, u in s.items():
        cob, risco, aurc = M.risco_cobertura(u, erro)
        out[nome] = {
            "auroc_erro": M.auroc(u, erro),
            "auroc_fronteira": M.auroc(u, fronteira),
            "aurc": aurc,
            "captura_erro": M.captura_por_orcamento(u, erro, ORCAMENTOS),
            "captura_fronteira": M.captura_por_orcamento(u, fronteira, ORCAMENTOS),
            "risco_cobertura_90": float(risco[int(0.9 * len(risco)) - 1]),
            "risco_cobertura_80": float(risco[int(0.8 * len(risco)) - 1]),
        }
    return out


def ic_captura(s: dict, fronteira: np.ndarray, grupos: np.ndarray, q=0.10) -> dict:
    """IC por video da captura a q, e da diferenca de cada sinal para a confianca.

    O orcamento e reaplicado dentro de cada reamostra (os q% mais incertos DELA), como
    a interface faria numa listagem nova.
    """
    out = {}
    for nome in ("decisao", "meio", "confianca", "entropia"):
        out[nome] = M.bootstrap_agrupado(grupos, lambda i: captura(s[nome][i], fronteira[i], q), N_BOOT)
        if nome != "confianca":
            out[f"{nome}-confianca"] = M.bootstrap_agrupado(
                grupos,
                lambda i: captura(s[nome][i], fronteira[i], q) - captura(s["confianca"][i], fronteira[i], q),
                N_BOOT,
            )
    return out


def protocolo_g() -> dict:
    caminho = RESULTADOS / "logits_agrupado.npz"
    if not caminho.exists():
        return {}
    d = np.load(caminho, allow_pickle=True)
    y = d["rotulos"]
    out = {}
    for e in ("video", "serie"):
        if f"logits__{e}" not in d.files:
            continue
        lg, fold = d[f"logits__{e}"], d[f"fold__{e}"]
        p = np.zeros_like(lg, dtype=np.float64)
        for k in np.unique(fold):
            fora = fold != k
            p[fold == k] = M.softmax(lg[fold == k], M.ajusta_temperatura(lg[fora], y[fora]))
        pred = p.argmax(1)
        erro, fronteira = pred != y, M.erro_fronteira(y, pred)
        s = sinais_de(p)
        grupos = d["grupo_video"] if e == "video" else d["grupo_serie"]
        out[e] = {
            "n": int(len(y)),
            "n_erros": int(erro.sum()),
            "n_erros_fronteira": int(fronteira.sum()),
            "sinais": bloco(s, erro, fronteira),
            "ic_captura_10": ic_captura(s, fronteira, grupos),
        }
        v = out[e]["sinais"]
        print(f"G/{e}: captura fronteira@10% decisao {v['decisao']['captura_fronteira']['0.10']:.3f} "
              f"meio {v['meio']['captura_fronteira']['0.10']:.3f} confianca {v['confianca']['captura_fronteira']['0.10']:.3f}")
    return out


def main():
    prepara_saidas()
    from treino import carrega_corpus

    d = np.load(RESULTADOS / "logits_referencia.npz", allow_pickle=True)
    cal = json.load(open(RESULTADOS / "calibracao.json", encoding="utf-8"))
    t = cal["referencia"]["temperatura"]
    itens = carrega_corpus()
    idx = d["idx_teste"]
    y = np.array([itens[i]["rotulo"] for i in idx])
    grupos = np.array([itens[i]["grupo_video"] for i in idx])

    pk = M.softmax(d["logits_teste"], t)
    p = pk.mean(0)
    pred = p.argmax(1)
    erro = pred != y
    fronteira = M.erro_fronteira(y, pred)
    s = sinais_de(p, pk)

    res = {
        "temperatura": t,
        "n": int(len(y)),
        "n_erros": int(erro.sum()),
        "n_erros_fronteira": int(fronteira.sum()),
        "erros_fronteira_neg_para_nao": int(((y == M.NEG) & (pred != M.NEG)).sum()),
        "erros_fronteira_nao_para_neg": int(((y != M.NEG) & (pred == M.NEG)).sum()),
        "rebaixados_com_pneg_abaixo_de_meio": int(((pred == M.NEG) & (p[:, M.NEG] < 0.5)).sum()),
        "pneg_min": float(p[:, M.NEG].min()),
        "pneg_max": float(p[:, M.NEG].max()),
        "sinais": bloco(s, erro, fronteira),
        "ic_captura_10": ic_captura(s, fronteira, grupos),
        "risco_sem_deferencia": float(erro.mean()),
        "agrupado": protocolo_g(),
    }
    for nome, v in res["sinais"].items():
        print(f"{nome:10s} AUROC erro {v['auroc_erro']:.3f} fronteira {v['auroc_fronteira']:.3f} "
              f"AURC {v['aurc']:.4f} captura fronteira {v['captura_fronteira']}")
    print("IC", {k: [round(a, 3) for a in v] for k, v in res["ic_captura_10"].items()})

    with open(RESULTADOS / "deferencia.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    figura(s, erro, fronteira, "en")
    figura(s, erro, fronteira, "pt")


def figura(s, erro, fronteira, idioma: str = "en") -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType: o PDF nao leva fonte Type 3
    import matplotlib.pyplot as plt

    from matplotlib.ticker import FuncFormatter

    pt = idioma == "pt"
    rotulos = {
        "confianca": ("Confiança" if pt else "Confidence"),
        "margem": ("Margem" if pt else "Margin"),
        "entropia": ("Entropia" if pt else "Entropy"),
        "desacordo": ("Desacordo do ensemble" if pt else "Ensemble disagreement"),
        "meio": ("$P(\\mathrm{Neg})$ perto de 1/2" if pt else "$P(\\mathrm{Neg})$ near 1/2"),
        "decisao": ("Distância à decisão de rebaixar" if pt else "Distance to demotion decision"),
    }
    cores = {
        "confianca": ("#b2182b", "-"), "margem": ("#ef8a62", "--"), "entropia": ("#999999", "-."),
        "desacordo": ("#67a9cf", ":"), "meio": ("#5aae61", "--"), "decisao": ("#2166ac", "-"),
    }
    estilo = {k: (rotulos[k], *cores[k]) for k in rotulos}
    # largura de uma coluna do template (3,4 pol.), em escala 1:1: o texto sai no tamanho declarado
    fig, eixos = plt.subplots(2, 1, figsize=(3.4, 5.0))
    ax = eixos[0]
    for nome in SINAIS:
        cob, risco, _ = M.risco_cobertura(s[nome], erro)
        rot, cor, ls = estilo[nome]
        ax.plot(cob, risco, c=cor, ls=ls, lw=1.2, label=rot)
    ax.axhline(erro.mean(), c="0.5", lw=0.8, ls=":")
    ax.set_xlabel("Cobertura (fração de itens não marcados)" if pt else "Coverage (share of items not flagged)", fontsize=7.5)
    ax.set_ylabel("Erro nos não marcados" if pt else "Error rate, unflagged", fontsize=7.5)
    ax.set_title("Risco–cobertura" if pt else "Risk–coverage", fontsize=8)
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0, 0.32)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.5)

    ax = eixos[1]
    q = np.linspace(0, 0.5, 51)
    for nome in SINAIS:
        rot, cor, ls = estilo[nome]
        ordem = np.argsort(-s[nome], kind="mergesort")
        f = fronteira[ordem]
        cap = [f[: int(round(x * len(f)))].sum() / f.sum() for x in q]
        ax.plot(q, cap, c=cor, ls=ls, lw=1.2, label=rot)
    ax.plot(q, q, c="0.5", lw=0.8, ls=":", label="Marca aleatória" if pt else "Random flagging")
    ax.set_xlabel("Orçamento de conferência (fração marcada)" if pt else "Review budget (share of items flagged)", fontsize=7.5)
    ax.set_ylabel("Erros de fronteira capturados" if pt else "Boundary errors captured", fontsize=7.5)
    ax.set_title("Erros caros enviados ao responsável" if pt else "Costly errors sent to the caregiver", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.5)
    # legenda abaixo do painel: dentro dele, encobria as curvas
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, -0.24), ncol=2, frameon=False,
              handlelength=2.2, columnspacing=1.0)
    fig.tight_layout()
    if pt:
        virgula = FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))
        for a in eixos:
            a.xaxis.set_major_formatter(virgula)
            a.yaxis.set_major_formatter(virgula)
    fig.savefig((FIGURAS_PT if pt else FIGURAS) / "fig-deferencia.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()

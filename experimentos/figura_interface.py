"""Figuras ILUSTRATIVAS da interface, com probabilidades SIMULADAS.

Nao sao capturas de tela nem saida do sistema. Explicam ao leitor como as tres leituras
(posicao, cor, percentual) sao calculadas pelas formulas atuais, nas duas consultas da
versao de conferencia ("Bob Esponja" e "South Park"). As miniaturas sao as dos videos que
aparecem nas capturas publicadas na versao de conferencia; os titulos sao marcadores e as
probabilidades foram escolhidas a mao para cobrir os casos que o artigo discute
(Positivo, Neutro, Negativo confiante e uma decisao apertada). Por isso a figura traz,
DENTRO dela, o aviso de que os valores sao simulados: nenhum valor e o que o sistema
calculou para aquele video.

Classe, confianca C, adequacao A, Safety Score S e a nova ordem saem das equacoes do artigo:

    A = P(Pos) + 0.5 P(Neu)
    S = 0.10 + 0.20 (1 - C)   se a classe e Negativo
        0.70 + 0.15 A         caso contrario

As miniaturas (latex/figuras/miniaturas/) sao recortadas das capturas da versao de
conferencia por recorta_miniaturas(), quando a pasta daquele artigo esta ao lado. Sao
imagens de terceiros e ficam FORA de src/ (nao vao ao repositorio publico); sem elas, a
figura sai com marcadores cinza no lugar.

Saidas (ingles e portugues):
  figuras/fig-ilustracao-listagem.pdf   listagens das duas consultas, como a interface atual as mostra
  figuras/fig-cartao-proposto.pdf       o mesmo item no cartao atual e no cartao proposto (D1--D3)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType: o PDF nao leva fonte Type 3
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from caminhos import BASE, FIGURAS, FIGURAS_PT, prepara_saidas

MINIATURAS = FIGURAS / "miniaturas"
CAPTURAS_WICS = BASE.parents[1] / "2026--wics--youtubesafekids-sentimentos" / "latex" / "figuras"
# (arquivo da captura, prefixo, faixas x dos quatro cartoes, y do topo dos cartoes)
RECORTES = [
    ("fig06-lista-score-alto.png", "bob", [(20, 411), (425, 816), (830, 1221), (1235, 1626)], 65),
    ("fig08-lista-score-baixo.png", "south", [(5, 394), (410, 799), (815, 1204), (1220, 1609)], 13),
]

# (posicao devolvida pela plataforma, P(Neg), P(Neu), P(Pos)) -- valores simulados.
# A miniatura do resultado k e a do k-esimo cartao da captura correspondente.
CONSULTAS = {
    "Bob Esponja": ("bob", [(1, 0.02, 0.55, 0.43), (2, 0.01, 0.09, 0.90), (3, 0.05, 0.25, 0.70), (4, 0.03, 0.80, 0.17)]),
    "South Park": ("south", [(1, 0.92, 0.06, 0.02), (2, 0.10, 0.75, 0.15), (3, 0.97, 0.02, 0.01), (4, 0.48, 0.40, 0.12)]),
}
CLASSES = ["Neg", "Neu", "Pos"]
COR = {"Neg": "#d73027", "Neu": "#e6b800", "Pos": "#1a9850"}
LIMIAR_APERTADO = 0.15  # distancia a decisao de rebaixar abaixo da qual o cartao proposto pede conferencia
ASPECTO_MINIATURA = 16 / 9

TXT = {
    "en": {
        "consulta": "Query", "miniatura": "thumbnail", "resultado": "Result",
        "orig": "rank", "rebaixado": "demoted",
        "atual": "Current card", "proposto": "Proposed card (D1–D3)",
        "sinal": "Emotionally negative\npassages", "faixa": "Suitability: uncertain",
        "confira": "Check this: close decision", "nota": "demoted for tone,\nnot for verified harm",
        "aviso": "Illustration — simulated values, not system output",
    },
    "pt": {
        "consulta": "Consulta", "miniatura": "miniatura", "resultado": "Resultado",
        "orig": "posição", "rebaixado": "rebaixado",
        "atual": "Cartão atual", "proposto": "Cartão proposto (D1–D3)",
        "sinal": "Passagens emocionalmente\nnegativas", "faixa": "Adequação: incerta",
        "confira": "Confira este: decisão apertada", "nota": "rebaixado pelo tom,\nnão por dano verificado",
        "aviso": "Ilustração — valores simulados, não é saída do sistema",
    },
}


def recorta_miniaturas() -> None:
    """Recorta as miniaturas das capturas da versao de conferencia, se ela estiver ao lado."""
    from PIL import Image

    if not CAPTURAS_WICS.is_dir():
        return
    MINIATURAS.mkdir(parents=True, exist_ok=True)
    for arquivo, prefixo, cartoes, topo in RECORTES:
        origem = CAPTURAS_WICS / arquivo
        if not origem.exists():
            continue
        img = Image.open(origem).convert("RGB")
        px = img.load()
        for k, (x0, x1) in enumerate(cartoes, start=1):
            # a miniatura termina na primeira linha em que o miolo do cartao fica branco
            cx0, cx1 = x0 + (x1 - x0) // 10, x1 - (x1 - x0) // 10
            y = topo + 5
            while y < img.height and min(min(px[x, y]) for x in range(cx0, cx1, 4)) <= 240:
                y += 1
            img.crop((x0 + 1, topo + 1, x1 - 1, y)).save(MINIATURAS / f"{prefixo}-{k}.png")


def miniatura(prefixo: str, k: int):
    caminho = MINIATURAS / f"{prefixo}-{k}.png"
    return plt.imread(str(caminho)) if caminho.exists() else None


def leituras(pneg, pneu, ppos):
    p = [pneg, pneu, ppos]
    c = max(range(3), key=lambda i: p[i])
    conf = p[c]
    a = ppos + 0.5 * pneu
    s = 0.10 + 0.20 * (1 - conf) if c == 0 else 0.70 + 0.15 * a
    decisao = abs(pneg - max(pneu, ppos))
    return CLASSES[c], conf, a, s, decisao


def fmt(x, casas, idioma):
    s = f"{x:.{casas}f}"
    return s.replace(".", ",") if idioma == "pt" else s


def pct_interface(a: float) -> int:
    """Como o frontend arredonda (Math.round): meio para cima, nao o arredondamento bancario."""
    return int(100 * a + 0.5)


class Escala:
    """Converte larguras em altura nos eixos, para a miniatura manter 16:9."""

    def __init__(self, fig, ax):
        largura_pol, altura_pol = fig.get_size_inches()
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        self.pol_por_x = largura_pol / (x1 - x0)
        self.pol_por_y = altura_pol / (y1 - y0)

    def altura_para(self, largura: float, aspecto: float) -> float:
        return largura * self.pol_por_x / aspecto / self.pol_por_y


def desenha_miniatura(ax, esc, x, y_topo, w, img, T):
    """Miniatura 16:9 com o topo em y_topo; devolve o y da base."""
    largura = w - 0.02
    altura = esc.altura_para(largura, ASPECTO_MINIATURA)
    base = y_topo - altura
    if img is not None:
        ax.imshow(img, extent=(x + 0.01, x + 0.01 + largura, base, y_topo), aspect="auto", zorder=2)
    else:
        ax.add_patch(Rectangle((x + 0.01, base), largura, altura, fc="0.88", ec="none", zorder=2))
        ax.text(x + w / 2, base + altura / 2, T["miniatura"], ha="center", va="center", fontsize=6.3,
                color="0.55", style="italic", zorder=3)
    return base


def cartao(ax, esc, x, y, w, h, titulo, classe, a, linhas, img=None, destaque=None, T=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc="white", ec="0.6", lw=0.9, zorder=1))
    base = desenha_miniatura(ax, esc, x, y + h - 0.012, w, img, T)
    resto = base - y  # espaco abaixo da miniatura
    ax.text(x + 0.012, base - 0.16 * resto, titulo, ha="left", va="center", fontsize=7.0, weight="bold")
    bw, bh = 0.37 * w, 0.17 * resto
    yb = base - 0.40 * resto
    ax.add_patch(FancyBboxPatch((x + 0.012, yb - bh / 2), bw, bh,
                                boxstyle="round,pad=0.004,rounding_size=0.01", fc=COR[classe], ec="none"))
    ax.text(x + 0.012 + bw / 2, yb, f"Score: {pct_interface(a)}%", ha="center", va="center",
            fontsize=6.2, color="white", weight="bold")
    if destaque:
        ax.text(x + w - 0.012, yb, destaque, ha="right", va="center", fontsize=6.0, color="#d73027", style="italic")
    for k, ln in enumerate(linhas):
        ax.text(x + 0.012, base - (0.61 + 0.15 * k) * resto, ln, ha="left", va="center", fontsize=5.6, color="0.25")


def aviso(fig, T):
    fig.text(0.995, 0.995, T["aviso"], ha="right", va="top", fontsize=6.2, color="#b2182b", style="italic")


def figura_listagem(idioma: str) -> None:
    T = TXT[idioma]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.97, bottom=0.005)
    esc = Escala(fig, ax)
    w, h, gap = 0.232, 0.42, 0.02
    for linha, (consulta, (prefixo, itens)) in enumerate(CONSULTAS.items()):
        y0 = 0.505 - linha * 0.5
        ax.text(0.0, y0 + h + 0.03, f"{T['consulta']}: “{consulta}”", ha="left", va="center", fontsize=8, weight="bold")
        calculados = []
        for orig, pn, pu, pp in itens:
            classe, conf, a, s, _ = leituras(pn, pu, pp)
            calculados.append((s, orig, classe, conf, a, (pn, pu, pp)))
        calculados.sort(key=lambda t: -t[0])
        for nova, (s, orig, classe, conf, a, p) in enumerate(calculados, start=1):
            x = (nova - 1) * (w + gap)
            pstr = " / ".join(fmt(v, 2, idioma) for v in p)
            linhas = [
                f"P(Neg, Neu, Pos) = {pstr}",
                f"C = {fmt(conf, 2, idioma)}  A = {fmt(a, 2, idioma)}  S = {fmt(s, 3, idioma)}",
                f"{T['orig']} {orig} → {nova}",
            ]
            cartao(ax, esc, x, y0, w, h, f"{T['resultado']} {orig}", classe, a, linhas,
                   img=miniatura(prefixo, orig), destaque=T["rebaixado"] if classe == "Neg" else None, T=T)
    aviso(fig, T)
    fig.savefig((FIGURAS_PT if idioma == "pt" else FIGURAS) / "fig-ilustracao-listagem.pdf", dpi=300)
    plt.close(fig)


def figura_cartao(idioma: str) -> None:
    """O resultado 4 de "South Park": Negativo por pouco, o caso que D1--D3 tratam."""
    T = TXT[idioma]
    prefixo, itens = CONSULTAS["South Park"]
    orig, pn, pu, pp = itens[3]
    classe, conf, a, s, decisao = leituras(pn, pu, pp)
    assert classe == "Neg" and decisao < LIMIAR_APERTADO
    img = miniatura(prefixo, orig)

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.005)
    esc = Escala(fig, ax)
    w, h, y0 = 0.45, 0.86, 0.02
    pstr = " / ".join(fmt(v, 2, idioma) for v in (pn, pu, pp))

    # cartao atual
    ax.text(0.02 + w / 2, y0 + h + 0.045, T["atual"], ha="center", va="center", fontsize=7.5, weight="bold")
    cartao(ax, esc, 0.02, y0, w, h, f"{T['resultado']} {orig}", classe, a, [f"P = {pstr}"], img=img, T=T)

    # cartao proposto
    x = 0.53
    ax.text(x + w / 2, y0 + h + 0.045, T["proposto"], ha="center", va="center", fontsize=7.5, weight="bold")
    ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc="white", ec=COR["Neg"], lw=1.1, zorder=1))
    topo = y0 + h - 0.012
    base = desenha_miniatura(ax, esc, x, topo, w, img, T)
    ax.add_patch(FancyBboxPatch((x + 0.02, topo - 0.075), 0.22, 0.055,
                                boxstyle="round,pad=0.004,rounding_size=0.02", fc="#fdae61", ec="none", zorder=4))
    ax.text(x + 0.13, topo - 0.0475, T["confira"].split(":")[0], ha="center", va="center", fontsize=6.0,
            weight="bold", zorder=5)
    resto = base - y0
    ax.text(x + 0.012, base - 0.13 * resto, f"{T['resultado']} {orig}", ha="left", va="center", fontsize=7.0, weight="bold")
    ax.text(x + 0.012, base - 0.36 * resto, T["sinal"], ha="left", va="center", fontsize=5.6,
            color=COR["Neg"], weight="bold", linespacing=1.05)
    ax.text(x + 0.012, base - 0.57 * resto, T["faixa"], ha="left", va="center", fontsize=5.6, color="0.25")
    ax.text(x + 0.012, base - 0.70 * resto, T["confira"], ha="left", va="center", fontsize=5.4, color="#b35806")
    ax.text(x + 0.012, base - 0.87 * resto, T["nota"], ha="left", va="center", fontsize=4.9, color="0.35",
            style="italic", linespacing=1.0)
    # O que muda em relacao ao cartao atual, com o numero da diretriz da Secao 6
    for rotulo, y in (("D1", base - 0.36 * resto), ("D2", base - 0.57 * resto), ("D3", base - 0.70 * resto),
                      ("D3", topo - 0.0475)):
        ax.text(x + w - 0.014, y, rotulo, ha="right", va="center", fontsize=5.0, weight="bold", color="white",
                zorder=6, bbox=dict(boxstyle="round,pad=0.18", fc="0.3", ec="none"))
    aviso(fig, T)
    fig.savefig((FIGURAS_PT if idioma == "pt" else FIGURAS) / "fig-cartao-proposto.pdf", dpi=300)
    plt.close(fig)


def main():
    prepara_saidas()
    recorta_miniaturas()
    for idioma in ("en", "pt"):
        figura_listagem(idioma)
        figura_cartao(idioma)
    for consulta, (_, itens) in CONSULTAS.items():
        for orig, pn, pu, pp in itens:
            classe, conf, a, s, d = leituras(pn, pu, pp)
            print(f"{consulta:12s} #{orig} {classe} C={conf:.2f} A={a:.3f} S={s:.3f} dist={d:.2f}")


if __name__ == "__main__":
    main()

"""JSON de resultados -> numeros.tex e tabelas/*.tex, em ingles e em portugues.

Nenhum numero do manuscrito e digitado: todos saem daqui. Para corrigir um valor no
texto, corrige-se o script que o produz, roda-se de novo e recompila-se.

Dois alvos, com os MESMOS numeros:
  en  latex/numeros.tex, latex/tabelas/   -- o manuscrito submetido
  pt  latex/pt/numeros.tex, latex/pt/tabelas/  -- versao para conferencia do autor

Formatacao. Ingles: ponto decimal, virgula de milhar, intervalos "[a, b]". Portugues:
virgula decimal, ponto de milhar, intervalos "[a; b]" (a virgula ja e o separador
decimal). F1, acuracia, recall e ECE em porcentagem com uma casa; Brier, NLL e AUROC em
fracao com tres casas; temperatura com duas. Cada legenda e cabecalho vem em par
tr(ingles, portugues), no lugar onde e usado.

Uso: python gerar_numeros.py   (depois de todas as analises)
"""

from __future__ import annotations

import json
import time

from caminhos import LATEX, LATEX_PT, RESULTADOS, TABELAS, TABELAS_PT, prepara_saidas

IDIOMA = "en"  # trocado por main() a cada passada


def tr(en: str, pt: str) -> str:
    return pt if IDIOMA == "pt" else en


def CAT(c):
    return {
        "adulta": tr("Adult animation", "Animação adulta"),
        "classica": tr("Classic slapstick", "Desenho clássico"),
        "narrativa": tr("Children's entertainment", "Entretenimento infantil"),
        "educativa": tr("Children's educational", "Educativo infantil"),
    }[c]


def ESQ(e):
    return {
        "frase": tr("By sentence", "Por frase"),
        "video": tr("By video", "Por vídeo"),
        "serie": tr("By series", "Por série"),
    }[e]


def VAR(v):
    return {
        "limpo": tr("Curated sentence (reference)", "Frase revisada (referência)"),
        "legenda": tr("Caption format", "Formato de legenda"),
        "legenda+pal10": tr("Caption + 10\\% words dropped", "Legenda + 10\\% das palavras perdidas"),
        "legenda+pal20": tr("Caption + 20\\% words dropped", "Legenda + 20\\% das palavras perdidas"),
        "legenda+let05": tr("Caption + 5\\% letters corrupted", "Legenda + 5\\% das letras corrompidas"),
        "legenda+let10": tr("Caption + 10\\% letters corrupted", "Legenda + 10\\% das letras corrompidas"),
    }[v]


def SINAL(s):
    return {
        "confianca": tr("Confidence", "Confiança"),
        "margem": tr("Margin", "Margem"),
        "entropia": tr("Entropy", "Entropia"),
        "desacordo": tr("Ensemble disagreement", "Desacordo do ensemble"),
        "meio": tr("$P(\\text{Neg})$ near $\\frac12$", "$P(\\text{Neg})$ perto de $\\frac12$"),
        "decisao": tr("Distance to demotion decision", "Distância à decisão de rebaixar"),
    }[s]


# ------------------------------------------------------------------- formatacao


def dec(x: float, casas: int) -> str:
    s = f"{x:,.{casas}f}"
    if IDIOMA == "pt":
        s = s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return s


def pct(x, casas=1):
    return "--" if x is None else f"{dec(100 * x, casas)}\\%"


def pnum(x, casas=1):
    """Porcentagem sem o sinal, para tabelas com a unidade no cabecalho."""
    return "--" if x is None else dec(100 * x, casas)


def num(x, casas=3):
    return "--" if x is None else dec(x, casas)


def inteiro(n):
    return dec(int(n), 0)


def ic(par, casas=1):
    sep = "; " if IDIOMA == "pt" else ", "
    return f"[{dec(100 * par[0], casas)}{sep}{dec(100 * par[1], casas)}]"


class Macros:
    def __init__(self):
        self.linhas = []
        self.nomes = set()

    def __call__(self, nome, valor):
        assert nome.isalpha(), nome
        assert nome not in self.nomes, f"macro repetida: {nome}"
        self.nomes.add(nome)
        self.linhas.append(f"\\newcommand{{\\{nome}}}{{{valor}}}")


def carrega(nome):
    caminho = RESULTADOS / nome
    return json.load(open(caminho, encoding="utf-8")) if caminho.exists() else None


def letras(s: str) -> str:
    """Nome de macro so com letras: 'legenda+pal10' -> 'LegendaPaldez'."""
    trad = {"05": "Cinco", "10": "Dez", "20": "Vinte"}
    for k, v in trad.items():
        s = s.replace(k, v)
    return "".join(p.capitalize() for p in s.replace("+", " ").replace("_", " ").replace("-", " ").split())


def grava_tabela(nome: str, linhas: list[str]) -> None:
    destino = (TABELAS_PT if IDIOMA == "pt" else TABELAS) / nome
    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")


CABECALHO = "% GERADO por src/experimentos/gerar_numeros.py -- nao editar a mao."


# ------------------------------------------------------------------- corpus


def macros_corpus(m: Macros, corpus: dict) -> None:
    m("corpusTotal", inteiro(corpus["total"]))
    m("corpusTitulos", inteiro(corpus["n_titulos"]))
    m("corpusFranquias", inteiro(corpus["n_franquias"]))
    m("corpusVideos", inteiro(corpus["n_videos"]))
    rc = corpus["rotulos_por_categoria"]
    total_neg = sum(v.get("Negativo", 0) for v in rc.values())
    for c, v in rc.items():
        nome = letras(c)
        n = sum(v.values())
        m(f"cat{nome}N", inteiro(n))
        m(f"cat{nome}Neg", inteiro(v.get("Negativo", 0)))
        m(f"cat{nome}NegPct", pct(v.get("Negativo", 0) / n))
    adulto_classico = rc["adulta"].get("Negativo", 0) + rc["classica"].get("Negativo", 0)
    m("negTotal", inteiro(total_neg))
    m("negAdultoClassicoPct", pct(adulto_classico / total_neg))
    infantil_neg = rc["narrativa"].get("Negativo", 0) + rc["educativa"].get("Negativo", 0)
    infantil_n = sum(rc["narrativa"].values()) + sum(rc["educativa"].values())
    m("negInfantilN", inteiro(infantil_neg))
    m("infantilN", inteiro(infantil_n))
    m("negInfantilPct", pct(infantil_neg / infantil_n))

    # Tabela 1: categoria x rotulo
    lin = [
        CABECALHO,
        "\\begin{table}[!t]",
        "\\caption{"
        + tr(
            "Composition of the corpus by content category, with the human label distribution."
            " Category is derived from the video title by explicit rules (\\texttt{fontes.py}).",
            "Composição do corpus por categoria de conteúdo, com a distribuição dos rótulos"
            " humanos. A categoria é derivada do título do vídeo por regras explícitas"
            " (\\texttt{fontes.py}).",
        )
        + "}\\label{tab:corpus}",
        "\\centering\\footnotesize\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\hline\\hline",
        tr("Category & Sentences & Negative & Neutral & Positive \\\\", "Categoria & Frases & Negativo & Neutro & Positivo \\\\"),
        "\\hline",
    ]
    for c in ("adulta", "classica", "narrativa", "educativa"):
        v = rc[c]
        lin.append(f"{CAT(c)} & {inteiro(sum(v.values()))} & {inteiro(v.get('Negativo', 0))} & {inteiro(v.get('Neutro', 0))} & {inteiro(v.get('Positivo', 0))} \\\\")
    soma = {r: sum(v.get(r, 0) for v in rc.values()) for r in ("Negativo", "Neutro", "Positivo")}
    lin += [
        "\\hline",
        f"Total & {inteiro(corpus['total'])} & {inteiro(soma['Negativo'])} & {inteiro(soma['Neutro'])} & {inteiro(soma['Positivo'])} \\\\",
        "\\hline\\hline", "\\end{tabular}", "\\end{table}",
    ]
    grava_tabela("tab-corpus.tex", lin)


# ------------------------------------------------------------- generalizacao


def macros_generalizacao(m: Macros, g: dict) -> None:
    for e, b in g["esquemas"].items():
        E = letras(e)
        m(f"gen{E}FUmMacro", pct(b["f1_macro"]))
        m(f"gen{E}FUmMacroIC", ic(b["ic_f1_macro"]))
        m(f"gen{E}RecallNeg", pct(b["recall_neg"]))
        m(f"gen{E}RecallNegIC", ic(b["ic_recall_neg"]))
        m(f"gen{E}FUmNeg", pct(b["f1_neg"]))
        m(f"gen{E}Acuracia", pct(b["acuracia"]))
        m(f"gen{E}Fronteira", pct(b["erro_fronteira"]))
        m(f"gen{E}FUmNeu", pct(b["f1_neu"]))
        m(f"gen{E}FUmPos", pct(b["f1_pos"]))
        for c, v in b["por_categoria"].items():
            C = letras(c)
            m(f"gen{E}{C}PredNeg", pct(v["taxa_pred_neg"]))
            m(f"gen{E}{C}FalsoNeg", pct(v["falso_neg"]))
            m(f"gen{E}{C}FalsoNegN", inteiro(v["n_falso_neg"]))
            if v["recall_neg"] is not None:
                m(f"gen{E}{C}RecallNeg", pct(v["recall_neg"]))
            if v["precisao_neg"] is not None:
                m(f"gen{E}{C}PrecisaoNeg", pct(v["precisao_neg"]))
            m(f"gen{E}{C}NPredNeg", inteiro(v["n_pred_neg"]))
            m(f"gen{E}{C}NAcertoNeg", inteiro(v["n_acerto_neg"]))
            m(f"gen{E}{C}NNeg", inteiro(v["n_neg"]))
            m(f"gen{E}{C}N", inteiro(v["n"]))
        for c, v in g["rebaixamento_videos"][e].items():
            if c.startswith("_"):
                continue
            C = letras(c)
            m(f"vid{E}{C}N", inteiro(v["n_videos"]))
            m(f"vid{E}{C}Oraculo", inteiro(v["rebaixa_oraculo"]))
            m(f"vid{E}{C}Modelo", inteiro(v["rebaixa_modelo"]))
            m(f"vid{E}{C}SoModelo", inteiro(v["rebaixa_modelo_nao_oraculo"]))
            m(f"vid{E}{C}SoOraculo", inteiro(v["oraculo_nao_modelo"]))
            m(f"vid{E}{C}NaoOraculo", inteiro(v["n_videos"] - v["rebaixa_oraculo"]))
        # Agregados do nivel do video: concordancia da decisao com o oraculo e o
        # conteudo infantil (entretenimento + educativo) rebaixado so pelo modelo
        cats = {c: v for c, v in g["rebaixamento_videos"][e].items() if not c.startswith("_")}
        total = sum(v["n_videos"] for v in cats.values())
        discorda = sum(v["rebaixa_modelo_nao_oraculo"] + v["oraculo_nao_modelo"] for v in cats.values())
        infantis = [cats[c] for c in ("narrativa", "educativa")]
        m(f"vid{E}Total", inteiro(total))
        m(f"vid{E}Concorda", inteiro(total - discorda))
        m(f"vid{E}InfantilN", inteiro(sum(v["n_videos"] for v in infantis)))
        m(f"vid{E}InfantilSoModelo", inteiro(sum(v["rebaixa_modelo_nao_oraculo"] for v in infantis)))
    for k, v in g["diferencas"].items():
        K = letras(k.replace("-", " "))
        m(f"dif{K}FUm", dec(100 * v["delta_f1_macro"], 1))
        m(f"dif{K}FUmIC", ic(v["ic"]))
        m(f"dif{K}RecallNeg", dec(100 * v["delta_recall_neg"], 1))
    pr = g["protocolo"]
    m("protGruposVideo", inteiro(pr["n_grupos_video"]))
    m("protGruposSerie", inteiro(pr["n_grupos_serie"]))
    m("protSeriesMulti", inteiro(pr["n_series_multivideo"]))
    m("protFrasesSeriesMulti", inteiro(pr["frases_series_multivideo"]))
    m("protSerieFoldMin", inteiro(pr["serie_fold_min"]))
    m("protSerieFoldMax", inteiro(pr["serie_fold_max"]))
    m("protSerieFoldNegMin", pct(pr["serie_fold_neg_min"], 0))
    m("protSerieFoldNegMax", pct(pr["serie_fold_neg_max"], 0))
    m("protSerieTreinoMin", inteiro(pr["serie_treino_min"]))
    for c, v in g["esquemas"]["video"]["por_categoria"].items():
        m(f"genVideo{letras(c)}TaxaRotuloNeg", pct(v["taxa_rotulo_neg"]))

    # Tabela 2: esquemas
    n_frases = inteiro(g["esquemas"]["frase"]["n"])
    lin = [
        CABECALHO,
        "\\begin{table*}[!t]",
        "\\caption{"
        + tr(
            f"Out-of-fold performance of the same classifier recipe under three partition schemes"
            f" (5 folds, {n_frases} sentences, one model per fold). Values in \\%; brackets are 95\\%"
            " intervals from a bootstrap that resamples videos (series, in the by-series row), not"
            " sentences. ``Boundary errors'' cross the risk boundary (Negative taken for"
            " non-Negative or the reverse).",
            f"Desempenho fora da amostra da mesma receita de classificador em três esquemas de"
            f" partição (5 folds, {n_frases} frases, um modelo por fold). Valores em \\%; entre"
            " colchetes, intervalos de 95\\% por bootstrap que reamostra vídeos (séries, na linha"
            " por série), e não frases. ``Erros de fronteira'' cruzam a fronteira de risco"
            " (Negativo tomado por não-Negativo ou o inverso).",
        )
        + "}\\label{tab:generalizacao}",
        "\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}lcccccc@{}}",
        "\\hline\\hline",
        tr(
            "Partition & Macro F1 & Recall Neg. & F1 Neg. & F1 Neu. & F1 Pos. & Boundary err. \\\\",
            "Partição & F1 macro & Recall Neg. & F1 Neg. & F1 Neu. & F1 Pos. & Erro fronteira \\\\",
        ),
        "\\hline",
    ]
    for e in ("frase", "video", "serie"):
        b = g["esquemas"].get(e)
        if not b:
            continue
        lin.append(
            f"{ESQ(e)} & {pnum(b['f1_macro'])} {ic(b['ic_f1_macro'])} & {pnum(b['recall_neg'])} {ic(b['ic_recall_neg'])}"
            f" & {pnum(b['f1_neg'])} & {pnum(b['f1_neu'])} & {pnum(b['f1_pos'])} & {pnum(b['erro_fronteira'])} \\\\"
        )
    lin += ["\\hline\\hline", "\\end{tabular}", "\\end{table*}"]
    grava_tabela("tab-generalizacao.tex", lin)

    # Tabela 3: quem e rebaixado, esquema por video
    e = "video"
    b, rv = g["esquemas"][e], g["rebaixamento_videos"][e]
    lin = [
        CABECALHO,
        "\\begin{table*}[!t]",
        "\\caption{"
        + tr(
            "What the sentiment signal demotes, by content category, under the by-video partition."
            " Sentence level: share of sentences predicted Negative, precision and recall of those"
            " predictions (only where there are at least ten), and share of non-Negative sentences"
            " predicted Negative (undue demotions). Video level: a video is demoted when the mean of"
            " its sentence probabilities favours Negative; the oracle applies the same rule to the"
            " human labels, i.e.\\ it is what a perfect sentiment classifier would demote.",
            "O que o sinal de sentimento rebaixa, por categoria de conteúdo, na partição por vídeo."
            " Nível da frase: fração de frases preditas Negativo, precisão e recall dessas predições"
            " (só onde há pelo menos dez) e fração das frases não-Negativas preditas Negativo"
            " (rebaixamentos indevidos). Nível do vídeo: um vídeo é rebaixado quando a média das"
            " probabilidades das suas frases favorece Negativo; o oráculo aplica a mesma regra aos"
            " rótulos humanos, isto é, mostra o que um classificador de sentimento perfeito"
            " rebaixaria.",
        )
        + "}\\label{tab:fontes}",
        "\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}lrrrrrrrrr@{}}",
        "\\hline\\hline",
        tr(
            " & \\multicolumn{5}{c}{Sentences} & \\multicolumn{4}{c}{Videos} \\\\",
            " & \\multicolumn{5}{c}{Frases} & \\multicolumn{4}{c}{Vídeos} \\\\",
        ),
        tr(
            "Category & $n$ & Pred.\\ Neg. & Prec.\\ Neg. & Recall Neg. & Undue Neg. & $n$ & Oracle demotes & Model demotes & Model only \\\\",
            "Categoria & $n$ & Pred.\\ Neg. & Prec.\\ Neg. & Recall Neg. & Neg.\\ indevido & $n$ & Oráculo rebaixa & Modelo rebaixa & Só o modelo \\\\",
        ),
        "\\hline",
    ]
    for c in ("adulta", "classica", "narrativa", "educativa"):
        v, w = b["por_categoria"][c], rv[c]
        lin.append(
            f"{CAT(c)} & {inteiro(v['n'])} & {pct(v['taxa_pred_neg'])} & {pct(v['precisao_neg'])} & {pct(v['recall_neg'])} & {pct(v['falso_neg'])}"
            f" & {inteiro(w['n_videos'])} & {inteiro(w['rebaixa_oraculo'])} & {inteiro(w['rebaixa_modelo'])} & {inteiro(w['rebaixa_modelo_nao_oraculo'])} \\\\"
        )
    lin += ["\\hline\\hline", "\\end{tabular}", "\\end{table*}"]
    grava_tabela("tab-fontes.tex", lin)


# --------------------------------------------------------------- calibracao


def macros_calibracao(m: Macros, c: dict) -> None:
    r = c["referencia"]
    m("calTemp", num(r["temperatura"], 2))
    m("calNVal", inteiro(r["n_validacao"]))
    m("calNTeste", inteiro(r["n_teste"]))
    m("calNVideosTeste", inteiro(r["n_videos_teste"]))
    m("calTempMudaClasse", inteiro(r["temperatura_muda_classe"]))
    m("calTempMudaRebaixa", inteiro(r["temperatura_muda_rebaixamento"]))
    for fase, F in (("teste_antes", "Antes"), ("teste_depois", "Depois")):
        b = r[fase]
        m(f"cal{F}Acuracia", pct(b["acuracia"]))
        m(f"cal{F}EceTopo", pct(b["ece_topo"]))
        m(f"cal{F}EceNeg", pct(b["ece_neg"]))
        m(f"cal{F}EceAdeq", pct(b["ece_adequacao"]))
        m(f"cal{F}Brier", num(b["brier"]))
        m(f"cal{F}Nll", num(b["nll"]))
        m(f"cal{F}ConfMedia", pct(b["conf_media"]))
        m(f"cal{F}FracAlta", pct(b["frac_conf_alta"]))
        m(f"cal{F}NAlta", inteiro(b["n_conf_alta"]))
        m(f"cal{F}AcAlta", pct(b["acuracia_conf_alta"]))
        m(f"cal{F}AcBaixa", pct(b["acuracia_conf_baixa"]))
    for k, v in r["ic"].items():
        m("calIC" + letras(k), ic(v))
    for e, d in c.get("agrupado", {}).items():
        E = letras(e)
        m(f"calG{E}EceAntes", pct(d["antes"]["ece_topo"]))
        m(f"calG{E}EceDepois", pct(d["depois"]["ece_topo"]))
        m(f"calG{E}EceAdeqAntes", pct(d["antes"]["ece_adequacao"]))
        m(f"calG{E}EceAdeqDepois", pct(d["depois"]["ece_adequacao"]))
        m(f"calG{E}AcAlta", pct(d["antes"]["acuracia_conf_alta"]))
        m(f"calG{E}FracAlta", pct(d["antes"]["frac_conf_alta"]))
        m(f"calG{E}Temp", num(sum(d["temperaturas"]) / len(d["temperaturas"]), 2))
        m(f"calG{E}EceTRef", pct(d["com_t_referencia"]["ece_topo"]))
        m(f"calG{E}EceNegAntes", pct(d["antes"]["ece_neg"]))
        m(f"calG{E}EceNegDepois", pct(d["depois"]["ece_neg"]))
        m(f"calG{E}EceNegTRef", pct(d["com_t_referencia"]["ece_neg"]))
        m(f"calG{E}EceAdeqTRef", pct(d["com_t_referencia"]["ece_adequacao"]))

    lin = [
        CABECALHO,
        "\\begin{table*}[!t]",
        "\\caption{"
        + tr(
            f"Calibration of what the interface displays, before and after temperature scaling."
            f" Reference partition: the five-model ensemble on the {inteiro(r['n_teste'])} held-out"
            " sentences, with $T$ fitted on out-of-fold validation predictions. By-video and"
            " by-series partitions: out-of-fold predictions of single models, with $T$ cross-fitted"
            " on the other folds. ECE values in percentage points (15 bins for confidence and"
            " $P(\\text{Negative})$, 10 for suitability). ``$C\\geq$0.99'' is the share of items"
            " whose predicted class carries a confidence of at least 0.99, and ``Acc.'' the"
            " accuracy among them.",
            f"Calibração do que a interface exibe, antes e depois da escala de temperatura. Partição"
            f" de referência: o ensemble de cinco modelos nas {inteiro(r['n_teste'])} frases do teste"
            " retido, com $T$ ajustada nas predições de validação fora da amostra. Partições por"
            " vídeo e por série: predições fora da amostra de modelos únicos, com $T$ de ajuste"
            " cruzado nos outros folds. ECE em pontos percentuais (15 faixas para a confiança e"
            " $P(\\text{Negativo})$, 10 para a adequação). ``$C\\geq$0,99'' é a fração de itens cuja"
            " classe predita tem confiança de pelo menos 0,99, e ``Acur.'' a acurácia entre eles.",
        )
        + "}\\label{tab:calibracao}",
        "\\centering\\small",
        "\\begin{tabular}{@{}llccccccc@{}}",
        "\\hline\\hline",
        tr(
            "Partition & & ECE conf. & ECE $P(\\text{Neg})$ & ECE suitab. & Brier & NLL & $C\\geq$0.99 & Acc. \\\\",
            "Partição & & ECE conf. & ECE $P(\\text{Neg})$ & ECE adeq. & Brier & NLL & $C\\geq$0,99 & Acur. \\\\",
        ),
        "\\hline",
    ]

    def linha(nome, sub, b):
        return (f"{nome} & {sub} & {pnum(b['ece_topo'])} & {pnum(b['ece_neg'])} & {pnum(b['ece_adequacao'])}"
                f" & {num(b['brier'])} & {num(b['nll'])} & {pct(b['frac_conf_alta'])} & {pct(b['acuracia_conf_alta'])} \\\\")

    bruto = tr("raw", "bruto")
    lin.append(linha(tr("Reference (ensemble)", "Referência (ensemble)"), bruto, r["teste_antes"]))
    lin.append(linha("", f"$T={num(r['temperatura'], 2)}$", r["teste_depois"]))
    for e in ("video", "serie"):
        d = c.get("agrupado", {}).get(e)
        if d:
            lin.append(linha(ESQ(e), bruto, d["antes"]))
            lin.append(linha("", tr("cross-fitted $T$", "$T$ de ajuste cruzado"), d["depois"]))
    lin += ["\\hline\\hline", "\\end{tabular}", "\\end{table*}"]
    grava_tabela("tab-calibracao.tex", lin)


# ---------------------------------------------------------------- deferencia


def macros_deferencia(m: Macros, d: dict) -> None:
    m("defN", inteiro(d["n"]))
    m("defErros", inteiro(d["n_erros"]))
    m("defFronteira", inteiro(d["n_erros_fronteira"]))
    m("defFronteiraNegNao", inteiro(d["erros_fronteira_neg_para_nao"]))
    m("defFronteiraNaoNeg", inteiro(d["erros_fronteira_nao_para_neg"]))
    m("defAcertoDecisao", pct(1 - d["n_erros_fronteira"] / d["n"]))
    m("defRiscoBase", pct(d["risco_sem_deferencia"]))
    m("defRebaixaAbaixoMeio", inteiro(d["rebaixados_com_pneg_abaixo_de_meio"]))
    m("defPNegMin", num(d["pneg_min"], 2))
    m("defPNegMax", num(d["pneg_max"], 2))
    orc = (("0.05", "Cinco"), ("0.10", "Dez"), ("0.20", "Vinte"), ("0.30", "Trinta"))
    for s, v in d["sinais"].items():
        S = letras(s)
        m(f"def{S}AurocErro", num(v["auroc_erro"]))
        m(f"def{S}AurocFronteira", num(v["auroc_fronteira"]))
        m(f"def{S}Aurc", num(v["aurc"]))
        for q, nome in orc:
            m(f"def{S}CapErro{nome}", pct(v["captura_erro"][q]))
            m(f"def{S}CapFronteira{nome}", pct(v["captura_fronteira"][q]))
        # uma marcacao aleatoria captura, em media, exatamente o orcamento
        m(f"def{S}VezesAcasoDez", dec(v["captura_fronteira"]["0.10"] / 0.10, 1))
        m(f"def{S}RiscoNoventa", pct(v["risco_cobertura_90"]))
        m(f"def{S}RiscoOitenta", pct(v["risco_cobertura_80"]))
    for k, v in d["ic_captura_10"].items():
        m(f"defIC{letras(k)}", ic(v))
    for e, b in d.get("agrupado", {}).items():
        E = letras(e)
        m(f"defG{E}Fronteira", inteiro(b["n_erros_fronteira"]))
        for s, v in b["sinais"].items():
            S = letras(s)
            m(f"defG{E}{S}CapFronteiraDez", pct(v["captura_fronteira"]["0.10"]))
            m(f"defG{E}{S}CapFronteiraVinte", pct(v["captura_fronteira"]["0.20"]))
        for k, v in b["ic_captura_10"].items():
            m(f"defG{E}IC{letras(k)}", ic(v))

    ag = d.get("agrupado", {})
    lin = [
        CABECALHO,
        "\\begin{table*}[!t]",
        "\\caption{"
        + tr(
            f"Uncertainty signals as a basis for flagging items for the caregiver to check."
            f" Reference partition: ensemble with temperature scaling, {inteiro(d['n'])} held-out"
            f" sentences, {inteiro(d['n_erros'])} errors of which {inteiro(d['n_erros_fronteira'])}"
            " cross the risk boundary. AUROC of each signal for detecting an error and a boundary"
            " error; AURC is the area under the risk--coverage curve (lower is better); capture is"
            " the share of boundary errors that fall among the flagged items for a given review"
            " budget. The last two columns repeat the capture at a 10\\% budget for unseen videos"
            " and series (single models, cross-fitted temperature; no ensemble, hence no"
            " disagreement). A random flag captures, on average, exactly the budget.",
            f"Sinais de incerteza como base para marcar itens para o responsável conferir. Partição"
            f" de referência: ensemble com escala de temperatura, {inteiro(d['n'])} frases do teste"
            f" retido, {inteiro(d['n_erros'])} erros, dos quais {inteiro(d['n_erros_fronteira'])}"
            " cruzam a fronteira de risco. AUROC de cada sinal na detecção de um erro e de um erro de"
            " fronteira; AURC é a área sob a curva risco--cobertura (menor é melhor); captura é a"
            " fração dos erros de fronteira que cai entre os itens marcados para um dado orçamento"
            " de conferência. As duas últimas colunas repetem a captura com orçamento de 10\\% para"
            " vídeos e séries não vistos (modelos únicos, temperatura de ajuste cruzado; sem"
            " ensemble, logo sem desacordo). Uma marca aleatória captura, em média, exatamente o"
            " orçamento.",
        )
        + "}\\label{tab:deferencia}",
        "\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}lccccccccc@{}}",
        "\\hline\\hline",
        tr(
            " & \\multicolumn{2}{c}{AUROC} & & \\multicolumn{4}{c}{Boundary errors captured (reference)} & \\multicolumn{2}{c}{At 10\\%, unseen} \\\\",
            " & \\multicolumn{2}{c}{AUROC} & & \\multicolumn{4}{c}{Capturados (referência)} & \\multicolumn{2}{c}{A 10\\%, não vistos} \\\\",
        ),
        tr(
            "Signal & Any error & Boundary & AURC & 5\\% & 10\\% & 20\\% & 30\\% & Videos & Series \\\\",
            "Sinal & Qualquer erro & Fronteira & AURC & 5\\% & 10\\% & 20\\% & 30\\% & Vídeos & Séries \\\\",
        ),
        "\\hline",
    ]
    for s in ("confianca", "margem", "entropia", "desacordo", "meio", "decisao"):
        v = d["sinais"][s]
        cf = v["captura_fronteira"]
        gv = ag.get("video", {}).get("sinais", {}).get(s)
        gs = ag.get("serie", {}).get("sinais", {}).get(s)
        lin.append(
            f"{SINAL(s)} & {num(v['auroc_erro'])} & {num(v['auroc_fronteira'])} & {num(v['aurc'])}"
            f" & {pnum(cf['0.05'])} & {pnum(cf['0.10'])} & {pnum(cf['0.20'])} & {pnum(cf['0.30'])}"
            f" & {pnum(gv['captura_fronteira']['0.10']) if gv else '--'} & {pnum(gs['captura_fronteira']['0.10']) if gs else '--'} \\\\"
        )
    lin += ["\\hline\\hline", "\\end{tabular}", "\\end{table*}"]
    grava_tabela("tab-deferencia.tex", lin)


# ------------------------------------------------------------------ robustez


def macros_robustez(m: Macros, r: dict) -> None:
    m("robN", inteiro(r["n"]))
    for v, d in r["variantes"].items():
        V = letras(v)
        m(f"rob{V}FUm", pct(d["f1_macro"]))
        m(f"rob{V}RecallNeg", pct(d["recall_neg"]))
        m(f"rob{V}FUmNeg", pct(d["f1_neg"]))
        m(f"rob{V}Muda", pct(d["mudanca_rebaixamento"]))
        m(f"rob{V}Ece", pct(d["ece_topo"]))
        m(f"rob{V}Conf", pct(d["conf_media"]))
        m(f"rob{V}PrecNeg", pct(d["precisao_neg"]))
        m(f"rob{V}PredNeg", pct(d["taxa_pred_neg"]))
        m(f"rob{V}FUmPos", pct(d["f1_pos"]))
        m(f"rob{V}FUmNeu", pct(d["f1_neu"]))
        m(f"rob{V}Novos", dec(d["rebaixamentos_novos"], 0))
        m(f"rob{V}Perdidos", dec(d["rebaixamentos_perdidos"], 0))
    base = r["variantes"]["limpo"]
    for v, d in r["variantes"].items():
        if v == "limpo":
            continue
        m(f"rob{letras(v)}QuedaFUm", dec(100 * (base["f1_macro"] - d["f1_macro"]), 1))
        m(f"rob{letras(v)}QuedaRecallNeg", dec(100 * (base["recall_neg"] - d["recall_neg"]), 1))

    lin = [
        CABECALHO,
        "\\begin{table*}[!t]",
        "\\caption{"
        + tr(
            "Ensemble behaviour when the held-out sentences arrive in degraded forms (mean over three"
            " noise seeds; sample standard deviation in parentheses where noise is random)."
            " Decisions are those of the ensemble as evaluated, without temperature. ``Pred.\\ Neg.''"
            " is the share of sentences predicted Negative, i.e.\\ demoted. ``Flips'' is the share of"
            " sentences whose demote/keep decision differs from the decision on the curated sentence,"
            " and ``new/lost'' counts the flips that create and that remove a demotion. ECE and mean"
            " confidence use the temperature fitted on curated validation sentences. Values in \\%"
            " except counts.",
            "Comportamento do ensemble quando as frases do teste retido chegam degradadas (média sobre"
            " três sementes de ruído; desvio-padrão amostral entre parênteses onde o ruído é"
            " aleatório). As decisões são as do ensemble como avaliado, sem temperatura. ``Pred.\\"
            " Neg.'' é a fração de frases preditas Negativo, isto é, rebaixadas. ``Mudanças'' é a"
            " fração de frases cuja decisão de rebaixar ou manter difere da decisão na frase revisada,"
            " e ``novos/perdidos'' conta as mudanças que criam e que removem um rebaixamento. ECE e"
            " confiança média usam a temperatura ajustada nas frases revisadas de validação. Valores"
            " em \\%, exceto contagens.",
        )
        + "}\\label{tab:robustez}",
        # 3pt: com 4pt os rotulos em portugues estouram a largura do texto
        "\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lccccccccc@{}}",
        "\\hline\\hline",
        tr(
            "Input form & Macro F1 & F1 Pos. & Recall Neg. & Prec.\\ Neg. & Pred.\\ Neg. & Flips & New/lost & ECE conf. & Mean conf. \\\\",
            "Forma da entrada & F1 macro & F1 Pos. & Recall Neg. & Prec.\\ Neg. & Pred.\\ Neg. & Mudanças & Novos/perdidos & ECE conf. & Conf.\\ média \\\\",
        ),
        "\\hline",
    ]
    for v in ("limpo", "legenda", "legenda+pal10", "legenda+pal20", "legenda+let05", "legenda+let10"):
        d = r["variantes"].get(v)
        if not d:
            continue
        dp = lambda k: f" ({dec(100 * d[k + '_dp'], 1)})" if d["sementes"] > 1 else ""
        lin.append(
            f"{VAR(v)} & {pnum(d['f1_macro'])}{dp('f1_macro')} & {pnum(d['f1_pos'])} & {pnum(d['recall_neg'])}"
            f" & {pnum(d['precisao_neg'])} & {pnum(d['taxa_pred_neg'])} & {pnum(d['mudanca_rebaixamento'])}"
            f" & {dec(d['rebaixamentos_novos'], 0)}/{dec(d['rebaixamentos_perdidos'], 0)}"
            f" & {pnum(d['ece_topo'])} & {pnum(d['conf_media'])} \\\\"
        )
    lin += ["\\hline\\hline", "\\end{tabular}", "\\end{table*}"]
    grava_tabela("tab-robustez.tex", lin)


# ------------------------------------------------------------- classificador


def macros_classificador(m: Macros) -> None:
    """Desempenho do retreino no teste retido (sem temperatura: e o sistema como esta)."""
    import numpy as np

    import metricas as M
    from treino import carrega_corpus

    d = np.load(RESULTADOS / "logits_referencia.npz", allow_pickle=True)
    itens = carrega_corpus()
    y = np.array([itens[i]["rotulo"] for i in d["idx_teste"]])
    p = M.ensemble(d["logits_teste"])
    pc = M.por_classe(y, p.argmax(1))
    m("clsFUmMacro", pct(pc["f1_macro"]))
    m("clsAcuracia", pct(pc["acuracia"]))
    m("clsFUmNeg", pct(pc["neg"]["f1"]))
    m("clsRecallNeg", pct(pc["neg"]["recall"]))
    meta = json.loads(str(d["meta"]))
    m("clsMinutos", dec(meta["minutos"], 0))
    m("clsNTeste", inteiro(len(y)))
    m("clsNResto", inteiro(len(d["idx_resto"])))
    m("clsNTrabalho", inteiro(len(itens)))
    total = json.load(open(RESULTADOS / "corpus_resumo.json", encoding="utf-8"))["total"]
    m("clsNExcluidas", inteiro(total - len(itens)))
    rep = carrega("reproducao.json")
    if rep:
        m("repMesmaClasse", inteiro(rep["mesma_classe"]))
        expo = int(np.floor(np.log10(rep["max_dif_prob"]))) + 1
        m("repMaxDifExpoente", str(expo))  # a maior diferenca e menor que 10^expo


# --------------------------------------------------------------------- main


def gera(idioma: str) -> None:
    global IDIOMA
    IDIOMA = idioma
    m = Macros()
    corpus = carrega("corpus_resumo.json")
    if corpus:
        macros_corpus(m, corpus)
    g = carrega("generalizacao.json")
    if g:
        macros_generalizacao(m, g)
    c = carrega("calibracao.json")
    if c:
        macros_calibracao(m, c)
        macros_classificador(m)
    d = carrega("deferencia.json")
    if d:
        macros_deferencia(m, d)
    r = carrega("robustez.json")
    if r:
        macros_robustez(m, r)

    cab = [CABECALHO, f"% Fonte: src/resultados/*.json ({time.strftime('%Y-%m-%d')}); idioma: {idioma}"]
    destino = (LATEX_PT if idioma == "pt" else LATEX) / "numeros.tex"
    destino.write_text("\n".join(cab + m.linhas) + "\n", encoding="utf-8")
    print(f"{len(m.linhas)} macros -> {destino}")


def main():
    prepara_saidas()
    for idioma in ("en", "pt"):
        gera(idioma)


if __name__ == "__main__":
    main()

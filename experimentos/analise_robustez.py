"""Robustez ao formato das transcricoes que a plataforma entrega.

As frases do corpus foram revisadas: tem caixa, pontuacao e trechos ruidosos removidos.
A legenda automatica que o sistema recebe em operacao nao tem nada disso. Este script
mede o que acontece com o ensemble -- e com o que a interface exibe -- quando a mesma
frase chega no formato de legenda e com perdas de palavras e letras
(variantes definidas em protocolo_referencia.py).

As DECISOES (classe, rebaixamento) sao as do ensemble sem temperatura, isto e, as do
sistema tal como avaliado: a temperatura e uma correcao do que se exibe, e aplica-la
antes da media mudaria algumas classes. A temperatura (ajustada nas frases limpas de
validacao) entra so no ECE e na confianca media -- e e justamente o que se quer medir:
se o ruido torna o modelo excessivamente confiante, a calibracao feita em laboratorio
deixa de valer em operacao.

Para cada variante, media e desvio-padrao amostral (ddof=1) sobre as sementes de ruido:
  - macro F1, F1 por classe, recall e precisao de Negativo, taxa de predicao Negativa;
  - decisoes de rebaixamento que mudam em relacao a frase limpa, separadas em
    rebaixamentos novos (nao era Negativo, passa a ser) e perdidos (o inverso);
  - ECE da confianca e confianca media, com a temperatura das frases limpas.

Saidas: resultados/robustez.json
"""

from __future__ import annotations

import json

import numpy as np

import metricas as M
from caminhos import RESULTADOS, prepara_saidas


def main():
    prepara_saidas()
    from treino import carrega_corpus

    d = np.load(RESULTADOS / "logits_referencia.npz", allow_pickle=True)
    t = json.load(open(RESULTADOS / "calibracao.json", encoding="utf-8"))["referencia"]["temperatura"]
    itens = carrega_corpus()
    idx_teste = d["idx_teste"]
    y = np.array([itens[i]["rotulo"] for i in idx_teste])

    rebaixa_limpo = M.ensemble(d["logits_teste"]).argmax(1) == M.NEG

    res = {"temperatura": t, "n": int(len(y)), "variantes": {}}
    for nome in d["nomes_variantes"]:
        arr = d[f"ruido__{nome}"]  # [fold, semente, n, 3]
        por_semente = []
        for s in range(arr.shape[1]):
            pred = M.ensemble(arr[:, s]).argmax(1)  # decisao: sem temperatura
            p_cal = M.ensemble(arr[:, s], t)  # exibicao: com temperatura
            pc = M.por_classe(y, pred)
            rebaixa = pred == M.NEG
            por_semente.append(
                {
                    "f1_macro": pc["f1_macro"],
                    "f1_neg": pc["neg"]["f1"],
                    "f1_neu": pc["neu"]["f1"],
                    "f1_pos": pc["pos"]["f1"],
                    "recall_neg": pc["neg"]["recall"],
                    "precisao_neg": pc["neg"]["precisao"],
                    "taxa_pred_neg": float(rebaixa.mean()),
                    "acuracia": pc["acuracia"],
                    "mudanca_rebaixamento": float((rebaixa != rebaixa_limpo).mean()),
                    "rebaixamentos_novos": int((rebaixa & ~rebaixa_limpo).sum()),
                    "rebaixamentos_perdidos": int((~rebaixa & rebaixa_limpo).sum()),
                    "ece_topo": M.ece_topo(p_cal, y),
                    "conf_media": float(p_cal.max(1).mean()),
                }
            )
        chaves = por_semente[0].keys()
        n_s = len(por_semente)
        res["variantes"][str(nome)] = {
            "sementes": n_s,
            **{k: float(np.mean([x[k] for x in por_semente])) for k in chaves},
            **{
                f"{k}_dp": float(np.std([x[k] for x in por_semente], ddof=1)) if n_s > 1 else 0.0
                for k in chaves
            },
        }
        v = res["variantes"][str(nome)]
        print(f"{nome:18s} F1 {v['f1_macro']:.4f} recall-neg {v['recall_neg']:.4f} prec-neg {v['precisao_neg']:.4f} "
              f"pred-neg {v['taxa_pred_neg']:.3f} novos {v['rebaixamentos_novos']:.1f} perdidos {v['rebaixamentos_perdidos']:.1f} "
              f"ECE {v['ece_topo']:.4f}")

    with open(RESULTADOS / "robustez.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

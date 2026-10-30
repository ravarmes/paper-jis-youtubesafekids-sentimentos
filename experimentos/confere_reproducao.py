"""Confere o retreino do protocolo R contra as predicoes da execucao original.

O classificador auditado e o mesmo da execucao de referencia (manuscrito relacionado,
em avaliacao). Este script compara, frase a frase, o ensemble retreinado aqui com as
predicoes salvas daquela execucao: mesma particao, mesma classe, e a maior diferenca
de probabilidade. Precisa do arquivo de referencia ao lado do monorepo; fora dele, nao
roda, e resultados/reproducao.json (versionado) guarda o que ele encontrou.

Saida: resultados/reproducao.json
"""

from __future__ import annotations

import json

import numpy as np

import metricas as M
from caminhos import RESULTADOS, prepara_saidas
from protocolo_referencia import REFERENCIA


def main():
    prepara_saidas()
    if not REFERENCIA.exists():
        raise SystemExit(f"arquivo de referencia ausente: {REFERENCIA}")
    a = np.load(RESULTADOS / "logits_referencia.npz", allow_pickle=True)
    b = np.load(REFERENCIA, allow_pickle=True)
    pa, pb = M.ensemble(a["logits_teste"]), b["probs_ensemble"]
    res = {
        "mesma_particao": bool(np.array_equal(a["idx_teste"], b["idx"])),
        "n": int(len(pa)),
        "mesma_classe": int((pa.argmax(1) == pb.argmax(1)).sum()),
        "max_dif_prob": float(np.abs(pa - pb).max()),
    }
    with open(RESULTADOS / "reproducao.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(res)


if __name__ == "__main__":
    main()

# Can Caregivers Trust the Score? — code and data

Code and data behind the article *Can Caregivers Trust the Score? Auditing the Signals of a
Sentiment-Based Re-ranker for Children's Video Search* (submitted to the Journal on
Interactive Systems).

> Public repository: <https://github.com/ravarmes/paper-jis-youtubesafekids-sentimentos>.
> Mirrored from a private monorepo; pull requests are not accepted here.

The article audits the three readings that the YouTube Safe Kids prototype shows a caregiver
for each search result — the position (Safety Score), the colour (predicted sentiment class)
and the percentage (predicted suitability) — and a fourth reading it proposes, a "check this"
flag. It does not propose a new classifier: it retrains the deployed one with its original
recipe and measures generalisation to unseen videos and series, whose content gets demoted,
calibration, deferral to the caregiver and robustness to caption-like input.

## How to run

Python 3.9+, with `torch` (CUDA recommended), `transformers`, `scikit-learn`, `scipy`,
`numpy` and `matplotlib`. BERTimbau (`neuralmind/bert-base-portuguese-cased`) is read from
the local Hugging Face cache; download it once beforehand.

```bash
cd experimentos
python fontes.py                 # franchise and content category of every sentence
python protocolo_referencia.py   # Protocol R: 5-model ensemble on the reference partition (~10 min on a laptop GPU)
python protocolo_agrupado.py     # Protocol G: by-sentence, by-video and by-series CV (~35 min)
python analise_generalizacao.py  # RQ1, RQ2
python analise_calibracao.py     # RQ3 (+ figure)
python analise_deferencia.py     # RQ4 (+ figure)
python analise_robustez.py       # RQ5
python figura_interface.py       # illustrative interface figures (simulated probabilities, not system output)
python confere_reproducao.py     # optional: compares the retrained ensemble with the original run (needs its predictions file)
python gerar_numeros.py          # every number, table and macro of the manuscript
```

The two training scripts are the only expensive steps. Their outputs (`logits_*.npz`) are
versioned in `resultados/`, so every analysis can be re-run without retraining.

## Structure

| Path | Content |
|---|---|
| `dados/corpus.csv` | 2,749 sentences, each with four independent human annotations (`P1`, `A1`–`A3`), the final label (`AS`, majority vote), video code, video title and narrative position |
| `dados/corpus-rubrica*.md` | annotation guidelines (Portuguese and English) |
| `experimentos/treino.py` | training recipe of the deployed classifier, unchanged |
| `experimentos/fontes.py` | explicit rules that assign franchise and content category |
| `experimentos/protocolo_*.py` | the two evaluation protocols |
| `experimentos/analise_*.py` | one script per research question |
| `experimentos/metricas.py` | definitions of every metric (ECE, Brier, AUROC, AURC, cluster bootstrap) |
| `experimentos/gerar_numeros.py` | JSON results → LaTeX macros and tables |
| `resultados/` | logits, JSON results and `fontes_videos.csv` (category of every title, for audit) |

## Data

The corpus is the one used to train YouTube Safe Kids. Its 2,749 sentences were transcribed
from YouTube videos with the support of transcription tools (Clipto and NotebookLM); each row
carries a video code (`VIDEO`), the video title and the position in the narrative (`PARTE`).
The labels are human: four annotators (`P1`, `A1`–`A3`) labelled every sentence by hand and
independently, following `dados/corpus-rubrica.md`, and the final label (`AS`) is their
majority vote, with ties settled by `P1`. No automatic or AI tool took part in the annotation;
the classifier is trained and evaluated on these labels, and none of its predictions is stored
in the corpus. A video can appear under more than one title (compilations split into numbered parts), so the
by-video partition groups sentences by `VIDEO`; `fontes.py` checks that every title belongs to a
single video.

## How to cite

See `CITATION.cff`.

## Licence

Code under MIT. The annotation (four labels per sentence, final label, guidelines) under
CC BY 4.0. The transcribed sentences are short excerpts of third-party videos, reproduced as
quotation for research purposes; no licence over them is granted. See `LICENSE`.

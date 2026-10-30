"""Caminhos compartilhados pelos scripts de experimento.

Um unico lugar decide onde estao os insumos e para onde vao as saidas.

  dados/        corpus.csv -- entra, nunca e escrito
  resultados/   .json, .npz e .csv versionados: permitem auditar os numeros do artigo
                sem reexecutar o treino
  saidas/       tabelas .tex, figuras e numeros.tex, quando esta pasta e usada sozinha.
                Dentro do monorepo do artigo vao direto para ../../latex, de onde o
                main.tex as le.
"""

from __future__ import annotations

from pathlib import Path

EXPERIMENTOS = Path(__file__).parent
BASE = EXPERIMENTOS.parent

DADOS = BASE / "dados"
RESULTADOS = BASE / "resultados"
CORPUS = DADOS / "corpus.csv"

_LATEX_ARTIGO = BASE.parent / "latex"
LATEX = _LATEX_ARTIGO if _LATEX_ARTIGO.is_dir() else BASE / "saidas"
FIGURAS = LATEX / "figuras"
TABELAS = LATEX / "tabelas"

# Versao em portugues, so para conferencia do autor (nao vai a revista): mesmos numeros,
# formatacao pt-BR, tabelas e rotulos de figura traduzidos.
LATEX_PT = LATEX / "pt"
FIGURAS_PT = LATEX_PT / "figuras"
TABELAS_PT = LATEX_PT / "tabelas"


def prepara_saidas() -> None:
    """Cria os diretorios de escrita. Chamar antes de gravar qualquer coisa."""
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    for d in (FIGURAS, TABELAS, FIGURAS_PT, TABELAS_PT):
        d.mkdir(parents=True, exist_ok=True)

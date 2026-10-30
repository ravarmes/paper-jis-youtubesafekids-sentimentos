"""Origem de cada video do corpus: franquia e categoria de conteudo.

POR QUE EXISTE
--------------
Duas analises do artigo precisam saber de onde vem cada frase:

  - a validacao agrupada (protocolo_agrupado.py) separa treino e teste por FRANQUIA,
    para medir o desempenho sobre series que o modelo nunca viu -- que e a situacao de
    toda busca real;
  - a analise de quem e rebaixado (analise_fontes.py) cruza rotulo e predicao com a
    CATEGORIA do conteudo.

O corpus nao traz essa informacao: so o titulo. As regras abaixo a derivam do titulo,
uma a uma, e ficam no codigo para que qualquer atribuicao possa ser conferida e
contestada. A primeira regra que casa decide. Titulo que nao casa com nenhuma regra e
erro (nao ha categoria padrao): melhor parar do que classificar em silencio.

CATEGORIAS
----------
  adulta    animacao produzida para adultos (South Park, Rick and Morty, Boondocks...)
            e videos de comentario sobre essas series
  classica  desenho classico de humor pastelao para todas as idades, historicamente
            exibido para criancas (Looney Tunes, Pica-Pau, Flintstones)
  narrativa conteudo infantil de entretenimento e narrativa: series infantis, contos de
            fadas, fabulas, cancoes, trechos de filmes infantis
  educativa conteudo infantil de proposito didatico ou socioemocional: matematica,
            alfabetizacao, ciencias, valores, mensagens de autoestima

Franquia e a serie (ou o canal de uma serie). Video avulso, sem serie, e a sua propria
franquia: as regras agrupam avulsos sob rotulos de conveniencia ("Educativo avulso",
"Contos e fabulas"...) para o relatorio, e grupo_serie() devolve o titulo nesses casos,
para que a validacao por serie nao trate dezenas de videos independentes como um so.

VIDEO
-----
Cada frase traz o codigo do video de onde foi transcrita (coluna VIDEO, v001...). Um video
pode aparecer sob mais de um titulo (compilacoes repartidas em titulos numerados), por
isso o agrupamento por video usa o codigo, e nao o titulo. atribui_grupos() confere que
cada titulo pertence a um unico video.

Uso: python fontes.py   -> grava resultados/fontes_videos.csv e
                           resultados/corpus_resumo.json (totais e categorias)
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter

from caminhos import CORPUS, RESULTADOS, prepara_saidas


def normaliza(txt: str) -> str:
    txt = unicodedata.normalize("NFKD", txt or "")
    txt = "".join(c for c in txt if not unicodedata.combining(c))
    txt = re.sub(r"[^\w\s]", " ", txt.lower())
    return re.sub(r"\s+", " ", txt).strip()


# (padrao sobre o titulo normalizado, franquia, categoria). A ordem importa: regras
# especificas antes das genericas.
REGRAS: list[tuple[str, str, str]] = [
    # --- animacao adulta e comentario sobre ela
    (r"south park|cartman|stan e kyle|kyle salva|wendy sendo xingada|butters", "South Park", "adulta"),
    (r"boondocks|tio ruckus", "The Boondocks", "adulta"),
    (r"rick and morty|rick e morty|rick morty|\brick\b.*\bmorty\b|jerry vai se vingar|jerry daria", "Rick and Morty", "adulta"),
    (r"brickleberry", "Brickleberry", "adulta"),
    (r"aqua ?teen|mestre shake", "Aqua Teen", "adulta"),
    (r"family guy|familia da pesada", "Family Guy", "adulta"),
    (r"simpsons|\bbart\b", "The Simpsons", "adulta"),
    (r"big mouth", "Big Mouth", "adulta"),
    (r"bojack", "BoJack Horseman", "adulta"),
    (r"close enough", "Close Enough", "adulta"),
    # --- desenho classico de humor pastelao
    (r"looney tunes|pernalonga|patolino", "Looney Tunes", "classica"),
    (r"pica pau", "Pica-Pau", "classica"),
    (r"flintstones", "Flintstones", "classica"),
    # --- series infantis de entretenimento
    (r"peppa|pepa ping", "Peppa Pig", "narrativa"),
    (r"aprenda com luna", "O Show da Luna", "educativa"),
    (r"show da luna", "O Show da Luna", "narrativa"),
    (r"bob esponja|spongebob|hamburguer de siri", "Bob Esponja", "narrativa"),
    (r"teen titans", "Teen Titans Go", "narrativa"),
    (r"gumball", "Gumball", "narrativa"),
    (r"garfield", "Garfield", "narrativa"),
    (r"patrulha canina", "Patrulha Canina", "narrativa"),
    (r"backyardigans", "Backyardigans", "narrativa"),
    (r"turma da monica", "Turma da Monica", "narrativa"),
    (r"irmao do jorel", "Irmao do Jorel", "narrativa"),
    (r"detetive labrador", "Detetive Labrador", "narrativa"),
    (r"desenho infantil 32 minutos|desenho infantil 28 minutos", "Desenho Infantil (canal)", "narrativa"),
    (r"sonic x", "Sonic X", "narrativa"),
    (r"scooby", "Scooby-Doo", "narrativa"),
    (r"he man", "He-Man", "narrativa"),
    (r"spidey", "Spidey", "narrativa"),
    (r"krypto", "Krypto", "narrativa"),
    (r"ricky zoom|ricky tem um pneu", "Ricky Zoom", "narrativa"),
    (r"44 gatos", "44 Gatos", "narrativa"),
    (r"george o curioso", "George, o Curioso", "narrativa"),
    (r"tres espias", "Tres Espias Demais", "narrativa"),
    (r"robofuse", "Robofuse", "narrativa"),
    (r"pea pea", "Pea Pea", "narrativa"),
    (r"dora seguiu", "Dora", "narrativa"),
    (r"bingo", "Bingo", "narrativa"),
    (r"tartarugas ninja", "Cancoes infantis (canal)", "narrativa"),
    (r"tucano estava fazendo|obby no roblox", "Cancoes infantis (canal)", "narrativa"),
    (r"galinha pintadinha", "Galinha Pintadinha", "narrativa"),
    (r"caminhao curioso", "Leo, o Caminhao", "narrativa"),
    (r"funk do patinho", "Cantinho Encantado", "narrativa"),
    (r"sitio do seu lobato", "Sitio do Seu Lobato", "narrativa"),
    (r"looloo", "LooLoo Kids", "narrativa"),
    (r"historinhas da sara|sarah e eloah", "Sarah e Eloah", "narrativa"),
    (r"sarah em regras de conduta", "Sarah e Eloah", "educativa"),
    (r"minecraft", "Minecraft (video)", "narrativa"),
    (r"tucci", "Bolsa da Tucci", "narrativa"),
    (r"amigu", "Os Amiguinhos (canal)", "narrativa"),
    # trechos de filmes e series infantis (Disney, Nickelodeon, Cartoon Network)
    (r"aladdin|bambi|procurando nemo|princesa e o sapo|frozen|cinderela", "Disney (trechos)", "narrativa"),
    (r"avatar", "Avatar", "narrativa"),
    (r"hora de aventura", "Hora de Aventura", "narrativa"),
    (r"rugrats|anjinhos", "Rugrats", "narrativa"),
    # --- contos de fadas e fabulas avulsos
    (r"patinho feio|nuvenzinha|chapeuzinho|joao e maria|sete cabritinhos|3 porquinhos|rapunzel|lebre e a tartaruga|cigarra e a formiga|contacao de historias", "Contos e fabulas", "narrativa"),
    # --- conteudo de curiosidade sobre desenhos (comentario, nao infantil nem adulto)
    (r"desenhos que foram banidos|desenhos assustadores feitos por criancas|com quantos anos voce descobriu", "Curiosidades sobre desenhos", "narrativa"),
    # --- conteudo educativo e socioemocional
    (r"matematica|tabuada|fracoes|alfabeto|vogais|aprender a ler|ortograficas|conhecimentos gerais|pre historia|sistema solar|dinossauros|sentidos para as criancas|alimentacao saudavel|ingles infantil|experimentos de ciencia|cohete|manualidades", "Educativo avulso", "educativa"),
    (r"convivencia|elogio|autoestima|resiliencia|perseveranca|coragem|amizade|alegria e diversao|trabalho em equipe|beleza e a magia da infancia|afeto e carinho|respeito e diversidade|inclusao|amor e bondade|responsabilidade e carater|sonhar alto|aprender brincando|licoes de vida|frases inspiradoras", "Mensagens socioemocionais", "educativa"),
]

_COMPILADAS = [(re.compile(p), f, c) for p, f, c in REGRAS]

# Rotulos que agrupam videos avulsos so para o relatorio; nao sao series.
AVULSOS = {
    "Educativo avulso",
    "Mensagens socioemocionais",
    "Contos e fabulas",
    "Curiosidades sobre desenhos",
    "Disney (trechos)",
}


def fonte(titulo: str) -> tuple[str, str]:
    """(franquia, categoria) de um titulo. Levanta ValueError se nenhuma regra casar."""
    t = normaliza(titulo)
    for padrao, franquia, categoria in _COMPILADAS:
        if padrao.search(t):
            return franquia, categoria
    raise ValueError(f"titulo sem regra de origem: {titulo!r}")


def grupo_serie(titulo: str) -> str:
    """Chave de agrupamento por serie: a franquia, ou o proprio titulo se avulso."""
    franquia, _ = fonte(titulo)
    return f"avulso:{normaliza(titulo)}" if franquia in AVULSOS else franquia


def atribui_grupos(linhas: list[dict]) -> list[str]:
    """Video de cada linha (coluna VIDEO).

    Falha se um titulo aparecer em mais de um video: frases do mesmo video dos dois lados
    de uma particao agrupada invalidariam a avaliacao sobre videos nao vistos.
    """
    grupos = [(r.get("VIDEO") or "").strip() for r in linhas]
    if not all(grupos):
        raise ValueError("linha sem codigo de video (coluna VIDEO)")
    por_titulo: dict[str, set[str]] = {}
    for r, g in zip(linhas, grupos):
        por_titulo.setdefault(normaliza(r.get("TITULO")), set()).add(g)
    repartidos = [t for t, v in por_titulo.items() if len(v) > 1]
    if repartidos:
        raise ValueError(f"titulos em mais de um video: {repartidos[:5]}")
    return grupos


def main() -> None:
    prepara_saidas()
    with open(CORPUS, encoding="utf-8-sig") as fh:
        linhas = list(csv.DictReader(fh, delimiter=";"))

    por_titulo: dict[str, Counter] = {}
    for r in linhas:
        por_titulo.setdefault(r["TITULO"].strip(), Counter())[r["AS"].strip()] += 1

    sem_regra = []
    saida = []
    for titulo, cont in por_titulo.items():
        try:
            franquia, categoria = fonte(titulo)
        except ValueError:
            sem_regra.append(titulo)
            continue
        saida.append((titulo, franquia, categoria, cont["Negativo"], cont["Neutro"], cont["Positivo"]))

    if sem_regra:
        for t in sem_regra:
            print("SEM REGRA:", t)
        raise SystemExit(f"{len(sem_regra)} titulos sem regra de origem")

    destino = RESULTADOS / "fontes_videos.csv"
    with open(destino, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["TITULO", "FRANQUIA", "CATEGORIA", "N_NEG", "N_NEU", "N_POS"])
        w.writerows(sorted(saida, key=lambda x: (x[2], x[1], x[0])))

    cat = Counter()
    for _, _, c, n, e, p in saida:
        cat[c] += n + e + p
    print(f"{len(saida)} titulos, {len({s[1] for s in saida})} franquias -> {destino.name}")
    for c, n in cat.most_common():
        print(f"  {c:10s} {n} frases")

    grupos = atribui_grupos(linhas)
    resumo = {
        "total": len(linhas),
        "n_titulos": len(saida),
        "n_videos": len(set(grupos)),
        "n_franquias": len({s[1] for s in saida if s[1] not in AVULSOS}),
        "categorias_frases": dict(cat),
        "rotulos_por_categoria": {
            c: dict(Counter(r["AS"].strip() for r in linhas if fonte(r["TITULO"])[1] == c)) for c in cat
        },
    }
    with open(RESULTADOS / "corpus_resumo.json", "w", encoding="utf-8") as fh:
        json.dump(resumo, fh, indent=2, ensure_ascii=False)
    print(f"  {resumo['n_videos']} videos")


if __name__ == "__main__":
    main()

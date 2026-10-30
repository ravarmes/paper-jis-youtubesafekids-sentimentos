# Rubrica de anotação — análise de sentimentos

Define as três classes de polaridade do corpus `corpus.csv` e o que cada uma significa,
com exemplo. É o esquema contra o qual os quatro anotadores rotularam as 2.749 sentenças,
e é também o que os rótulos preditos pelo modelo querem dizer — as classes são as mesmas
dos dois lados.

Cada sentença recebe exatamente uma classe. O rótulo final da coluna `AS` sai por voto
majoritário entre as quatro rotulagens individuais (`P1`, `A1`, `A2`, `A3`); em caso de
empate, o docente (`P1`) arbitra.

## As três classes

- **0 — Negativo**
  Transmite tristeza, raiva, frustração, medo ou crítica.
  *Exemplo:* `"Você nunca faz nada direito!"`

- **1 — Neutro**
  Informação ou fala sem carga emocional relevante.
  *Exemplo:* `"O cachorro correu pelo quintal."`

- **2 — Positivo**
  Expressa alegria, empolgação, carinho ou elogio.
  *Exemplo:* `"Você é o melhor amigo que eu poderia ter!"`

## Critério de aplicação

A unidade de julgamento é a **sentença isolada**, como ela aparece na coluna `FRASE` —
sem o vídeo, sem o contexto das sentenças vizinhas e sem o título. Essa escolha é
deliberada e o artigo discute o que ela custa: parte dos erros do classificador vem de
enunciados que só se resolvem em contexto.

Casos ambíguos — sobretudo humor e ironia — passaram por revisão manual adicional.

---

> Tradução para o inglês em [`corpus-rubrica-en.md`](corpus-rubrica-en.md), como apoio de
> leitura — a anotação foi feita sobre este texto em português.

# Annotation rubric — sentiment analysis (English translation)

> **This is a translation, not the source document.** The rubric the annotation was carried
> out against is [`corpus-rubrica.md`](corpus-rubrica.md), in Brazilian Portuguese: the
> annotators worked from the Portuguese wording, and the class definitions must be audited
> there. This file is a reading aid. Every example is given in the original Portuguese, in
> italics, followed by an English gloss in single quotes — the same policy the article
> applies to its own examples.

Defines the three polarity classes of the `corpus.csv` corpus and what each one means, with
an example. It is the scheme against which the four annotators labeled the 2,749 sentences,
and it is also what the classes predicted by the model mean — the classes are the same on
both sides.

Each sentence receives exactly one class. The final label in the `AS` column is set by
majority vote among the four individual labelings (`P1`, `A1`, `A2`, `A3`); in case of a
tie, the faculty member (`P1`) acts as arbiter.

## The three classes

- **0 — Negative**
  Conveys sadness, anger, frustration, fear or criticism.
  *Example:* `"Você nunca faz nada direito!"` — 'You never do anything right!'

- **1 — Neutral**
  Information or speech without relevant emotional charge.
  *Example:* `"O cachorro correu pelo quintal."` — 'The dog ran across the yard.'

- **2 — Positive**
  Expresses joy, excitement, affection or praise.
  *Example:* `"Você é o melhor amigo que eu poderia ter!"` — 'You are the best friend I
  could have!'

## How it is applied

The unit of judgment is the **isolated sentence**, as it appears in the `FRASE` column —
without the video, without the context of neighboring sentences and without the title. The
choice is deliberate, and the article discusses what it costs: part of the classifier's
errors come from utterances that can only be resolved in context.

Ambiguous cases — above all humor and irony — underwent additional manual review.

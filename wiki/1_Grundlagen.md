# 1. Grundlagen & Begriffe

Um die Innovationen im Bereich der SLMs einzuordnen, ist ein klares Verständnis der Begrifflichkeiten notwendig.

## Foundation Models

Der Begriff "Foundation Model" wurde vom Stanford Institute for Human-Centered AI (HAI) geprägt. Ein **Foundation Model** ist ein KI-Modell, das auf einer sehr breiten Datenbasis (oft Petabytes an Text, Bildern, Audio) trainiert wurde und als "Allzweck-Basis" für verschiedene spezifische Aufgaben dient. [[1]](#quellen)

*   *Beispiel:* Ein Modell lernt, Sätze sinnvoll fortzuführen und Text im Stil der Trainingsdaten zu generieren — z.B. "Die Hauptstadt von Frankreich ist" → "Paris". Es besitzt breites Weltwissen, kann aber noch keine Anweisungen befolgen oder gezielte Aufgaben lösen. Dafür ist ein nachgelagerter Schritt (z.B. SFT) notwendig.
*   *Technologie:* Meist Transformer-Architekturen und **Self-Supervised Learning** — das Modell generiert seine Lernsignale aus den Daten selbst, z.B. durch Maskierung von Wörtern (BERT) oder Vorhersage des nächsten Tokens (GPT). [[1]](#quellen)
*   *Nicht nur Text:* Foundation Models umfassen auch Modelle für Bilder (z.B. Stable Diffusion), Audio (z.B. MusicGen) oder Robotik (z.B. RT-2). Der "AI Foundation Model Transparency Act" (USA, 2023) definiert sie als KI-Modelle mit mindestens einer Milliarde Parametern, die auf breiten Daten mit Selbstüberwachung trainiert wurden. [[1]](#quellen)

## Large Language Models (LLMs)

LLMs sind eine Unterklasse der Foundation Models, spezialisiert auf die Verarbeitung und Generierung natürlicher Sprache.

*   **Definition:** Meist Modelle mit mehr als 10 Milliarden Parametern.
*   **Architektur:** Fast alle modernen LLMs basieren auf dem **Transformer** (Google, 2017). [[2]](#quellen) Dessen Kern ist der **Self-Attention-Mechanismus**, der es dem Modell erlaubt, Beziehungen zwischen Wörtern unabhängig von deren Distanz zu gewichten — ein entscheidender Fortschritt gegenüber RNNs und LSTMs, die bei langen Texten "vergessen".
*   **Scaling Laws:** Mit zunehmender Größe (Parameter, Daten, Rechenleistung) steigt nicht nur die Leistung, sondern es **emergieren** ab bestimmten Schwellenwerten neue Fähigkeiten (logisches Schließen, Mathematik), die nicht explizit trainiert wurden. [[3]](#quellen)

## Evolution: Von GPT zu InstructGPT

Ein **Foundation Model** nach dem Pre-Training ist nur eine "Next-Token-Prediction"-Maschine — es vervollständigt Muster, versteht aber keine Anweisungen. Die Evolution zum nützlichen Assistenten verläuft in aufeinander aufbauenden Trainingsschritten:

1.  **Pre-Training → Foundation Model:** Das Modell lernt Sprache, indem es Terabytes an Text liest.
    *   *Prompt:* "Die Hauptstadt von Frankreich ist" → *Modell:* "Paris".
    *   *Problem:* Es versteht keine Anweisungen. Auf "Backe einen Kuchen" antwortet es vielleicht mit "Backe ein Brot", weil es Muster vervollständigt, nicht Intentionen. [[1]](#quellen)
2.  **SFT (Supervised Fine-Tuning) → Instruction-following Model:** Menschen schreiben Beispiel-Prompts mit idealen Antworten. Das Modell lernt, Anweisungen zu erkennen und zu befolgen — aus dem Textgenerator wird ein Assistent, der Aufgaben löst. Dies ist der entscheidende Schritt: Aus dem Foundation Model wird ein nutzbares Modell. [[8]](#quellen)
    *   *Aber:* Das Modell befolgt Anweisungen ohne Rücksicht auf Sicherheit oder menschliche Werte — auch schädliche Anfragen werden erfüllt.
3.  **Reward Modeling + RLHF → Aligned Assistant:** Um das Modell nicht nur fähig, sondern auch wertekonform zu machen, folgt ein weiterer Schritt: [[8]](#quellen)
    *   **Reward Modeling:** Das Modell generiert mehrere Antworten auf einen Prompt. Menschen bewerten diese per Ranking. Daraus wird ein **Belohnungsmodell** trainiert, das lernt, was Menschen bevorzugen.
    *   **RLHF (Reinforcement Learning from Human Feedback):** Das Sprachmodell wird mittels Reinforcement Learning (z.B. PPO-Algorithmus) optimiert, um die Bewertung des Belohnungsmodells zu maximieren. Das Ergebnis: Das Modell lernt, Antworten zu bevorzugen, die von Menschen als hilfreich, ehrlich und harmlos bewertet werden. [[9]](#quellen)

Diese Schritte transformieren LLMs von reinen Textgeneratoren zu den heute bekannten Chatbots und Assistenten.

## Grenzen der Skalierung — und die Nische für SLMs

Trotz der Erfolge der LLMs zeigen sich fundamentale Grenzen:

1.  **Daten-Knappheit:** Hochwertige Textdaten sind eine endliche Ressource. Modelle, die wahllos mit dem gesamten Internet trainiert werden, übernehmen auch Fehlinformationen und Bias.
2.  **Latenz:** Viele Anwendungen (Sprachassistenten im Auto, Code-Vervollständigung) benötigen Antworten in Millisekunden. Große Modelle sind dafür oft zu langsam. [[4]](#quellen)
3.  **Kosten und Energie:** Die Inferenzkosten skalieren mit der Modellgröße. Das Training erfordert Rechenzentren, der Energieverbrauch ist immens. [[5]](#quellen) Für viele Unternehmen rechnen sich die Kosten pro Token bei einem 175B-Modell nicht.
4.  **Datenschutz:** Die Größe erzwingt oft Cloud-APIs, was in datensensiblen Branchen (Finanzen, Gesundheit) aufgrund von DSGVO und Compliance problematisch ist. [[6]](#quellen)

Hier öffnet sich die Nische für **Small Language Models (SLMs)**: Modelle mit typischerweise unter 10 Milliarden Parametern, die nicht auf Leistung verzichten, sondern die Frage stellen: *"Wie viel Intelligenz kann in ein begrenztes Parameterbudget gepackt werden?"* [[7]](#quellen)

> **Warum ist das für SLMs wichtig?**
> SLMs durchlaufen dieselben Trainingsschritte — insbesondere SFT ist für jedes spezialisierte SLM essenziell. Da sie weniger "Speicherkapazität" (Parameter) haben, müssen sie durch **Qualität statt Quantität** trainiert werden.

---

## Quellen

1. Foundation model — Wikipedia. https://en.wikipedia.org/wiki/Foundation_model
2. Gemma: Open Models Based on Gemini — Google. https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf
3. Emergent Abilities of Large Language Models — Wei et al., 2022. https://arxiv.org/abs/2206.07682
4. SLMs vs LLMs — Datacamp. https://www.datacamp.com/blog/slms-vs-llms
5. Energy Considerations of LLM Inference — ACL Anthology. https://aclanthology.org/2025.acl-long.1563.pdf
6. AI, LLMs and Data Protection — Irish Data Protection Commission (DPC). https://www.dataprotection.ie/en/dpc-guidance/blogs/AI-LLMs-and-Data-Protection
7. Small language models vs. large language models — Invisible Technologies. https://invisibletech.ai/blog/how-small-language-models-can-outperform-llms
8. Ouyang et al. (2022): Training language models to follow instructions with human feedback — arXiv. https://arxiv.org/abs/2203.02155
9. Illustrating Reinforcement Learning from Human Feedback (RLHF) — Hugging Face. https://huggingface.co/blog/rlhf

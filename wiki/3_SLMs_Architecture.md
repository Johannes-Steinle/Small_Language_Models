# 3. Small Language Models: Architektur und Paradigmenwechsel

Small Language Models (typischerweise < 10 Mrd. Parameter) sind nicht einfach nur "geschrumpfte" LLMs. Um bei einer Größe von 2 bis 8 Milliarden Parametern konkurrenzfähig zu sein, bedarf es grundlegend anderer Ansätze in Bezug auf **Daten** und **Architektur**.

## Die "Textbooks Are All You Need"-Hypothese

Der wichtigste Treiber für die Leistungsfähigkeit moderner SLMs ist die **Qualität der Trainingsdaten**. Bisher wurden LLMs oft mit dem "Common Crawl" trainiert — einem riesigen Abbild des Internets, das viel Rauschen enthält: unvollständige Sätze, schlechter Code, Werbung, inkohärente Diskussionen.

Microsoft Research stellte mit der **Phi**-Modellreihe die Hypothese auf: *"Textbooks Are All You Need"*. [[1]](#quellen) Die Idee: Ein Modell lernt viel effizienter, wenn es (ähnlich wie ein menschlicher Schüler) mit didaktisch aufbereitetem, klarem und logischem Material trainiert wird, anstatt mit rohen Internetdaten.

Der Prozess der Datenkuratierung umfasst drei Stufen:

1.  **Filterung:** Bestehende Webdaten werden extrem rigoros gefiltert. Ein Klassifikator (oft selbst ein LLM) bewertet Texte nach ihrem "edukativen Wert". Nur Inhalte, die logische Schlussfolgerungen fördern und faktenbasiert sind, bleiben erhalten. [[2]](#quellen)
2.  **Synthetische Daten:** Da es nicht genügend natürliche "Lehrbücher" gibt, werden große LLMs (wie GPT-4) genutzt, um synthetische Daten zu generieren. Für das Phi-1-Modell (Coding) wurden beispielsweise Millionen von Python-Übungen generiert, die spezifische Konzepte abdecken.
3.  **Curriculum Learning:** Das Training beginnt mit einfacheren Konzepten und steigert die Komplexität langsam — analog zum menschlichen Lernen.

Dieser datenzentrierte Ansatz ermöglicht es Modellen wie Phi-3-Mini (3.8B), in Benchmarks Modelle zu schlagen, die deutlich größer sind. [[3]](#quellen)

## Architektonische Effizienz

Da SLMs oft auf Geräten mit begrenztem Speicher (Smartphones, Laptops mit 16 GB RAM) laufen sollen, ist die Speichereffizienz während der **Inferenz** (Laufzeit) kritisch.

### Attention-Mechanismus — Query, Key, Value

Um die folgenden Optimierungen zu verstehen, ist ein Blick auf die Komponenten des **Attention-Mechanismus** notwendig. In der Transformer-Architektur wird jeder Token in drei Vektoren projiziert: [[4]](#quellen)

*   **Query (Q):** Repräsentiert die aktuelle Anfrage eines Tokens (*"Wonach suche ich im Kontext?"*).
*   **Key (K):** Dient als Identifikator (*"Was bin ich?"*). Das Skalarprodukt zwischen Query und Key bestimmt die Relevanz (Attention Score).
*   **Value (V):** Enthält die eigentliche inhaltliche Information, die weitergegeben wird, wenn Query und Key übereinstimmen.
*   **Heads (Attention Heads):** Um verschiedene sprachliche Aspekte (z.B. grammatikalische Bezüge in einem Head, semantische in einem anderen) gleichzeitig zu erfassen, wird dieser Prozess parallel in mehreren "Köpfen" ausgeführt.

### Grouped-Query Attention (GQA)

In der klassischen **Multi-Head Attention (MHA)** hat jeder Head eigene Matrizen für Keys und Values. Diese müssen während der Textgenerierung im Speicher gehalten werden (der sogenannte **KV-Cache**). Bei langen Texten wächst dieser Cache enorm.

**GQA** ist ein Mittelweg: Mehrere Query-Heads teilen sich einen einzigen Key-Value-Head. [[4]](#quellen) Dies reduziert die Größe des KV-Caches drastisch (oft um Faktor 8 oder mehr), was es SLMs erlaubt, auch lange Kontexte (z.B. 128k Token) auf Consumer-Hardware zu verarbeiten.

### Sliding Window Attention (z.B. Gemma 3)

Anstatt dass jeder Token auf **alle** vorherigen Token "achtet" (quadratischer Aufwand), nutzen Modelle wie Gemma 3 eine **Alternating Local and Global Attention**: [[4]](#quellen)

*   Die Mehrheit der Schichten beschränkt sich auf ein **lokales Fenster** (z.B. nur die letzten 1024 Token).
*   Im Verhältnis **5:1** behält jede sechste Schicht die **globale Attention**, um den Gesamtzusammenhang nicht zu verlieren.
*   Diese Hybrid-Architektur spart Rechenzeit und Speicher, da die Aufmerksamkeitsmatrix in den lokalen Schichten viel kleiner ist — und ermöglicht so Kontextlängen von 128k Token.

## Knowledge Distillation (Wissensdestillation)

Hierbei dient ein großes "Teacher"-Modell (z.B. GPT-4 oder Llama-3-70B) als Lehrer für das kleine "Student"-Modell (SLM). [[5]](#quellen) Es gibt drei Methoden:

1.  **Response-based:** Der Student lernt, die **Ausgaben** des Lehrers zu reproduzieren. Dies ist die Basis für das Training mit synthetischen Daten.
2.  **Logit-based (Soft Targets):** Der Student lernt nicht nur die "harte" Antwort (z.B. "Das ist eine Katze"), sondern die **Wahrscheinlichkeitsverteilung** des Lehrers (z.B. "90% Katze, 9% Hund, 1% Auto"). Diese "Soft Labels" enthalten Informationen über die Ähnlichkeit von Konzepten ("Dunkelwissen"), was dem Studenten hilft, besser zu generalisieren. [[6]](#quellen)
3.  **Step-by-Step Distillation:** Der Lehrer generiert nicht nur die Antwort, sondern auch den **Gedankengang** ("Chain of Thought"). Der Student lernt die Logik der Herleitung — besonders wichtig für Reasoning-Aufgaben in SLMs. [[7]](#quellen)

## Vergleich der Ökosysteme: LLM vs. SLM

| Merkmal | LLM (z.B. Llama-3-70B) | SLM (z.B. Phi-4-mini, Gemma-3-4B) |
| :--- | :--- | :--- |
| **Primärer Fokus** | Kapazität & Breite (Weltwissen) | Effizienz & Spezialisierung |
| **Datenstrategie** | Quantität ("The Pile", Common Crawl) | Qualität & Dichte ("Textbooks", Filtered) |
| **Training** | Compute-bound (Monate auf Clustern) | Data-bound (Wochen/Tage) |
| **Architektur** | Standard MHA (meistens) | Optimiert (GQA, Sliding Window, Tied Embeddings) |
| **Deployment** | Cloud / Multi-GPU Cluster | Edge / Single-GPU / CPU |

---

## Quellen

1. Phi-3 Technical Report — arXiv. https://arxiv.org/pdf/2404.14219
2. Textbooks Are All You Need — arXiv. https://arxiv.org/abs/2306.11644v1
3. Introducing Phi-3 — Microsoft Azure Blog. https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/
4. Gemma explained: What's new in Gemma 3 — Google Developers Blog. https://developers.googleblog.com/en/gemma-explained-whats-new-in-gemma-3/
5. Knowledge Distillation for LLMs — Zilliz Learn. https://zilliz.com/learn/knowledge-distillation-from-large-language-models-deep-dive
6. Distilling the Knowledge in a Neural Network — Hinton et al., 2015. https://arxiv.org/abs/1503.02531
7. Symbolic Chain-of-Thought Distillation: Small Models Can Also "Think" Step-by-Step — Li et al., ACL 2023. https://arxiv.org/abs/2306.14050

# 4. Fallstudie: Google Gemma 3 — Ein modernes SLM

Google Gemma dient als ideales Beispiel, um die Theorie in die Praxis zu übertragen. Gemma ist eine Familie von "Open Weights"-Modellen, die technologisch auf Googles Gemini-Modellen basieren. [[1]](#quellen)

## Architektur und Evolution

Gemma 3 ist ein **Decoder-only Transformer**. Zentrale Eigenschaften:

*   **Vokabular:** Extrem groß mit **262.144 Token** (Gemini 2.0 SentencePiece-Tokenizer). Dies ermöglicht eine effiziente Kompression von Texten in über 140 Sprachen und im Code-Bereich. [[1]](#quellen)
*   **Lizenz:** Gemma Terms of Use (Open Weights) — das Modell kann heruntergeladen und lokal betrieben werden.
*   **Multimodalität:** Ab der 4B-Variante integriert Gemma 3 einen **SigLIP Vision Encoder** (~400M Parameter), der Text- und Bildverarbeitung in einem Modell vereint. [[2]](#quellen)

### Gemma 3: Architektur-Innovation

Mit **Gemma 3** führte Google tiefgreifende architektonische Änderungen ein, die die Vorgängerversion (Gemma 2) in allen Bereichen übertreffen: [[2]](#quellen)

1.  **QK-Norm statt Soft-Capping:** Gemma 2 nutzte Logit Soft-Capping (Tanh-Funktion), um extreme Werte zu begrenzen. Gemma 3 ersetzt dies durch **Query-Key-Normalisierung (QK-Norm)** — eine RMSNorm auf die Query- und Key-Vektoren vor der Attention-Berechnung. Dies stabilisiert das Training effizienter und ist kompatibel mit optimierten Attention-Implementierungen wie FlashAttention. [[2]](#quellen)

2.  **Sliding Window Attention (5:1 Ratio):** Gemma 3 nutzt ein deutlich kleineres lokales Fenster von nur **1024 Token** (Gemma 2: 4096) bei einem Verhältnis von **5:1** — fünf lokale Schichten pro eine globale Schicht. Dies reduziert den KV-Cache-Speicherbedarf massiv und ermöglicht Kontextlängen von **128k Token** (Gemma 2: 8k) auf Consumer-Hardware (siehe auch [Kapitel 3](3_SLMs_Architecture)). [[2]](#quellen)

## Leistungsvergleich: Gemma 3, Phi-4-mini, Qwen3

Die folgende Tabelle vergleicht aktuelle SLMs anhand gängiger Benchmarks. Die Quellenangabe pro Modell steht im Spaltenkopf — alle Werte einer Spalte stammen aus derselben Quelle.

| Metrik / Modell | Gemma 3 4B-IT [[1]](#quellen) | Phi-4-mini-instruct [[4]](#quellen) | Qwen3-8B [[6]](#quellen) |
| :--- | :--- | :--- | :--- |
| **Parameter** | ~4,3 Mrd. | 3,8 Mrd. | 8,2 Mrd. |
| **MMLU (Wissen)** | 58,1% | 67,3% | **76,9%**\* |
| **MATH (Mathe)** | **75,6%** | 64,0% | 87,4%\*\* |
| **HumanEval (Code)** | 71,3% | **74,4%** | ~67,7%\*\*\* |
| **Max. Kontextlänge** | 128k | 128k | 128k |

\* Qwen3: MMLU des Base-Modells (5-shot). Gemma 3 und Phi-4-mini: Instruct-Modell.
\*\* Qwen3: MATH-500 im Non-Thinking-Modus; mit Thinking Mode 97,4%. Gemma 3 und Phi-4-mini berichten den vollen MATH-Benchmark.
\*\*\* Qwen3: EvalPlus-Durchschnitt (HumanEval, MBPP, HumanEval+, MBPP+) des Base-Modells.

### Analyse der Ergebnisse

*   **Qwen3** dominiert beim allgemeinen Wissen (MMLU) und verfügt über einen **Hybrid Thinking Mode**, der Chain-of-Thought-Reasoning bei Bedarf aktiviert und die MATH-Scores auf 97,4% hebt. Die enormen 36 Billionen Trainingstoken machen sich in der Wissensbreite bemerkbar.
*   **Phi-4-mini** erreicht mit nur 3,8 Milliarden Parametern die stärksten Code-Ergebnisse (HumanEval) und beweist damit erneut die Wirksamkeit des datenzentrierten Ansatzes mit synthetischen Trainingsdaten.
*   **Gemma 3** erzielt den stärksten MATH-Score unter den Standard-Instruct-Modellen (ohne Thinking Mode) und bringt als einziges der drei Modelle **Multimodalität** (Bild + Text) bereits ab 4B Parametern mit.

### Vergleich der Kern-Philosophien

| Feature | Google Gemma 3 | Microsoft Phi-4-mini | Alibaba Qwen3 (8B) |
| :--- | :--- | :--- | :--- |
| **Kern-Philosophie** | Architektur-Innovation (QK-Norm, Hybrid Attention, Multimodal) | Daten-Qualität (Synthetic Data, "Textbooks") | Skalierung & Hybrid Thinking (36T Token, Thinking Mode) |
| **Vokabular** | 262.144 Token | 200.000 Token | 151.936 Token |
| **Attention** | Sliding Window + Global (5:1) | Grouped-Query Attention (GQA) | Grouped-Query Attention (GQA) |
| **Besonderheiten** | QK-Norm, SigLIP Vision Encoder, 128k Kontext | MoLoRA (Multimodal-Adapter), aggressive Datenfilterung | Hybrid Thinking Mode (Think/No-Think), 36T Trainingstoken |
| **Lizenz** | Gemma Terms (Open Weights) | MIT License (Open Source) | Apache 2.0 (Open Source) |

> **Fazit:** Es gibt kein "bestes" SLM — nur das passende Werkzeug für den jeweiligen Zweck. Gemma 3 glänzt als multimodaler Allrounder mit innovativer Architektur, Phi-4-mini überzeugt durch maximale Effizienz pro Parameter dank synthetischer Daten, und Qwen3 setzt mit seinem Hybrid Thinking Mode neue Maßstäbe im Reasoning.

---

## Quellen

1. Gemma 3 Technical Report — Google DeepMind. https://arxiv.org/abs/2503.19786
2. Gemma explained: What's new in Gemma 3 — Google Developers Blog. https://developers.googleblog.com/en/gemma-explained-whats-new-in-gemma-3/
3. Gemma 3 4B Model Card — Hugging Face. https://huggingface.co/google/gemma-3-4b-it
4. Phi-4-mini Technical Report — Microsoft. https://arxiv.org/abs/2503.01743
5. Qwen3: Think Deeper, Act Faster — Qwen Blog. https://qwenlm.github.io/blog/qwen3/
6. Qwen3 Technical Report — Alibaba. https://arxiv.org/abs/2505.09388

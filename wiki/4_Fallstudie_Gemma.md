# 4. Fallstudie: Google Gemma — Ein modernes SLM

Google Gemma dient als ideales Beispiel, um die Theorie in die Praxis zu übertragen. Gemma ist eine Familie von "Open Weights"-Modellen, die technologisch auf Googles Gemini-Modellen basieren. [[1]](#quellen)

## Architektur und Evolution

Gemma ist ein **Decoder-only Transformer**. Zentrale Eigenschaften:

*   **Vokabular:** Extrem groß mit 256.000 Token (vgl. Llama 3 mit 128k, Phi-3 mit 32k). Dies ermöglicht eine effiziente Kompression von Texten, insbesondere in anderen Sprachen als Englisch und im Code-Bereich. [[1]](#quellen)
*   **Lizenz:** Gemma Terms of Use (Open Weights) — das Modell kann heruntergeladen und lokal betrieben werden.

### Gemma 2: Signifikante Verbesserungen

Mit **Gemma 2** führte Google architektonische Änderungen ein, die die Leistung drastisch steigerten: [[2]](#quellen)

1.  **Logit Soft-Capping:** In großen Modellen können die Logits (Werte vor der Aktivierungsfunktion) extrem groß werden, was das Training instabil macht. Gemma 2 begrenzt diese Werte sanft mittels einer Tanh-Funktion:

    $$\text{logits} = C \cdot \tanh(x / C)$$

    Dabei ist $x$ der ursprüngliche Logit-Wert und $C$ eine Konstante, die die Obergrenze definiert (Soft-Cap). Durch die Tanh-Funktion werden extreme Werte sanft begrenzt, ohne sie hart abzuschneiden. Dies stabilisiert das Training und verbessert die Qualität der generierten Texte.

2.  **Alternating Local and Global Attention:** Gemma 2 wechselt zwischen lokaler (Sliding Window, z.B. 4096 Token) und globaler Aufmerksamkeit. Dies reduziert den Speicherbedarf drastisch, ohne den Gesamtzusammenhang zu verlieren (siehe auch [Kapitel 3](3_SLMs_Architecture)).

## Leistungsvergleich: Gemma 2, Llama 3, Phi-3

Die folgende Tabelle vergleicht die aktuellen SLMs anhand gängiger Benchmarks: [[3]](#quellen) [[4]](#quellen)

| Metrik / Modell | Gemma 2 (9B) | Llama 3.1 (8B) | Phi-3-Mini (3.8B) |
| :--- | :--- | :--- | :--- |
| **Parameter** | 9,2 Mrd. | 8,0 Mrd. | 3,8 Mrd. |
| **MMLU (Wissen)** | **71,3%** | 69,4% | ~69% |
| **GSM8K (Mathe)** | 68,6% | **84,5%** | 82,6% |
| **HumanEval (Code)** | 40,2% | **72,6%** | 58,5% |
| **Max. Kontextlänge** | 8k | 128k | 128k |

### Analyse der Ergebnisse

*   **Gemma 2** dominiert beim allgemeinen Wissen (MMLU). Dies deutet darauf hin, dass die Destillation vom großen Gemini-Modell viel Weltwissen übertragen hat.
*   **Llama 3** ist führend bei "harten" Fähigkeiten wie Programmieren und Mathe. Dies spiegelt Metas Fokus auf große Mengen an Code-Daten im Pre-Training wider.
*   **Phi-3** ist das beeindruckendste Modell in Bezug auf **Effizienz**: Mit weniger als halb so vielen Parametern erreicht es in Mathe (GSM8K) Werte, die fast an Llama 3 heranreichen und Gemma 2 schlagen. Dies ist der praktische Beweis für die "Textbooks"-Hypothese.

### Vergleich der Kern-Philosophien

| Feature | Google Gemma 2 | Microsoft Phi-3 | Meta Llama 3 (8B) |
| :--- | :--- | :--- | :--- |
| **Kern-Philosophie** | Architektur-Innovation (Hybrid Attention) | Daten-Qualität ("Textbooks") | Skalierung & Menge (Trainingstoken) |
| **Vokabular** | 256.000 Token | 32.000 Token | 128.000 Token |
| **Attention** | Sliding Window + Global (Hybrid) | Standard (MHA/GQA) | Grouped-Query Attention (GQA) |
| **Besonderheiten** | Logit Soft-Capping, GeGLU | Extrem aggressive Datenfilterung | Fokus auf Code & Math |
| **Lizenz** | Gemma Terms (Open Weights) | MIT License (Open Source) | Llama Community License |

> **Fazit:** Es gibt kein "bestes" SLM — nur das passende Werkzeug für den jeweiligen Zweck. Gemma 2 glänzt als Allrounder mit starkem Weltwissen, Phi-3 überzeugt durch maximale Effizienz pro Parameter, und Llama 3 ist die erste Wahl für Code und Mathematik.

---

## Quellen

1. Gemma: Open Models Based on Gemini — Google. https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf
2. Gemma explained: What's new in Gemma 2 — Google Developers Blog. https://developers.googleblog.com/en/gemma-explained-new-in-gemma-2/
3. Gemma 2 vs LLaMA 3: Which AI Model Wins? — Kanerika. https://kanerika.com/blogs/gemma-2-vs-llama-3/
4. Gemma 2 9B vs Llama 3.1 8B Instruct — LLM Stats. https://llm-stats.com/models/compare/gemma-2-9b-it-vs-llama-3.1-8b-instruct

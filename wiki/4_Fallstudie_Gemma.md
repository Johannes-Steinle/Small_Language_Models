# 4. Fallstudie: Google Gemma

Google Gemma dient als ideales Beispiel, um die Theorie in die Praxis zu übertragen. Es basiert auf der Gemini-Technologie.

## Architektur
*   **Vokabular:** Extrem groß (256.000 Token). Das hilft bei mehrsprachigen Texten und Code.
*   **Gemma 2 Verbesserungen:**
    *   *Logit Soft-Capping:* Eine mathematische Bremse, die das Training stabilisiert.
    *   *Alternating Attention:* Wechsel zwischen lokalem und globalem Fokus.

## Vergleich: Gemma 2 vs. Llama 3 vs. Phi-3

| Modell | Stärke | Ideal für... |
| :--- | :--- | :--- |
| **Gemma 2 (9B)** | Allgemeinwissen, Nuancen | Chatbots, RAG, Kreatives Schreiben |
| **Llama 3 (8B)** | Hard Skills (Mathe, Code) | Coding-Assistenten, Logik-Aufgaben |
| **Phi-3 (3.8B)** | Effizienz pro Parameter | Mobile Geräte, Edge-Computing |

Das Fazit unserer Untersuchung: Es gibt kein "bestes" SLM, nur das passende Werkzeug für den jeweiligen Zweck. Gemma 2 glänzt als Allrounder.

# 3. Die Ära der SLMs

Small Language Models (< 10 Mrd. Parameter) verfolgen eine andere Philosophie als ihre großen Brüder: **Effizienz durch Qualität.**

## Die "Textbooks Are All You Need" Hypothese
Microsofts Phi-Modelle zeigten: Ein Modell lernt viel schneller und besser, wenn es mit **hochwertigen "Lehrbuch"-Daten** trainiert wird, statt mit dem "rohen" Internet (Common Crawl).
*   **Daten-Filterung:** Alles Unwichtige (Werbung, schlechter Code) wird rigoros entfernt.
*   **Synthetische Daten:** GPT-4 schreibt "Lehrbücher" für das kleine Modell, um Konzepte zu erklären.

## Archtiektonische Tricks

### Grouped-Query Attention (GQA) & Sliding Window
Um Speicher zu sparen, erinnern sich SLMs selektiver an den Textkontext.
*   **GQA:** Mehrere "Attention Heads" teilen sich den Speicher. Das reduziert den VRAM-Verbrauch bei langen Texten massiv.
*   **Sliding Window (z.B. Gemma 2):** Das Modell schaut teilweise nur auf die letzten paar tausend Wörter zurück ("lokales Fenster") und nutzt nur wenige "globale" Blicke für den Gesamtzusammenhang.

## Knowledge Distillation (Wissensdestillation)
Wie in einer Schule lernt ein Schüler (Student Model, SLM) von einem Lehrer (Teacher Model, z.B. GPT-4).
*   Das SLM lernt nicht nur die *Antwort* des Lehrers, sondern versucht, dessen *Gedankengänge* (Chain-of-Thought) und Wahrscheinlichkeitsverteilungen nachzuahmen.

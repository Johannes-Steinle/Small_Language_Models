# 1. Grundlagen & Begriffe

Um die Innovationen im Bereich der SLMs einzuordnen, ist ein klares Verständnis der Begrifflichkeiten notwendig.

## Foundation Models
Ein **Foundation Model** ist ein KI-Modell, das auf einer sehr breiten Datenbasis (oft Petabytes an Text/Bildern) trainiert wurde und als "Allzweck-Basis" für verschiedene spezifische Aufgaben dient. 
*   *Beispiel:* Ein Modell kann Texte zusammenfassen, Code schreiben oder Fragen beantworten, ohne für eines davon exklusiv trainiert worden zu sein.
*   *Technologie:* Meist Transformer-Architekturen und Self-Supervised Learning (Lernen aus den Daten selbst, z.B. "Vorhersage des nächsten Wortes").

## Large Language Models (LLMs)
LLMs sind eine Unterklasse der Foundation Models, spezialisiert auf Sprache.
*   **Definition:** Meist Modelle > 10 Milliarden Parameter.
*   **Eigenschaft:** "Scaling Laws" besagen, dass mehr Daten + mehr Parameter = mehr Intelligenz.
*   **Emergenz:** Ab einer gewissen Größe entstehen Fähigkeiten wie logisches Schließen, die nicht explizit programmiert wurden.

## Evolution: Von GPT zu InstructGPT
Ein reines Sprachmodell ist nur eine "Next-Token-Prediction"-Maschine. Um es zum Assistenten zu machen, sind weitere Schritte nötig:

1.  **Pre-Training (GPT):** Das Modell lernt Sprache, indem es Terabytes an Text liest.
    *   *Prompt:* "Die Hauptstadt von Frankreich ist" -> *Modell:* "Paris".
    *   *Problem:* Es versteht keine Anweisungen. Auf "Backe einen Kuchen" antwortet es vielleicht mit "Backe ein Brot", weil es Muster vervollständigt.
2.  **InstructGPT (Alignment):**
    *   **SFT (Supervised Fine-Tuning):** Menschen schreiben perfekte Antworten auf Fragen. Das Modell lernt diese nachzuahmen.
    *   **RLHF (Reinforcement Learning from Human Feedback):** Menschen bewerten Antworten (Ranking). Ein "Belohnungsmodell" trainiert das Sprachmodell, bevorzugte Antworten zu geben.

> [!NOTE]
> **Warum ist das für SLMs wichtig?**
> SLMs nutzen diese Techniken extrem effizient. Da sie weniger "Speicherkapazität" (Parameter) haben, müssen sie durch **Qualität statt Quantität** trainiert werden.

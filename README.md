# Small Language Models (SLM) Project

Willkommen zum Repository der Projektarbeit "Small Language Models".

## Inhalt
Dieses Repository enthält die praktische Implementierung und Dokumentation zu SLMs, mit einem Fokus auf **Google Gemma**.

### Dokumentation (Wiki)
Die theoretischen Grundlagen, Analysen und Hintergründe finden Sie im Ordner `wiki/`.
Wir empfehlen, die Dateien in der korrekten Reihenfolge zu lesen:
1. `wiki/Home.md`
2. `wiki/1_Grundlagen.md`
3. ...

### Code (Notebooks)
Der praktische Teil befindet sich in `notebooks/`.
Die Notebooks sind für **Google Colab** optimiert (laufen aber auch lokal mit GPU).

*   **`SLM_Finetuning_Demo.ipynb`**: Zeigt, wie man Gemma-2-2B mittels QLoRA auf einem eigenen Datensatz nachtrainiert.
*   **`SLM_Inference_Demo.ipynb`**: Ein einfacher Chat-Bot mit Gemma.

## Installation (Lokal)
Falls Sie den Code lokal ausführen möchten (statt Colab):

```bash
pip install -r requirements.txt
```

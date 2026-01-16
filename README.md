# Small Language Models (SLM) Project

Willkommen zum Repository der Projektarbeit "Small Language Models".

## Inhalt
Dieses Repository enthält die praktische Implementierung und Dokumentation zu SLMs, mit einem Fokus auf **Google Gemma**.

### Dokumentation (Wiki)
Die theoretischen Grundlagen, Analysen und Hintergründe finden sich im [Wiki](https://github.com/Johannes-Steinle/Small_Language_Models/wiki) sowie im Ordner `wiki/`.

### Code (Notebooks)
Der praktische Teil befindet sich in `notebooks/`. Die Notebooks sind für **Google Colab** optimiert und können direkt per Klick geöffnet werden:

| Notebook | Beschreibung | |
| :--- | :--- | :--- |
| **SLM_Finetuning_Demo** | Fine-Tuning von Gemma 3 mittels QLoRA + Inferenz-Test | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Johannes-Steinle/Small_Language_Models/blob/main/notebooks/SLM_Finetuning_Demo.ipynb) |

> **Voraussetzungen:** Ein [Hugging Face Account](https://huggingface.co/join) mit akzeptierten [Gemma 3 Nutzungsbedingungen](https://huggingface.co/google/gemma-3-4b-it) und ein Access Token (Read).

## Installation (Lokal)
Falls der Code lokal ausgeführt werden soll (statt Colab):

```bash
pip install -r requirements.txt
```

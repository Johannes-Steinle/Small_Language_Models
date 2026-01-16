# 5. Praxis: Fine-Tuning, Inferenz und Deployment

Dieser Projektteil zeigt, wie ein SLM (Gemma) angepasst, betrieben und eingesetzt werden kann — von der Anpassung auf eigenen Daten bis hin zum Deployment auf Edge-Geräten.

## Der Fine-Tuning-Workflow mit QLoRA

Das Anpassen eines SLMs an spezifische Unternehmensdaten (z.B. technische Dokumentationen) ist dank der Bibliotheken von Hugging Face (`transformers`, `peft`, `trl`) standardisiert. Der effizienteste Weg ist die Nutzung von **QLoRA**. [[1]](#quellen)

### Schritt 1: Modell in 4-Bit laden

Das Basismodell (z.B. `google/gemma-3-4b-it`) wird mit der `BitsAndBytesConfig` im NF4-Format geladen, was den VRAM-Verbrauch drastisch senkt (von ca. 10 GB auf ~3–4 GB):

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

### Schritt 2: LoRA-Adapter definieren

Anstatt das ganze Modell zu trainieren, werden die LoRA-Adapter konfiguriert. Typischerweise wird auf die Attention-Module abgezielt. Der Parameter `r` (Rank) steuert die Komplexität der Anpassung:

```python
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type="CAUSAL_LM"
)
```

### Schritt 3: Training mit SFTTrainer

Die `trl`-Bibliothek (Transformer Reinforcement Learning) bietet den `SFTTrainer`, der den komplexen Trainings-Loop abstrahiert. Man übergibt das Modell, den Datensatz (z.B. Frage-Antwort-Paare) und die PEFT-Konfiguration. [[2]](#quellen)

Dieser Workflow ermöglicht es, ein Sprachmodell auf einer einzelnen Consumer-Grafikkarte (z.B. NVIDIA RTX 3090 oder 4090) innerhalb weniger Stunden an eine neue Domäne anzupassen.

### Notebooks — direkt in Google Colab öffnen

| Notebook | Beschreibung | |
| :--- | :--- | :--- |
| **Fine-Tuning & Inferenz Demo** | Gemma 3 mit QLoRA fine-tunen und anschließend testen | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Johannes-Steinle/Small_Language_Models/blob/main/notebooks/SLM_Finetuning_Demo.ipynb) |

Nach dem Öffnen in Colab muss lediglich die Laufzeit auf **T4 GPU** gestellt und ein **Hugging Face Token** eingegeben werden. Danach können alle Zellen nacheinander ausgeführt werden ("Run All").

> **Hinweis:** Es wird ein [Hugging Face Account](https://huggingface.co/join) mit akzeptierten [Gemma 3 Nutzungsbedingungen](https://huggingface.co/google/gemma-3-4b-it) benötigt.

## Inferenz-Optimierung und Edge Deployment

Nach dem Training stellt sich die Frage des Betriebs. Hier zeigen SLMs ihre wahre Stärke: Während LLMs oft GPU-Cluster benötigen, können SLMs auf einer Vielzahl von Geräten laufen.

### GGUF und CPU-Inferenz

Mittels Quantisierung können Modelle in das **GGUF**-Format konvertiert werden (für CPU-Inferenz via `llama.cpp`). Ein Gemma-3-4B-Modell, quantisiert auf 4-Bit, benötigt etwa **3–4 GB Arbeitsspeicher** und läuft auf nahezu jedem modernen Rechner. [[3]](#quellen)

### Mobile und Browser

*   **Android/iOS:** Google bietet mit **Gemini Nano** und der MediaPipe-Bibliothek Möglichkeiten, Modelle direkt auf Smartphones laufen zu lassen. [[4]](#quellen)
*   **Browser:** Mit WebGPU und Bibliotheken wie WebLLM können SLMs vollständig im Browser des Nutzers laufen, ohne dass Daten an einen Server gesendet werden.

Dies ermöglicht Anwendungsfälle, die vorher undenkbar waren: ein voll funktionsfähiger Offline-Chatbot für Wartungstechniker ohne Internetzugang, oder ein medizinischer Assistent auf einem Tablet, der Patientendaten lokal analysiert.

## Ökonomische und Ökologische Bewertung

Der Einsatz von SLMs ist nicht nur eine technische, sondern auch eine **strategische** Entscheidung.

### Energieeffizienz ("Green AI")

Studien zeigen, dass SLMs bei der Inferenz bis zu **70% weniger Energie** verbrauchen als LLMs bei vergleichbaren Aufgaben. [[5]](#quellen) In einer hybriden Architektur ("Hybrid AI") kann ein SLM als "Vorschaltgerät" dienen: Es beantwortet 80% der einfachen Anfragen kostengünstig und schnell. Nur bei komplexen, reasoning-intensiven Anfragen wird das teure LLM hinzugezogen. [[6]](#quellen)

### Kostenstruktur

Die Kosten pro 1 Million Token sind bei eigenen SLMs oft vernachlässigbar im Vergleich zu API-Kosten großer Anbieter. Ein Unternehmen, das täglich Millionen von Dokumenten klassifizieren muss, spart durch ein lokales, spezialisiertes SLM massive Betriebskosten gegenüber GPT-4-API-Aufrufen. [[3]](#quellen)

### Datensouveränität

Das vielleicht stärkste Argument: In einem geopolitisch unsicheren Umfeld und unter strengen Regulierungen wie dem **EU AI Act** ist die Abhängigkeit von US-amerikanischen Cloud-Modellen ein Risiko. SLMs ermöglichen **"Sovereign AI"** — KI, die vollständig unter der Kontrolle des Betreibers steht, auf eigener Hardware läuft, und deren Trainingsdaten und Outputs das Unternehmen nie verlassen. [[7]](#quellen)

## Fazit und Ausblick

Die Analyse in diesem Projekt zeigt: Small Language Models sind weit mehr als eine Kompromisslösung für schwache Hardware. Sie repräsentieren einen **Reifeprozess** der KI-Industrie. Der blinde Glaube an "Größer ist Besser" weicht einer differenzierten Betrachtung von Effizienz, Datenqualität und Anwendungszweck.

*   **LoRA und Quantisierung** haben die Eintrittsbarrieren für KI-Anpassung demokratisiert. Ein Student mit einem Gaming-Laptop kann heute Modelle fine-tunen, für die vor drei Jahren Supercomputer nötig waren.
*   Der **datenzentrierte Ansatz** von Phi-3 hat bewiesen, dass das Potenzial kleiner Netze lange unterschätzt wurde, weil sie mit "schlechten" Daten gefüttert wurden.
*   Für die Zukunft zeichnet sich der Trend zu **Agentic SLMs** ab: Kleine Modelle, die spezialisiert sind auf Tool-Nutzung, API-Aufrufe und Orchestrierung. Anstatt alles Weltwissen in den Gewichten zu speichern, lernt das SLM nur die Logik der Informationsbeschaffung (**RAG — Retrieval Augmented Generation**).

Während LLMs weiterhin die Grenzen der künstlichen Intelligenz erforschen, sind es die SLMs, die KI in die Breite der Gesellschaft und Wirtschaft bringen — **effizient, bezahlbar und privat**.

---

## Quellen

1. Finetune Gemma with peft, 4-bit Quantized LoRA — Kaggle. https://www.kaggle.com/code/harishiker99/finetune-gemma-with-peft-4-bit-quantized-lora
2. Fine-tuning with the Hugging Face ecosystem (TRL) — AMD ROCm. https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/fine_tuning_lora_qwen2vl.html
3. Gemma 3 4B Model Card — Hugging Face. https://huggingface.co/google/gemma-3-4b-it
4. Gemma 3n model overview — Google AI for Developers. https://ai.google.dev/gemma/docs/gemma-3n
5. AI's Environmental Cost: Comparing Resource Consumption Between SLMs and LLMs — ICAIR. https://papers.academic-conferences.org/index.php/icair/article/view/4345
6. NVIDIA Research Proves Small Language Models Superior to LLMs — Galileo AI. https://galileo.ai/blog/small-language-models-nvidia
7. SLM series — Domino Data Lab: Distillation brings LLM power to SLMs — Computer Weekly. https://www.computerweekly.com/blog/CW-Developer-Network/SLM-series-Domino-Data-Lab-Distillation-brings-LLM-power-to-SLMs

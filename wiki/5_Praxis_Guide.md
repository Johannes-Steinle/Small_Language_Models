# 5. Praxis: Fine-Tuning Guide

In diesem Projektteil zeigen wir, wie man ein SLM (Gemma) anpasst.

## Der Workflow (QLoRA)
Wir nutzen **Google Colab** (oder jede andere Jupyter-Umgebung), da es kostenlosen GPU-Zugriff bietet.

Das Notebook `notebooks/SLM_Finetuning_Demo.ipynb` führt folgende Schritte aus:

1.  **Setup:** Installation von `bitsandbytes` (Quantisierung) und `peft` (LoRA).
2.  **Laden:** Wir laden Gemma-2-2B in **4-Bit**. Das reduziert den VRAM-Verbrauch von ~6GB auf unter 2GB.
3.  **Config:** Wir definieren die LoRA-Adapter (Rank $r=16$).
4.  **Training:** Der `SFTTrainer` (Supervised Fine-Tuning) passt die Adapter an einem Mini-Datensatz an.

## Wie starte ich die Demo?

1.  Öffnen Sie das Notebook in Google Colab (Link im Notebook oder via Datei-Upload).
2.  Stellen Sie sicher, dass die Laufzeit auf **T4 GPU** eingestellt ist.
3.  Geben Sie beim Start Ihren **Hugging Face Token** ein (da Gemma ein geschütztes Modell ist).
4.  Führen Sie alle Zellen nacheinander aus ("Run All").

> [!WARNING]
> Für das Training benötigen Sie einen Hugging Face Account und müssen den Nutzungsbedingungen von Gemma zugestimmt haben.

# Small Language Models — Projekt-Wiki

## Motivation

Large Language Models (LLMs) wie GPT-4 beeindrucken durch ihre Fähigkeiten, stellen aber aufgrund ihres massiven Ressourcenbedarfs eine Hürde für viele Unternehmen dar. Das Training erfordert Rechenzentren, die Nutzung ist auf Cloud-APIs angewiesen, und in datensensiblen Branchen (Finanzen, Gesundheit) sind Datenschutz und DSGVO-Konformität ein Hindernis.

Der Trend geht daher zu **Small Language Models (SLMs)**: Modelle mit unter 10 Milliarden Parametern, die durch architektonische Innovationen, Effizienz und hochwertige Daten eine überraschend hohe Leistung bei einem Bruchteil der Kosten bieten — und lokal auf eigener Hardware betrieben werden können.

## Ziel dieses Projekts

Dieses Wiki dient als Dokumentation zur Projektarbeit "Small Language Models". Es soll ein tiefes Verständnis für die neuen Entwicklungen im Bereich SLMs schaffen — von der theoretischen Taxonomie über Techniken zur Komplexitätsreduktion bis zur praktischen Implementierung mittels **Fine-Tuning (QLoRA)** auf einer Consumer-GPU.

## Navigation

Die Sidebar führt durch die Themen in logischer Reihenfolge:

1.  **[Grundlagen & Begriffe](1_Grundlagen)** — Foundation Models, LLMs, GPT → InstructGPT, und warum die Skalierung an ihre Grenzen stößt.
2.  **[Methodiken zur Komplexitätsreduktion](2_Komplexitaetsreduktion)** — Quantisierung (NF4), LoRA, QLoRA und Pruning im Detail.
3.  **[Small Language Models: Architektur](3_SLMs_Architecture)** — Die "Textbooks"-Hypothese, GQA, Sliding Window, Knowledge Distillation und der Vergleich LLM vs. SLM.
4.  **[Fallstudie: Google Gemma 3](4_Fallstudie_Gemma)** — Architektur, Benchmarks und Vergleich mit Phi-4-mini und Qwen3.
5.  **[Praxis: Fine-Tuning, Inferenz & Deployment](5_Praxis_Guide)** — QLoRA-Workflow, Edge Deployment, ökonomische Bewertung und Fazit.

## Code

Den dazugehörigen Code (Jupyter Notebooks für Google Colab) findet man im Ordner `notebooks/` dieses Repositories:

*   `SLM_Finetuning_Demo.ipynb` — Fine-Tuning von Gemma 3 mit QLoRA + Inferenz-Test

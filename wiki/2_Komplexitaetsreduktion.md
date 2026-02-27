# 2. Methodiken zur Komplexitätsreduktion

Die folgenden Techniken senken den Ressourcenbedarf von Sprachmodellen und machen sie damit für eine breitere Anwendung zugänglich. Diese Werkzeuge ermöglichen es, große Modelle effizienter zu machen oder kleine Modelle leistungsfähiger zu gestalten.

## 1. Quantisierung (Quantization)

Standardmäßig werden Deep-Learning-Modelle in 32-Bit Fließkommazahlen (FP32) oder 16-Bit (FP16/BF16) trainiert. Ein einziger Parameter benötigt in FP32 ganze 4 Byte Speicher. Ein 7B-Modell braucht so **ca. 28 GB VRAM** — allein für die Gewichte. [[1]](#quellen)

Quantisierung reduziert diese Präzision. Die Grundidee ist die Abbildung eines hochauflösenden Wertebereichs auf einen diskreten, niederauflösenden Bereich:

$$Q(x) = \text{round}(x / S + Z)$$

Wobei $x$ der ursprüngliche Gewichtswert (z.B. in FP32), $S$ der Skalierungsfaktor und $Z$ der Nullpunkt (Zero-Point) ist. [[1]](#quellen)

### Quantisierungsstufen

*   **FP32 (32-bit):** Volle Präzision, maximaler Speicherbedarf.
*   **FP16 / BF16 (16-bit):** Standard beim Training moderner Modelle. Halbierter Speicher.
*   **INT8 / FP8 (8-bit):** Gute Balance zwischen Genauigkeit und Speicher.
*   **INT4 / NF4 (4-bit):** Aggressivste Kompression, ermöglicht Betrieb auf Consumer-Hardware.

### NF4 (4-bit NormalFloat) und Double Quantization

Ein Problem der klassischen Integer-Quantisierung: Sie geht von einer **gleichmäßigen** Verteilung der Werte aus. Die Gewichte in neuronalen Netzen folgen jedoch typischerweise einer **Normalverteilung** (Gauß-Kurve) — die meisten Werte liegen nahe bei Null, wenige sind extrem groß oder klein.

**NF4** löst dieses Problem: Die Quantisierungsstufen werden so gewählt, dass jeder "Bin" die gleiche Wahrscheinlichkeitsmasse der Normalverteilung abdeckt. Das bedeutet eine höhere Auflösung nahe Null (wo die meisten Gewichte liegen) und eine geringere an den Rändern. NF4 ist damit **informationstheoretisch optimal** für normalverteilte Daten. [[2]](#quellen)

**Double Quantization** geht noch einen Schritt weiter: Da auch die Skalierungsfaktoren $S$ selbst Speicher benötigen, werden diese ebenfalls quantisiert (z.B. von FP32 auf FP8). Dies spart bei großen Modellen nochmals signifikant Speicher. [[3]](#quellen)

### Post-Training (PTQ) vs. Quantization-Aware Training (QAT)

*   **PTQ:** Nachträgliche Quantisierung eines fertig trainierten Modells. Schnell und einfach, aber leichter Genauigkeitsverlust möglich.
*   **QAT:** Die Quantisierung wird bereits während des Trainings simuliert. Bestmögliche Modellqualität trotz reduzierter Bit-Breite, aber rechenaufwendiger. [[4]](#quellen)

> **Ergebnis:** Ein 7B Modell in 4-Bit braucht nur noch **~4–5 GB VRAM** und läuft auf Consumer-GPUs.

## 2. LoRA (Low-Rank Adaptation)

Ein Modell komplett neuzutrainieren ("Full Fine-Tuning") ist extrem teuer, da für **jeden** Parameter Gradienten und Optimizer-States (z.B. Adam-Momentums) gespeichert werden müssen.

**LoRA** basiert auf der Hypothese, dass die Gewichtsänderungen $\Delta W$, die während der Anpassung an eine spezifische Aufgabe nötig sind, einen **niedrigen intrinsischen Rang** haben. [[5]](#quellen) Anstatt die volle Matrix zu verändern, werden parallel zwei kleine Matrizen eingefügt:

$$W' = W + \Delta W = W + B \cdot A$$

*   W ∈ ℝ<sup>d×k</sup>: Die **eingefrorene** Original-Matrix des Basismodells.
*   B ∈ ℝ<sup>d×r</sup> und A ∈ ℝ<sup>r×k</sup>: Die trainierbaren Adapter-Matrizen.
*   r ist der **Rang**, wobei r ≪ min(d, k).

### Rechenbeispiel

Wenn $d = 4096$ und der Rang $r = 8$ gewählt wird:
*   **Full Fine-Tuning:** $4096 \times 4096 \approx 16{,}7$ Millionen Parameter.
*   **LoRA:** $(4096 \times 8) + (8 \times 4096) \approx 65.000$ Parameter.
*   **Reduktion: Faktor 250.** [[6]](#quellen)

### Vorteile von LoRA

*   **Effizienz:** Fine-Tuning riesiger Modelle wird auf Consumer-Hardware möglich.
*   **Keine Inferenz-Latenz:** Nach dem Training können die Matrizen $B \cdot A$ vorberechnet und zur Basis-Matrix $W$ addiert werden. Es entstehen keine zusätzlichen Schichten, die die Ausführung verlangsamen. [[2]](#quellen)
*   **Modularität:** Verschiedene LoRA-Adapter für verschiedene Aufgaben (z.B. "Coding", "Deutsch", "Jura") können trainiert und zur Laufzeit ausgetauscht werden, ohne das große Basismodell neu zu laden.

### QLoRA = Quantisierung + LoRA

QLoRA kombiniert 4-Bit NF4-Quantisierung des Basismodells mit LoRA-Adaptern. Das ermöglicht das Fine-Tuning von Modellen mit dutzenden Milliarden Parametern auf einer **einzelnen Consumer-GPU** (z.B. RTX 3090/4090). [[3]](#quellen)

## 3. Pruning (Beschneidung)

Pruning entfernt "überflüssige" Teile des neuronalen Netzes. [[7]](#quellen)

### Unstructured Pruning
Einzelne Gewichte, die nahe Null sind, werden auf exakt Null gesetzt. Das Modell wird "sparse" (dünnbesetzt). Dies reduziert die Modellgröße theoretisch, bringt aber **kaum Speedup auf Standard-GPUs**, da diese für dichte Matrizen optimiert sind. [[4]](#quellen)

### Structured Pruning
Ganze Strukturen wie Neuronen, Kanäle oder **Attention Heads** werden entfernt. Dies führt zu kleineren, dichten Matrizen, die **sofort auf Standard-Hardware schneller laufen**.

Ein moderner Ansatz ist das **Depth Pruning (Layer Pruning)**: Ganze Schichten werden aus dem Modell entfernt. Nvidia-Forscher zeigten, dass durch Pruning und anschließende Wissensdestillation signifikant kleinere Modelle erzeugt werden können, die einen Großteil der Leistung behalten. [[7]](#quellen)

## Vergleichstabelle der Verfahren

| Verfahren | Funktionsweise | Primärer Nutzen | Nachteile |
| :--- | :--- | :--- | :--- |
| **PTQ** | Nachträgliche Reduktion der Bit-Präzision (z.B. FP16 → INT4). | Sofortige Speicherreduktion ohne Training. | Leichter Genauigkeitsverlust möglich. |
| **QAT** | Simulation der Quantisierung während des Trainings. | Bestmögliche Modellqualität trotz reduzierter Bit-Breite. | Rechenaufwendig im Training. |
| **LoRA** | Einfügen kleiner, trainierbarer Matrizen ($A$, $B$); Einfrieren des Basismodells. | Extrem effizientes Fine-Tuning; Modularität. | Etwas komplexere Implementierung. |
| **QLoRA** | Kombination aus 4-Bit Quantisierung (NF4) und LoRA. | Fine-Tuning riesiger Modelle auf Consumer-GPUs. | Leicht langsameres Training durch De-Quantisierung. |
| **Unstructured Pruning** | Setzen einzelner Gewichte auf Null (Sparse Matrix). | Theoretische Modellverkleinerung. | Kaum Speedup auf Standard-Hardware. |
| **Structured Pruning** | Entfernen ganzer Layer oder Attention Heads. | Echter Speedup und Speicherreduktion. | Kann Modellqualität stärker beeinträchtigen. |

---

## Quellen

1. Quantization — Hugging Face Optimum Documentation. https://huggingface.co/docs/optimum/concept_guides/quantization
2. LoRA vs. QLoRA: Efficient fine-tuning techniques for LLMs — Modal. https://modal.com/blog/lora-qlora
3. QLoRA: Efficient Finetuning of Quantized LLMs — MaximoFN. https://www.maximofn.com/en/qlora/
4. A Survey on Model Compression for Large Language Models — ACL Anthology. https://aclanthology.org/2024.tacl-1.85.pdf
5. LLM Optimization: LoRA and QLoRA — Towards Data Science. https://towardsdatascience.com/llm-optimization-lora-and-qlora/
6. Fundamentals of LoRA and low-rank fine-tuning — Nebius. https://nebius.com/blog/posts/fine-tuning/lora-low-rank-adaptation
7. Pruning and Distilling LLMs Using NVIDIA TensorRT Model Optimizer. https://developer.nvidia.com/blog/pruning-and-distilling-llms-using-nvidia-tensorrt-model-optimizer/

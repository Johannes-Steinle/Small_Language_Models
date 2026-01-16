# 2. Wie macht man Modelle kleiner?

Die Demokratisierung der KI basiert auf Techniken, die den Ressourcenhunger der Modelle senken.

## 1. Quantisierung (Quantization)
Standardmäßig rechnen KIs mit hoher Präzision (32-Bit Fließkommazahlen, FP32). Ein 7B Modell braucht so **28 GB VRAM**.
Quantisierung reduziert die Präzision, z.B. auf 4-Bit Integer.

*   **FP32 (32-bit):** Hohe Präzision, riesiger Speicher.
*   **INT8 / FP8:** Gute Balance.
*   **NF4 (4-bit NormalFloat):** Eine Spezialität von **QLoRA**. Da Gewichte in neuronalen Netzen "normalverteilt" sind (Glockenkurve), nutzt NF4 die verfügbaren Bits optimaler als reines Integer-Mapping.

> [!TIP]
> **Ergebnis:** Ein 7B Modell in 4-Bit braucht nur noch **~4-5 GB VRAM** und läuft auf Consumer-GPUs.

## 2. LoRA (Low-Rank Adaptation)
Ein Modell komplett neuzutrainieren ("Full Fine-Tuning") ist extrem teuer.
**LoRA** friert das Hauptmodell ein und trainiert nur zwei winzige Zusatz-Matrizen ($A$ und $B$) pro Layer.

$$W' = W + (B \cdot A)$$

*   $W$: Die eingefrorene riesige Matrix des Basismodells.
*   $A, B$: Die winzigen, trainierbaren Matrizen.

**Vorteile:**
*   **Kein Memory-Overhead:** Nur wenige MB statt GB an Parametern müssen optimiert werden.
*   **Modularität:** Man kann verschiedene Adapter (z.B. "Coding-Adapter", "German-Adapter") einfach austauschen.

## 3. Pruning
Hierbei werden "unnötige" Verbindungen im Gehirn der KI gekappt (auf Null gesetzt). Das Modell wird "dünner" (sparse). Dies ist technisch anspruchsvoller umzusetzen als Quantisierung.

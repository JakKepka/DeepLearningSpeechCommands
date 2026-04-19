# Dokumentacja techniczna projektu

## Speech Commands Classification with Transformers

**Autor:** Jakub Kępka, Damian Kąkol
**Data:** 2026-04-14

## Streszczenie

Niniejszy dokument stanowi techniczny plan realizacji projektu z przedmiotu Deep Learning dotyczącego klasyfikacji komend głosowych z użyciem architektur Transformer. Dokument jest przeznaczony jako specyfikacja wdrożeniowa dla agenta Copilot w Visual Studio Code oraz dla autora projektu. Zawiera architekturę rozwiązania, strukturę repozytorium, plan implementacji, harmonogram prac, plan eksperymentów, strategię testowania, kryteria akceptacji oraz zestaw zadań, które mogą być delegowane do agenta kodującego. Dokument uwzględnia również sposób prowadzenia badań nad klasami `silence` i `unknown`, które są jednym z głównych elementów zadania projektowego.

---

## 1. Cel dokumentu

Celem dokumentu jest przekształcenie ogólnego planu projektu w precyzyjną, techniczną specyfikację wykonawczą. Dokument ma umożliwić systematyczną realizację projektu przy pomocy VS Code Copilot Agent, bez konieczności improwizowania architektury, eksperymentów i testów w trakcie pracy.

Dokument pełni jednocześnie trzy role:

1. specyfikacji implementacyjnej dla repozytorium projektu,
2. planu eksperymentów badawczych,
3. checklisty wykonawczej do iteracyjnej realizacji przez agenta kodującego.

---

## 2. Zakres projektu

Projekt dotyczy klasyfikacji krótkich nagrań mowy w ustawieniu 12-klasowym. Klasy docelowe obejmują 10 komend:

`yes, no, up, down, left, right, on, off, stop, go`

oraz dwie klasy specjalne:

- `silence`
- `unknown`

Projekt ma odpowiedzieć na trzy pytania techniczne:

1. Czy lekki Transformer zaprojektowany pod keyword spotting daje przewagę nad prostym CNN?
2. Czy pretrained audio Transformer poprawia wyniki względem modelu trenowanego od zera?
3. Czy jawne modelowanie `silence` i `unknown`, w tym pipeline hierarchiczny, poprawia odporność systemu?

---

## 3. Założenia techniczne

### 3.1 Dane

Projekt wykorzystuje **Speech Commands Dataset** w konfiguracji 12-klasowej. Zakładamy użycie standardowego podziału train/validation/test dostępnego w praktycznych implementacjach tego zadania. Każda próbka audio ma długość około 1 sekundy i częstotliwość próbkowania 16 kHz.

### 3.2 Reprezentacja wejścia

Podstawową reprezentacją wejściową będzie **log-Mel spectrogram**. Taki wybór umożliwia:

- łatwe zbudowanie CNN baseline,
- bezpośrednie użycie lekkiego Transformera na sekwencjach tokenów ze spektrogramu,
- kompatybilność z modelami pretrained, w szczególności AST.

### 3.3 Framework i środowisko

Rekomendowany stack technologiczny:

- **Python 3.11**
- **PyTorch**
- **torchaudio**
- **scikit-learn**
- **matplotlib / seaborn** tylko do wizualizacji offline
- opcjonalnie **transformers** lub kod AST z repo referencyjnego

Kod powinien działać lokalnie na MacBook Pro M4 oraz być możliwy do uruchomienia na Google Colab lub Kaggle.

---

## 4. Architektura rozwiązania

Rozwiązanie należy zaprojektować modułowo. Minimalna architektura logiczna obejmuje następujące warstwy:

1. **Warstwa danych** — pobieranie, indeksowanie, podział i ładowanie datasetu.
2. **Warstwa feature extraction** — generowanie log-Mel spectrogramów i augmentacji.
3. **Warstwa modeli** — definicje architektur CNN, KWT-style Transformer oraz pretrained AST.
4. **Warstwa treningu** — wspólny trainer, logowanie metryk, checkpointing, early stopping.
5. **Warstwa ewaluacji** — accuracy, macro F1, recall klas krytycznych, confusion matrix.
6. **Warstwa eksperymentów** — konfiguracje YAML/JSON dla wielu uruchomień.
7. **Warstwa inferencji** — predykcja pojedynczych plików audio i pipeline hierarchiczny.

---

## 5. Struktura repozytorium

```text
project/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── data/
│   ├── model/
│   ├── train/
│   └── experiments/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   ├── sampler.py
│   │   └── labels.py
│   ├── models/
│   │   ├── cnn_baseline.py
│   │   ├── kwt.py
│   │   ├── ast_wrapper.py
│   │   └── hierarchical.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   ├── callbacks.py
│   │   └── seed.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── confusion.py
│   │   └── reports.py
│   ├── inference/
│   │   └── predict.py
│   └── utils/
│       ├── io.py
│       ├── config.py
│       └── logging.py
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── run_experiment.py
│   └── export_results.py
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_training.py
│   ├── test_metrics.py
│   └── test_inference.py
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   ├── figures/
│   └── tables/
└── docs/
    ├── technical_plan.tex
    └── experiment_log.md
```

---

## 6. Modele do implementacji

### 6.1 Model A: CNN baseline

Minimalna architektura:

- wejście: log-Mel spectrogram,
- 3 lub 4 bloki `Conv2D + BatchNorm + ReLU`,
- pooling pomiędzy blokami,
- global average pooling,
- klasyfikator liniowy do 12 klas.

Celem tego modelu jest szybki, stabilny baseline.

### 6.2 Model B: KWT-style Transformer

Założenia:

- tokenizacja spektrogramu do sekwencji,
- embedding pozycyjny,
- 4–6 warstw enkodera Transformer,
- 4 heads attention,
- dropout 0.1,
- klasyfikator końcowy dla 12 klas.

Należy przygotować dwa warianty:

- **small**: np. `d_model=128`, 4 warstwy,
- **medium**: np. `d_model=192`, 6 warstw.

### 6.3 Model C: Pretrained AST

Założenia:

- wykorzystanie pretrained Audio Spectrogram Transformer,
- fine-tuning klasyfikatora końcowego na 12 klas,
- opcjonalne zamrożenie części warstw w pierwszych epokach.

Jeżeli AST okaże się zbyt ciężki w praktyce, należy przygotować plan awaryjny w postaci lekkiego transfer learning baseline, ale w repo nadal trzeba zachować miejsce na wrapper AST.

---

## 7. Strategie dla `silence` i `unknown`

Należy zaimplementować trzy strategie.

### 7.1 Strategia 1: flat 12-class

Pojedynczy klasyfikator przewiduje jedną z 12 klas.

### 7.2 Strategia 2: flat 12-class z rebalancingiem

Ten sam klasyfikator, ale z modyfikacją procesu treningowego, np. przez:

- class weights,
- weighted sampler,
- kontrolowany oversampling klas trudnych.

### 7.3 Strategia 3: pipeline hierarchiczny

Należy zbudować dwuetapowy pipeline:

1. **Stage 1:** klasyfikacja do `silence`, `target-command`, `unknown-or-other`.
2. **Stage 2:** klasyfikacja tylko wśród 10 komend docelowych.

Logika inferencji:

- jeśli Stage 1 przewidzi `silence`, wynik końcowy to `silence`,
- jeśli Stage 1 przewidzi `unknown-or-other`, wynik końcowy to `unknown`,
- jeśli Stage 1 przewidzi `target-command`, uruchamiany jest Stage 2.

---

## 8. Plan implementacji dla VS Code Copilot Agent

Realizacja powinna przebiegać iteracyjnie. Każdy etap ma dostarczać działający fragment systemu.

### 8.1 Etap 1: bootstrap repozytorium

**Zadania:**

- utworzyć strukturę katalogów,
- przygotować `requirements.txt` lub `pyproject.toml`,
- dodać loader konfiguracji,
- przygotować podstawowy `README.md`.

**Kryterium akceptacji:** repozytorium uruchamia się lokalnie i posiada bazowy entrypoint do treningu.

### 8.2 Etap 2: data pipeline

**Zadania:**

- zaimplementować dataset loader,
- dodać mapowanie etykiet do 12 klas,
- przygotować preprocessing audio i log-Mel spectrogramów,
- zaimplementować augmentacje,
- dodać walidację kształtów tensorów.

**Kryterium akceptacji:** pojedynczy batch przechodzi przez loader i transformacje bez błędów.

### 8.3 Etap 3: CNN baseline

**Zadania:**

- zaimplementować CNN baseline,
- dodać train loop,
- logować loss i accuracy,
- zapisać pierwszy checkpoint.

**Kryterium akceptacji:** model kończy minimum jedną epokę treningu i generuje predykcje na walidacji.

### 8.4 Etap 4: ewaluacja i raportowanie wyników

**Zadania:**

- obliczać accuracy, macro F1 i per-class recall,
- generować confusion matrix,
- eksportować wyniki do CSV/JSON,
- zapisywać wykresy do `outputs/figures`.

**Kryterium akceptacji:** po treningu powstaje kompletny pakiet wyników dla pojedynczego runu.

### 8.5 Etap 5: KWT-style Transformer

**Zadania:**

- zaimplementować tokenizer spektrogramu,
- dodać model `kwt.py`,
- przygotować dwa rozmiary modelu,
- zintegrować z istniejącym trainerem.

**Kryterium akceptacji:** model transformerowy trenuje się tym samym skryptem co CNN.

### 8.6 Etap 6: obsługa klas trudnych

**Zadania:**

- dodać weighted sampler lub class weights,
- zaimplementować pipeline hierarchiczny,
- przygotować inferencję dwuetapową,
- porównać wyniki z klasyfikatorem płaskim.

**Kryterium akceptacji:** można uruchomić eksperyment `flat` oraz `hierarchical` z jednego interfejsu CLI.

### 8.7 Etap 7: pretrained AST

**Zadania:**

- przygotować wrapper dla AST,
- dodać fine-tuning head na 12 klas,
- ujednolicić preprocessing z resztą pipeline'u,
- zintegrować logowanie metryk.

**Kryterium akceptacji:** AST uruchamia trening i walidację, nawet jeśli wymaga osobnej konfiguracji.

### 8.8 Etap 8: finalizacja eksperymentów

**Zadania:**

- przygotować gotowe konfiguracje eksperymentów,
- uruchomić powtórzenia z różnymi seedami,
- wyeksportować tabele końcowe,
- przygotować materiał do raportu.

**Kryterium akceptacji:** w katalogu `outputs/tables` znajdują się gotowe zestawienia wyników końcowych.

---

## 9. Jak rozłożyć pracę w czasie

Ta sekcja jest przeznaczona do realizacji projektu i nie powinna trafić do finalnego raportu badawczego.

### 9.1  fundamenty

- utworzenie repozytorium,
- konfiguracja środowiska,
- implementacja data pipeline,
- przygotowanie log-Mel spectrogramów,
- test przejścia jednego batcha przez loader.

**Rezultat tygodnia:** działający pipeline danych.

### 9.2 baseline i ewaluacja

- implementacja CNN baseline,
- trening próbny,
- metryki, confusion matrix i eksport wyników,
- sanity-check na małej konfiguracji.

**Rezultat tygodnia:** pierwszy działający model i pełna ścieżka trening–ewaluacja.

### 9.3  Transformer

- implementacja KWT-style Transformer,
- uruchomienie wariantu small,
- uruchomienie wariantu medium,
- porównanie z baseline.

**Rezultat tygodnia:** główny model projektowy działa i daje porównywalne wyniki.

### 9.4 klasy trudne

- rebalancing dla flat 12-class,
- implementacja pipeline'u hierarchicznego,
- analiza wpływu na `silence` i `unknown`,
- przygotowanie confusion matrices.

**Rezultat tygodnia:** gotowy eksperyment wyróżniający projekt.

### 9.5 pretrained model i domknięcie eksperymentów

- integracja AST,
- fine-tuning lub eksperyment awaryjny,
- uruchomienie końcowych seedów,
- eksport finalnych tabel i figur.

**Rezultat tygodnia:** domknięty zestaw wyników do raportu.

---

## 10. Plan eksperymentów

Eksperymenty należy realizować w kontrolowanej kolejności.

### 10.1 Grupa A: porównanie architektur

| ID | Model           | Cel                                   |
| -- | --------------- | ------------------------------------- |
| A1 | CNN baseline    | Punkt odniesienia dla prostego modelu |
| A2 | KWT-small       | Główny lekki Transformer            |
| A3 | KWT-medium      | Wpływ większej pojemności modelu   |
| A4 | AST fine-tuning | Ocena transfer learningu              |

### 10.2 Grupa B: wpływ parametrów

| ID | Zmiana                  | Cel                                  |
| -- | ----------------------- | ------------------------------------ |
| B1 | KWT-small vs KWT-medium | Wpływ rozmiaru modelu               |
| B2 | Augmentation off vs on  | Wpływ augmentacji na generalizację |
| B3 | Dropout 0.1 vs 0.2      | Wpływ regularizacji                 |

### 10.3 Grupa C: `silence` / `unknown`

| ID | Strategia             | Cel                                      |
| -- | --------------------- | ---------------------------------------- |
| C1 | Flat 12-class         | Baseline dla klas trudnych               |
| C2 | Flat + rebalancing    | Sprawdzenie poprawy recall trudnych klas |
| C3 | Hierarchical pipeline | Główna hipoteza projektowa             |

### 10.4 Powtórzenia

Zalecany schemat:

- szybkie eksperymenty przesiewowe: 1 seed,
- eksperymenty finalne: 3 seedy,
- końcowe raportowanie: średnia i odchylenie standardowe dla najważniejszych wariantów.

---

## 11. Metryki i artefakty wynikowe

Każdy eksperyment powinien generować następujące wyniki:

- accuracy,
- macro F1,
- recall dla `silence`,
- recall dla `unknown`,
- confusion matrix,
- liczba parametrów modelu,
- czas treningu na epokę.

Wyniki powinny być eksportowane do:

- `CSV` — tabele metryk,
- `JSON` — szczegóły runu,
- `PNG/PDF` — confusion matrices i wykresy.

---

## 12. Plan testów

Testy należy podzielić na trzy poziomy.

### 12.1 Testy jednostkowe

Obowiązkowe testy:

- poprawność mapowania etykiet do 12 klas,
- poprawność kształtu log-Mel spectrogramu,
- poprawność działania augmentacji,
- poprawność forward pass dla każdego modelu,
- poprawność obliczania metryk.

### 12.2 Testy integracyjne

Obowiązkowe testy:

- przejście batcha danych przez loader, model i funkcję straty,
- wykonanie jednej epoki treningu bez wyjątku,
- wygenerowanie confusion matrix po ewaluacji,
- działanie inferencji dla pojedynczego pliku WAV,
- działanie pipeline'u hierarchicznego od wejścia do decyzji końcowej.

### 12.3 Testy eksperymentalne

Testy badawcze obejmują:

- porównanie CNN i KWT-small na tej samej konfiguracji danych,
- sprawdzenie wpływu augmentacji,
- sprawdzenie, czy rebalancing poprawia recall klas trudnych,
- sprawdzenie, czy pipeline hierarchiczny redukuje liczbę pomyłek `unknown` ↔ target command.

---

## 13. Kryteria akceptacji projektu

Projekt uznaje się za technicznie gotowy, jeżeli spełnione są następujące warunki:

1. repozytorium pozwala uruchomić trening co najmniej dwóch architektur,
2. istnieje kompletna ścieżka danych od surowego audio do metryk,
3. generowane są confusion matrices i tabele wyników,
4. zaimplementowano co najmniej jedną strategię specjalną dla `silence` i `unknown`,
5. można uruchomić eksperymenty z wielu konfiguracji bez ręcznego przepisywania kodu,
6. istnieją testy automatyczne dla kluczowych komponentów,
7. wyniki można łatwo przenieść do raportu końcowego.

---

## 14. Zadania dla Copilot Agent — gotowe polecenia

Poniższe polecenia można wydawać agentowi krok po kroku. Zaleca się używać języka angielskiego, bo zwykle poprawia to jakość generowanego kodu.

### Prompt 1: bootstrap

```text
Create the initial Python project structure for a speech commands
classification project in PyTorch. Add src/, scripts/, tests/, configs/
and outputs/ directories, plus requirements.txt and a basic README.
```

### Prompt 2: dataset

```text
Implement a PyTorch Dataset for Speech Commands in a 12-class setup:
yes, no, up, down, left, right, on, off, stop, go, silence, unknown.
Add label mapping, waveform loading, padding/trimming to 1 second,
and log-Mel spectrogram extraction using torchaudio.
```

### Prompt 3: augmentations

```text
Add data augmentation for speech commands: time shift, background noise
injection, and SpecAugment for spectrograms. Make them configurable.
```

### Prompt 4: CNN baseline

```text
Implement a lightweight CNN baseline for log-Mel spectrogram input.
Include forward pass, dropout, and a classifier head for 12 classes.
```

### Prompt 5: training loop

```text
Implement a reusable training pipeline in PyTorch with train/validation
loops, checkpointing, early stopping, metric logging, and fixed seed setup.
```

### Prompt 6: metrics

```text
Implement evaluation utilities for accuracy, macro F1, per-class recall,
and confusion matrix export for a 12-class speech commands task.
```

### Prompt 7: KWT

```text
Implement a lightweight Keyword Transformer style model for speech command
classification from log-Mel spectrogram tokens. Create small and medium
configurations and integrate them with the shared training pipeline.
```

### Prompt 8: hierarchical pipeline

```text
Implement a hierarchical inference pipeline: stage 1 predicts silence,
target-command, or unknown-or-other; stage 2 predicts one of the 10 target
commands. Add a wrapper that combines both predictions into the final label.
```

### Prompt 9: AST wrapper

```text
Create an AST fine-tuning wrapper for 12-class speech command classification.
Add a configurable classification head and support freezing selected layers.
```

### Prompt 10: tests

```text
Write unit and integration tests for the data pipeline, model forward passes,
training step, metrics, and hierarchical inference pipeline.
```

---

## 15. Ryzyka projektowe i sposób reakcji

- **Ryzyko: AST jest zbyt ciężki.**Reakcja: najpierw domknąć CNN i KWT, a AST uruchomić jako eksperyment dodatkowy lub częściowo zamrożony.
- **Ryzyko: błędne mapowanie `unknown`.**Reakcja: dodać jawne testy etykiet i statystyk rozkładu klas.
- **Ryzyko: overfitting małych modeli.**Reakcja: włączyć augmentację, dropout i early stopping.
- **Ryzyko: zbyt duża złożoność eksperymentów.**
  Reakcja: utrzymywać sztywną macierz eksperymentów z tej dokumentacji i nie rozszerzać zakresu bez potrzeby.

---

## 16. Podsumowanie

Dokument definiuje kompletny plan techniczny realizacji projektu: od repozytorium i pipeline'u danych, przez implementację modeli, po eksperymenty, testy i organizację pracy w czasie. Najważniejszą częścią badawczą projektu będzie porównanie prostego CNN, lekkiego Transformera oraz modelu pretrained, a także analiza strategii dla klas `silence` i `unknown`. Dokument może być używany bezpośrednio jako instrukcja operacyjna dla VS Code Copilot Agent oraz jako baza do późniejszego raportu końcowego.

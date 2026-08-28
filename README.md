# CIFAR-10: Iteracyjna ewolucja architektury i optymalizacja (MLP → CNN)

Mój pierwszy projekt z zakresu uczenia maszynowego i głębokiego uczenia. Głównym celem nie było stworzenie jednego, gotowego modelu "black-box", lecz **pełne prześledzenie procesu inżynierii modelu od zera** — od prostej sieci gęstej (MLP), przez klasyczne bloki splotowe (CNN), aż po zaawansowaną diagnostykę i strojenie hiperparametrów.

Repozytorium przedstawia ewolucję kolejnych architektur, gdzie każdy kolejny eksperyment stanowi bezpośrednią odpowiedź na ograniczenia zaobserwowane w poprzednim kroku.

---

## Przebieg procesu badawczego

Repozytorium zawiera ciąg notebooków/skryptów dokumentujących kolejne fazy projektu:

### Faza 0: Przygotowanie danych
* Pobranie zbioru danych i analiza otrzymanych danych.
* Wstępne feature engineering: przygotowanie zbioru walidacyjnego, wymieszanie i batchowanie danych.


### Faza 1: Baseline MLP (Multi-Layer Perceptron)
* Implementacja podstawowej sieci gęstej na spłaszczonych obrazach ($32 \times 32 \times 3$).
* Zrozumienie ograniczeń braku niezmienniczości przestrzennej i problemu eksplozji parametrów w warstwach Dense.
*  **LR Finder & 1cycle Policy:** Wyznaczanie optymalnego szczytowego współczynnika uczenia ($\text{LR}_{\max}$) i harmonogramu tempa uczenia w oparciu o dynamikę zbieżności.
*  Analiza górnej granicy klasycznej architektury MLP przy analizie obrazów, gdy poszerzy się pipeline o augmentacje danych i odpowiedni harmonogram uczenia

### Faza 2: Wejście w CNN i architektura VGG-style
* Zbudowanie pierwszych bloków splotowych (filtry $3 \times 3$, `MaxPool2D`).
* Zastąpienie ciężkich warstw FC mechanizmem **Global Average Pooling (GAP)** w celu redukcji przeuczenia.
* Przejście z prostej normalizacji Min-Max na standaryzację **Z-score** dopasowaną do statystyk kanałów CIFAR-10.

### Faza 3: Diagnostyka i skalowanie przestrzenne
* **Analiza Receptive Field (RF & ERF):** Przeliczenie teoretycznego i efektywnego pola widzenia dla kolejnych warstw konwolucyjnych w celu dopasowania percepcji neuronów do pełnego kadru $32 \times 32$ - poszerzenie modelu o dodatkowy blok konwolucyjny.
* **Struktura bloków:** Przejście do 4 bloków splotowych (w układzie 2-2-3-2) ze sprowadzeniem mapy cech do rozmiaru $2 \times 2$ przed GAP w celu uniknięcia nadmiernej utraty relacji przestrzennych.
* **Badanie luki generalizacji:** Zrozumienie, dlaczego wysokie train accuracy (~99.6%) nie gwarantuje wysokiego wyniku walidacyjnego bez odpowiedniej hierarchii cech.

### Faza 4: Optymalizacja i reżim treningowy
* **SGD + Nesterov vs Adam:** Porównanie zachowania optymalizatorów w zadaniach wizyjnych (wypychanie wag w stronę szerokich, płaskich minimów — *flat minima*).
* **Strojenie wielkości paczki (Batch Size):** Balansowanie między szumem stochastycznym a stabilnością statystyk `BatchNormalization`.

---

## Kluczowe wnioski z eksperymentów

1. **Efektywne Pole Widzenia (ERF):** Znajomość teoretycznego $RF$ nie wystarcza — neuron w praktyce skupia się na strefie Gaussa ($40\text{--}50\%$ wartości $RF$). Odpowiednia głębokość sieci jest kluczem do objęcia całego obiektu.
2. **Pułapka parametrów GAP:** Choć samo GAP znacząco redukuje liczbę parametrów, uśrednianie zbyt dużej siatki przestrzennej (np. $4 \times 4$) bez uprzedniej agregacji splotowej rozmywa relacje przestrzenne między cechami.
3. **SGD Momentum > Adam (dla CNN od zera):** Właściwie dostrojony SGD z Nesterovem w połączeniu z 1cycle pozwala przełamać plateau dokładności walidacyjnej, na którym zatrzymują się metody czysto adaptacyjne.

---

## Technologie
* Python 3
* TensorFlow / Keras
* NumPy, Matplotlib

## Podsumowanie wyników
| Architektura / Faza | Optymalizator | Test Accuracy | Kluczowy wniosek / Zmiana |
| :--- | :--- | :---: | :--- |
| **Faza 0: Data Engineering** | - | - | Kluczem do wytrenowania i diagnostyki modelu jest poznanie danych oraz odpowiednie ich przygotowanie  |
| **Faza 1: MLP Baseline** | Adam + 1cycle | ~67% | Brak cech przestrzennych, eksplozja parametrów |
| **Faza 2.1: CNN Baseline (FC Layers)** | Adam | ~89% | Dane na wyjściu bloków konwolucyjnych trafiają do gęstych warstw, które na ich podstawie liczą predykcje |
| **Faza 2.2: CNN Baseline (GAP 4x4)** | Adam | ~90% | Wprowadzenie splotów, redukcja przeuczenia dzięki GAP |
| **Faza 3: CNN 2-2-3-2 (GAP 2x2)** | Adam | ~90.5% | Skalowanie ERF (~33 px), optymalizator osiąga plateau |
| **Faza 4: CNN 2-2-3-2** | SGD + Nesterov (1cycle) | **~92%** | Przełamanie sufitu dzięki regularizacji i *flat minima* |

[Sprawozdanie](https://docs.google.com/document/d/1-7V3ma5Dan7shwS-XfoPM3hgOG_75nKDQnRvET-zWvo/edit?usp=sharing)
 
# 1. Tytuł i numer projektu
**Gesture Recognizer Game** (Projekt nr 6)
 
# 2. Zespół oraz termin zajęć
**Zespół 6** | Termin: **poniedziałek, 08:00**
 
# 3. Skład zespołu
* Maciej Banaś
* Konrad Ciężadło
* Mikhail Shupliakou
# 4. Podział obowiązków w zespole
* **Maciej Banaś:** Autorski klasyfikator gestów: zbieranie zdjęć (`auto_capture_gestures.py`), etykietowanie, augmentacja i trening modelu w aplikacji Gradio (`gesture_label_simple_gui.py`), normalizacja punktów dłoni (`helper_functions.py`), benchmarki klasyfikatorów i porównanie z MediaPipe (`results/`), skrypt testowy klasyfikatora (`test_gesture_classifier.py`) oraz podłączenie modelu do `scientific_tracker.py`.
* **Konrad Ciężadło:** Definicja i realizacja gry w silniku Unity (rozgrywka, spawnery, pociski, model fizyczny wiatru i oporu, interfejs) oraz skrypt trybu naukowego (`scientific_tracker.py`).
* **Mikhail Shupliakou:** Moduł Python do rozpoznawania gestów MediaPipe i komunikacji UDP z Unity (`gesture_tracker.py`), struktura repozytorium, integracja Python–Unity, dokumentacja (README, sprawozdanie).
# 5. Cel projektu
Celem projektu jest przygotowanie gry, w której gracz za pomocą gestów dłoni niszczy przeszkody. Wykorzystano gotowy model rozpoznawania dłoni MediaPipe Task. Gesty są interpretowane na podstawie autorskiego klasyfikatora, a wyniki porównywane z rozwiązaniem regułowym (MediaPipe Gesture Recognizer).
 
# 6. Struktura katalogów
```text
gesture-recognizer-game/
├── README.md
├── pythonPart/                          # Moduł Python: rozpoznawanie gestów i klasyfikator
│   ├── gesture_tracker.py               # Prosty tracker do gry (MediaPipe, tryb --headless)
│   ├── scientific_tracker.py            # Tracker do gry (--task / --trained) oraz tryb naukowy (A/B, CSV)
│   ├── helper_functions.py              # Normalizacja punktów dłoni, mapowanie nazw gestów
│   ├── auto_capture_gestures.py         # Robienie zdjęć gestów z kamery do zbioru danych
│   ├── gesture_label_simple_gui.py      # Gradio: etykietowanie, augmentacja, trening, benchmark, zapis modelu
│   ├── test_gesture_classifier.py       # Test autorskiego klasyfikatora na żywo z kamery
│   ├── dataset_captured/                # Zbiór danych: 156 zdjęć (pointing_up, thumb_up, thumb_down, victory)
│   ├── models/
│   │   └── custom_gesture_model.joblib  # Wytrenowany autorski klasyfikator (Random Forest)
│   └── results/                         # Wyniki benchmarków (CSV, raporty, macierze pomyłek, wykresy)
│       └── gesture_benchmark_<data>/    # Jedno uruchomienie treningu: porównanie modeli, augmentacji i MediaPipe
└── unityGame/unityGame/                 # Projekt Unity 6.6 (Assets, Packages, ProjectSettings)
    └── Assets/
        ├── Scenes/                      # MainMenu, Level (gra), Stats, SampleScene
        └── Scripts/                     # HandOperation, BulletSpawner, BulletController, Wind, ObstacleSpawner, ObstacleTile, GameManager, ...
```
 
Plik `gesture_recognizer.task` (model MediaPipe) nie jest przechowywany w repozytorium. Skrypty pobierają go automatycznie przy pierwszym uruchomieniu (wymagane połączenie z Internetem).
 
# 7. Wymagane środowisko i biblioteki
* Kamera internetowa
* Python 3.9 – 3.12
* Unity 6.6 (edytor `6000.6.7f1`) zainstalowany przez Unity Hub
* System Windows, macOS lub Linux (projekt testowany na Windows)
* Biblioteki Python (pełna lista w `wymagania.txt`):
  * `mediapipe`, `opencv-python`, `numpy` – wykrywanie dłoni i gestów (moduł gry),
  * `scikit-learn`, `joblib` – autorski klasyfikator (tryb `--trained`),
  * `gradio`, `matplotlib` – tylko do zbierania danych i trenowania własnego modelu.
# 8. Instrukcja uruchomienia
**1. Moduł Python** (uruchom jako pierwszy):
```bash
cd pythonPart
pip install mediapipe opencv-python numpy scikit-learn joblib
python scientific_tracker.py --task --port 5055
```
* `--task` – gra z gotowym modelem MediaPipe Gesture Recognizer,
* `--trained` – gra z autorskim klasyfikatorem (`models/custom_gesture_model.joblib`),
* `--gui` – podgląd z kamery, `--camera 1` – inny indeks kamery.
Do uruchomienia przykładowego scenariusza **nie jest potrzebne** ponowne trenowanie modelu, ponieważ wytrenowany klasyfikator znajduje się w repozytorium.
 
**2. Gra w Unity:**
1. W Unity Hub wybierz **Add → Add project from disk** i wskaż folder `unityGame/unityGame`.
2. Otwórz scenę `Assets/Scenes/MainMenu`.
3. (Weryfikacja) W scenie `Level` zaznacz obiekt `HandManager` i sprawdź, czy **Port** w komponencie **Hand Operation** wynosi **5055**.
4. Wróć do `MainMenu`, naciśnij **Play** w Unity i kliknij „Play” w menu gry.
**Sterowanie:** `Pointing_Up` – niebieski pocisk, `Thumb_Down` – czerwony, `Thumb_Up` – zielony, `Victory` – żółty. Aby zniszczyć płytkę, typ pocisku musi odpowiadać emotikonie na płytce (pewność gestu > 60%). Przepuszczenie płytki odbiera 1 z 3 punktów życia.
 
**3. (Opcjonalnie) Trenowanie własnego klasyfikatora:**
```bash
cd pythonPart
pip install gradio matplotlib scikit-learn joblib
python auto_capture_gestures.py        # robienie zdjęć gestów z kamery (dodatkowe dane)
python gesture_label_simple_gui.py     # Gradio: wczytaj zdjęcia -> „Dodaj wiele (auto nazwy)” -> „Trenuj, testuj i zapisz”
```
Wyniki treningu trafiają do `pythonPart/results/gesture_benchmark_<data>/`, a model do `pythonPart/models/custom_gesture_model.joblib`.
 
**Rozwiązywanie problemów:** brak szkieletu dłoni – sprawdź zgodność portu (`--port 5055` i ustawienie w Unity); błąd „Only one usage of each socket address” – port zajęty, zamknij poprzednią instancję Pythona; puste kwadraty zamiast emotikon – podepnij czcionkę emoji (np. `SEGUIEMJ SDF`) w polu `Emoji Font Asset` spawnerów; kamera się nie otwiera – zamknij inne programy używające kamery lub użyj `--camera 1`.
 
# 9. Wykorzystany silnik gry
Unity 6.6
 
# 10. Zastosowane elementy fizyki
Pocisk w grze posiada zmienne przyspieszenie w osiach X i Y (prostopadłych do długości tunelu), wyliczane z II zasady dynamiki Newtona jako wypadkowa siły wiatru i oporu powietrza. Użyta została również pół-niejawna metoda Eulera do iteracyjnego aktualizowania wektora prędkości i nowej pozycji pocisku w każdej klatce symulacji.
 
# 11. Odnośnik do repozytorium
https://github.com/mshupliakou/gesture-recognizer-game (wersja oceniana: commit `d1df805`, gałąź `master`)
 
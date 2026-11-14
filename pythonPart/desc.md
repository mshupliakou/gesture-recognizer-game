# Gra sterowana gestami dłoni (kamera + MediaPipe + Unity)

Projekt to zręcznościowa gra z perspektywy pierwszej osoby (w stylu Beat Saber), w której gracz niszczy nadlatujące z tunelu przeszkody, wykonując odpowiednie gesty dłonią przed kamerą. Projekt składa się z dwóch niezależnych aplikacji, które komunikują się ze sobą lokalnie przez UDP:

1. **Moduł Python** (`pythonPart`) – odczytuje obraz z kamerki, wykrywa dłoń i rozpoznaje gesty za pomocą modelu MediaPipe (`GestureRecognizer`) lub autorskiego klasyfikatora, a następnie wysyła wyniki jako pakiet JSON przez UDP.
2. **Gra w Unity** (`unityGame/unityGame`) – odbiera dane z portu UDP, renderuje w przestrzeni 3D neonowy szkielet dłoni gracza i na podstawie rozpoznanych gestów spawnuje pociski, które niszczą nadlatujące płytki z emotikonami.

```text
Kamera → Python (OpenCV + MediaPipe) → JSON przez UDP (127.0.0.1:5055) → Unity
```

## Struktura repozytorium

```text
pythonPart/
  ├── gesture_tracker.py       # Główny skrypt do zwykłej gry (z obsługą trybu --headless)
  ├── scientific_tracker.py    # Skrypt do trybu naukowego (testy A/B klasyfikatorów, zapis do CSV)
  ├── benchmarks/              # Folder na wygenerowane pliki CSV z wynikami pomiarów
unityGame/unityGame/           # Projekt Unity 6 (Assets, Packages, ProjectSettings)
```

## Wymagania

- Kamera internetowa
- Python 3.9 – 3.12
- Unity 6.6 zainstalowane przez Unity Hub
- System Windows, macOS lub Linux (projekt testowany na Windows)

## Uruchomienie (Zwykła Gra)

### 1. Moduł Python

```bash
cd pythonPart
pip install mediapipe opencv-python numpy
python scientific_tracker.py --port 5055
```

Wcześniej był plik `gesture_tracker.py` z flagą `--headless`.  `scientific_tracker.py` jest ładnie opisany w pliku.

### 2. Gra w Unity

1. Otwórz **Unity Hub** → **Add** → **Add project from disk** i wskaż folder `unityGame/unityGame`.
2. Otwórz projekt w edytorze Unity.
3. W oknie Project otwórz scenę główną: `Assets/Scenes/MainMenu`.
4. (Weryfikacja) Otwórz scenę `Level`, zaznacz obiekt `HandManager` i upewnij się, że wartość **Port** w komponencie **Hand Operation** to **5055**.
5. Wróć do sceny `MainMenu`.
6. **Najpierw uruchom skrypt Python**, a dopiero potem naciśnij **Play** w Unity i kliknij "Play" w menu.

### 3. Sterowanie i Mechanika

Gra korzysta z mapowania znormalizowanych współrzędnych dłoni na fizyczną płaszczyznę wewnątrz tunelu (przed kamerą w Unity).

| Rozpoznawany Gest | Emotikona | Działanie w grze |
|---|---|---|
| `Pointing_Up` | ☝️ | Spawnuje niebieski pocisk |
| `Thumb_Down` | 👎 | Spawnuje czerwony pocisk |
| `Thumb_Up` | 👍 | Spawnuje zielony pocisk |
| `Victory` | ✌️ | Spawnuje żółty pocisk |

**Zasady gry:**
- Z głębi tunelu nadlatują przeszkody (płytki) z wymalowaną emotikoną.
- Gracz musi ustawić dłoń na odpowiednim torze i wykonać gest odpowiadający emotikonie na płytce.
- Wykonanie poprawnego gestu (z pewnością modelu > 60%) spawnuje pocisk, który leci w głąb tunelu. 
- Aby zniszczyć przeszkodę, typ pocisku (gest) musi zgadzać się z typem przeszkody.
- Zmiany kierunku i siły wiatru (co 4 sekundy) znoszą pociski, wymuszając korektę pozycji dłoni.
- Przepuszczenie płytki skutkuje utratą 1 z 3 punktów HP.

## Tryb Naukowy (Scientific Benchmark)

Gra zawiera specjalny tryb do testów A/B dwóch klasyfikatorów uczenia maszynowego w czasie rzeczywistym. W tym trybie wiatr jest wyłączony, a prędkość płytek jest stała.

**Jak uruchomić:**
1. W terminalu uruchom moduł naukowy:
   `python scientific_tracker.py --port 5055` (możesz dodać `--gui`, aby widzieć podgląd z kamery).
2. Skrypt automatycznie co 5 sekund losuje gest i wysyła do Unity komendę stworzenia płytki.
3. W Unity otwórz i uruchom scenę `ScientificLevel`.
4. Skrypt Python na bieżąco porównuje czas reakcji i pewność modelu bazowego (MediaPipe) z modelem autorskim. Wyniki każdej 5-sekundowej próby są automatycznie zapisywane do pliku CSV w folderze `benchmarks/`.

## Format danych JSON (UDP)

Format został zoptymalizowany pod kątem parsera wbudowanego w Unity (`JsonUtility`), korzystając z płaskich obiektów zamiast zagnieżdżonych tablic.

```json
{
  "t": 12.453,
  "trial_id": 0,
  "spawn_command": "",
  "hands": [
    {
      "hand": "Right",
      "gesture": "Victory",
      "raw_gesture": "Victory",
      "score": 0.92,
      "palm": { "x": 0.52, "y": 0.41, "z": -0.02 },
      "pinch": 0.83,
      "landmarks": [
        {"x": 0.501, "y": 0.602, "z": 0.0},
        {"x": 0.523, "y": 0.580, "z": -0.015}
      ]
    }
  ]
}
```

## Rozwiązywanie problemów

- **Szkielet dłoni w Unity się nie pojawia** – sprawdź, czy skrypt Python został uruchomiony na właściwym porcie (`--port 5055`), czy dłoń jest dobrze oświetlona oraz czy port w ustawieniach skryptu `HandOperation.cs` w Unity również wynosi 5055.
- **`Only one usage of each socket address...`** – port 5055 jest zajęty przez inną aplikację lub poprzednią, niezamkniętą instancję Pythona. Upewnij się, że wyłączyłeś skrypt w terminalu (`Ctrl+C`) przed ponownym jego uruchomieniem.
- **Puste kwadraty zamiast emotikon na płytkach w Unity** – domyślna czcionka TextMeshPro nie posiada znaków Unicode dla emoji. Upewnij się, że w skryptach Spawnerów w polu `Emoji Font Asset` podpięty jest wygenerowany plik SDF oparty na czcionce systemowej (np. `SEGUIEMJ SDF`).
- **Kamera w Pythonie się nie otwiera** – zamknij inne programy używające kamery, takie jak MS Teams czy Discord, lub spróbuj uruchomić skrypt z flagą `--camera 1`.
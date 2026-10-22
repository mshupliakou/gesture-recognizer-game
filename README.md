# Gra sterowana gestami dłoni (kamera + MediaPipe + Unity)

Projekt składa się z dwóch niezależnych aplikacji, które komunikują się ze sobą lokalnie przez UDP:

1. **Moduł Python** (`pythonPart`) – odczytuje obraz z kamerki, wykrywa dłoń i rozpoznaje gest za pomocą gotowego modelu MediaPipe (`GestureRecognizer`), a następnie wysyła wynik jako JSON przez UDP.
2. **Gra w Unity** (`unityGame/unityGame`) – odbiera dane z portu UDP, przesuwa wirtualną dłoń (czerwona kula) i interpretuje gesty: zaciśnięta pięść chwyta piłkę, otwarta dłoń ją puszcza (rzut).

```
Kamera → Python (OpenCV + MediaPipe) → JSON przez UDP (127.0.0.1:5050) → Unity
```

## Struktura repozytorium

```
pythonPart/             skrypt rozpoznawania gestów (gesture_tracker.py)
unityGame/unityGame/    projekt Unity (Assets, Packages, ProjectSettings)
```

## Wymagania

- kamera internetowa,
- Python 3.9 – 3.12,
- Unity 6 (6000.x) zainstalowane przez Unity Hub,
- system Windows, macOS lub Linux (projekt testowany na Windows).

## Uruchomienie

### 1. Moduł Python

```bash
cd pythonPart
pip install mediapipe opencv-python numpy
python gesture_tracker.py --udp
```

Przy pierwszym uruchomieniu skrypt sam pobierze model `gesture_recognizer.task` (ok. 8 MB). Po chwili powinno pojawić się okno z obrazem z kamery, na którym widać świecący szkielet dłoni, nazwę rozpoznanego gestu i pasek pewności.

Flaga `--udp` włącza wysyłanie danych do gry. Bez niej skrypt działa tylko jako podgląd i zapisuje gesty do pliku.

Dodatkowe opcje:

| Opcja | Opis |
|---|---|
| `--udp` | wysyłaj dane przez UDP do gry |
| `--port 5055` | inny port (domyślnie 5050) |
| `--host 127.0.0.1` | adres odbiorcy |
| `--camera 1` | numer kamery, jeśli domyślna nie działa |
| `--frames` | zapisuj do pliku każdą klatkę z 21 punktami dłoni (np. do trenowania własnego klasyfikatora) |

Klawisze w oknie podglądu: `q` lub `Esc` – wyjście, `r` – włączenie/wyłączenie zapisu, `d` – przyciemnienie tła.

Zapisy gestów trafiają do folderu `recordings/` w formacie `.jsonl` (jedna linia JSON na zdarzenie).

### 2. Gra w Unity

1. Otwórz **Unity Hub** → **Add** → **Add project from disk** i wskaż folder `unityGame/unityGame`.
2. Otwórz projekt (Unity może zaproponować doinstalowanie odpowiedniej wersji edytora).
3. W oknie Project otwórz scenę `Assets/Scenes/SampleScene`.
4. Upewnij się, że wartość **Port** w komponencie **Hand Controller** na obiekcie `HandCursor` zgadza się z portem skryptu Python (domyślnie 5050).
5. **Najpierw uruchom skrypt Python**, a dopiero potem naciśnij **Play** w Unity.

### 3. Sterowanie

| Gest | Działanie w grze |
|---|---|
| Ruch dłonią przed kamerą | przesuwa czerwoną kulę (`HandCursor`) |
| `Closed_Fist` (pięść) | chwyta piłkę, jeśli kula dotyka piłki |
| `Open_Palm` (otwarta dłoń) | puszcza piłkę – leci zgodnie z prędkością ruchu dłoni |

Aby rzucić piłkę: dotknij jej czerwoną kulą, zaciśnij pięść, wykonaj szybki ruch w bok i otwórz dłoń.

## Format danych wysyłanych przez UDP

```json
{
  "t": 12.4,
  "hands": [
    {
      "hand": "Right",
      "gesture": "Open_Palm",
      "raw_gesture": "Open_Palm",
      "score": 0.97,
      "palm": { "x": 0.52, "y": 0.41, "z": -0.02 },
      "pinch": 0.83,
      "landmarks": [[0.5, 0.6, 0.0], "... 21 punktów ..."]
    }
  ]
}
```

- `gesture` – gest po wygładzeniu (musi utrzymać się przez 4 klatki z rzędu),
- `raw_gesture` – wynik modelu w tej konkretnej klatce,
- `palm` – środek dłoni we współrzędnych znormalizowanych (0–1, początek w lewym górnym rogu obrazu),
- `pinch` – stosunek odległości kciuka do palca wskazującego do rozmiaru dłoni.

Rozpoznawane gesty: `Closed_Fist`, `Open_Palm`, `Pointing_Up`, `Thumb_Up`, `Thumb_Down`, `Victory`, `ILoveYou`.

## Rozwiązywanie problemów

- **Kula w Unity się nie rusza** – sprawdź, czy skrypt Python został uruchomiony z flagą `--udp`, czy dłoń jest widoczna w kadrze i czy port w Unity jest taki sam jak w skrypcie.
- **`Only one usage of each socket address...`** lub **`forbidden by its access permissions`** – port jest zajęty albo zarezerwowany przez system. Zmień port (np. `--port 5055` w Pythonie i **Port** = 5055 w komponencie Hand Controller w Unity), a jeśli błąd nadal występuje, zrestartuj Unity. Zajęty port można sprawdzić poleceniem `netstat -ano | findstr 5050`.
- **Kamera się nie otwiera** – spróbuj `--camera 1` lub zamknij inne programy używające kamery.
- **Piłka odlatuje za daleko** – zmniejsz wartość **Throw Force** w komponencie Hand Controller.
- **Gest jest rozpoznawany niestabilnie** – zadbaj o równe oświetlenie i jednolite tło, trzymaj dłoń w odległości 40–80 cm od kamery.

## Technologie

Python, OpenCV, MediaPipe Tasks (GestureRecognizer), Unity 6 (C#), komunikacja UDP + JSON.

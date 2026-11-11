using UnityEngine;
using TMPro;

public class ObstacleSpawner : MonoBehaviour
{
    [Header("Interwał Czasowy Spawnu")]
    public float minSpawnInterval = 2.0f;
    public float maxSpawnInterval = 3.0f;

    [Header("Wymiary i Ruch Płytek")]
    [Tooltip("Głębokość Z (w głębi tunelu), na której pojawiają się płytki")]
    public float spawnDepthZ = 50.0f;
    [Tooltip("Początkowa (bazowa) prędkość lotu płytek w stronę gracza [m/s]")]
    public float baseTileSpeed = 10.0f;
    [Tooltip("O ile zwiększa się prędkość lotu płytek")]
    public float speedIncrement = 0.5f;
    [Tooltip("Co ile punktów prędkość ulega zwiększeniu")]
    public int pointsForSpeedUp = 5;

    public Vector2 minTileSize = new Vector2(1.4f, 1.4f);
    public Vector2 maxTileSize = new Vector2(2.8f, 2.6f);

    [Header("Granice Tunelu (Połowa szerokości i wysokości)")]
    [Tooltip("Maksymalne wychylenie od środka (0,0) - dopasowane do naszego tunelu 10.26 x 5.77")]
    public Vector2 tunnelHalfExtents = new Vector2(4.8f, 2.6f);

    [Header("Kolory i Wizualizacja")]
    [Tooltip("Opcjonalnie podepnij BulletSpawner, aby pobrać z niego dokładnie te same kolory")]
    public BulletSpawner bulletSpawnerRef;
    [Tooltip("Opcjonalny font obsługujący Emoji (np. Segoe UI Emoji SDF). Jeśli puste, wyświetli emotkę + skrót")]
    public TMP_FontAsset emojiFontAsset;

    // Domyślne kolory (identyczne z BulletSpawner.cs)
    public Color pointingUpColor = new Color(0.1f, 0.85f, 1.0f); // ☝️ Niebieski
    public Color thumbDownColor = new Color(1.0f, 0.20f, 0.25f); // 👎 Czerwony
    public Color thumbUpColor = new Color(0.2f, 1.00f, 0.35f); // 👍 Zielony
    public Color victoryColor = new Color(1.0f, 0.85f, 0.10f); // ✌️ Żółty

    private float spawnTimer;

    private static readonly ActiveGesture[] ValidGestures = new ActiveGesture[]
    {
        ActiveGesture.Pointing_Up,
        ActiveGesture.Thumb_Down,
        ActiveGesture.Thumb_Up,
        ActiveGesture.Victory
    };

    void Start()
    {
        ResetTimer();
    }

    void Update()
    {
        if (GameManager.Instance != null && GameManager.Instance.IsGameOver) return;

        spawnTimer -= Time.deltaTime;
        if (spawnTimer <= 0f)
        {
            SpawnObstacle();
            ResetTimer();
        }
    }

    private void ResetTimer()
    {
        // Opcjonalnie w przyszłości możesz tu również zmniejszać czas spawnu (min/max interval) wraz z wynikiem
        spawnTimer = Random.Range(minSpawnInterval, maxSpawnInterval);
    }

    private void SpawnObstacle()
    {
        // 1. Losowanie gestu (1 z 4)
        ActiveGesture chosenGesture = ValidGestures[Random.Range(0, ValidGestures.Length)];

        // 2. Losowanie szerokości i wysokości płytki
        float width = Random.Range(minTileSize.x, maxTileSize.x);
        float height = Random.Range(minTileSize.y, maxTileSize.y);
        Vector2 size = new Vector2(width, height);

        // 3. Losowanie pozycji X i Y z uwzględnieniem rozmiaru płytki (żeby nie wchodziła w ściany tunelu)
        float maxOffsetX = Mathf.Max(0.1f, tunnelHalfExtents.x - (width * 0.5f));
        float maxOffsetY = Mathf.Max(0.1f, tunnelHalfExtents.y - (height * 0.5f));

        float posX = Random.Range(-maxOffsetX, maxOffsetX);
        float posY = Random.Range(-maxOffsetY, maxOffsetY);
        Vector3 spawnPos = new Vector3(posX, posY, spawnDepthZ);

        // 4. Tworzenie obiektu płytki (Cube)
        GameObject tileObj = GameObject.CreatePrimitive(PrimitiveType.Cube);
        tileObj.name = $"ObstacleTile_{chosenGesture}";
        tileObj.transform.position = spawnPos;

        BoxCollider boxCol = tileObj.GetComponent<BoxCollider>();
        boxCol.isTrigger = true;

        // 5. Obliczenie aktualnej prędkości na podstawie wyniku gracza
        float currentTileSpeed = baseTileSpeed;
        if (GameManager.Instance != null)
        {
            // Dzielenie liczb całkowitych (np. 14 / 5 = 2). 
            // Prędkość wzrośnie o mnożnik dopiero przy pełnych piątkach.
            int multiplier = GameManager.Instance.Score / pointsForSpeedUp;
            currentTileSpeed += multiplier * speedIncrement;
        }

        // 6. Pobranie koloru i symbolu tekstowego
        Color tileColor = GetColor(chosenGesture);
        string symbol = GetDisplaySymbol(chosenGesture);

        ObstacleTile tileScript = tileObj.AddComponent<ObstacleTile>();
        tileScript.Initialize(chosenGesture, symbol, tileColor, currentTileSpeed, size, emojiFontAsset);
    }

    private Color GetColor(ActiveGesture gesture)
    {
        if (bulletSpawnerRef != null)
        {
            return bulletSpawnerRef.GetColorForGesture(gesture);
        }

        return gesture switch
        {
            ActiveGesture.Pointing_Up => pointingUpColor,
            ActiveGesture.Thumb_Down => thumbDownColor,
            ActiveGesture.Thumb_Up => thumbUpColor,
            ActiveGesture.Victory => victoryColor,
            _ => Color.white
        };
    }

    private string GetDisplaySymbol(ActiveGesture gesture)
    {
        return gesture switch
        {
            ActiveGesture.Pointing_Up => "☝",
            ActiveGesture.Thumb_Down => "👎",
            ActiveGesture.Thumb_Up => "👍",
            ActiveGesture.Victory => "✌",
            _ => "?"
        };
    }
}
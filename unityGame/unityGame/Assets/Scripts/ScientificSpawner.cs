using UnityEngine;
using TMPro;

public class ScientificSpawner : MonoBehaviour
{
    [Header("Referencje")]
    [Tooltip("Podepnij obiekt HandManager ze skryptem HandOperation")]
    public HandOperation handOperation;
    [Tooltip("Opcjonalnie podepnij BulletSpawner dla spójności kolorów")]
    public BulletSpawner bulletSpawnerRef;
    [Tooltip("Czcionka z wypieczonymi emotkami (SEGUIEMJ SDF)")]
    public TMP_FontAsset emojiFontAsset;

    [Header("Parametry Płytki (Stałe w trybie naukowym)")]
    [Tooltip("Głębokość Z spawnu płytki")]
    public float spawnDepthZ = 50.0f;
    [Tooltip("Stała prędkość (13 m/s gwarantuje dolot w 3.5s, czyli przed końcem 5-sekundowego okna)")]
    public float constantTileSpeed = 13.0f;
    [Tooltip("Stały rozmiar płytki, aby wyeliminować losowość rozmiaru w badaniu")]
    public Vector2 fixedTileSize = new Vector2(2.2f, 2.2f);
    public Vector2 tunnelHalfExtents = new Vector2(4.5f, 2.4f);

    [Header("Kolory Domyślne")]
    public Color pointingUpColor = new Color(0.1f, 0.85f, 1.0f);
    public Color thumbDownColor = new Color(1.0f, 0.20f, 0.25f);
    public Color thumbUpColor = new Color(0.2f, 1.00f, 0.35f);
    public Color victoryColor = new Color(1.0f, 0.85f, 0.10f);

    private int lastProcessedTrialId = 0;

    void Update()
    {
        if (GameManager.Instance != null && GameManager.Instance.IsGameOver) return;
        if (handOperation == null) return;

        // Sprawdzamy, czy Python rozpoczął nowe 5-sekundowe okno pomiarowe
        if (handOperation.LatestTrialId > lastProcessedTrialId)
        {
            lastProcessedTrialId = handOperation.LatestTrialId;
            SpawnCommandedObstacle(handOperation.LatestSpawnCommand);
        }
    }

    private void SpawnCommandedObstacle(string gestureName)
    {
        if (!TryParseGesture(gestureName, out ActiveGesture targetGesture))
        {
            return;
        }

        // Losujemy pozycję X, Y wewnątrz tunelu (lub ustaw 0, 0 jeśli chcesz badać tylko czas reakcji na środku)
        float maxOffsetX = Mathf.Max(0.1f, tunnelHalfExtents.x - (fixedTileSize.x * 0.5f));
        float maxOffsetY = Mathf.Max(0.1f, tunnelHalfExtents.y - (fixedTileSize.y * 0.5f));

        float posX = Random.Range(-maxOffsetX, maxOffsetX);
        float posY = Random.Range(-maxOffsetY, maxOffsetY);
        Vector3 spawnPos = new Vector3(posX, posY, spawnDepthZ);

        GameObject tileObj = GameObject.CreatePrimitive(PrimitiveType.Cube);
        tileObj.name = $"SciTile_Trial{lastProcessedTrialId}_{targetGesture}";
        tileObj.transform.position = spawnPos;

        BoxCollider boxCol = tileObj.GetComponent<BoxCollider>();
        boxCol.isTrigger = true;

        Color tileColor = GetColor(targetGesture);
        string symbol = GetDisplaySymbol(targetGesture);

        ObstacleTile tileScript = tileObj.AddComponent<ObstacleTile>();
        tileScript.Initialize(targetGesture, symbol, tileColor, constantTileSpeed, fixedTileSize, emojiFontAsset);
    }

    private bool TryParseGesture(string rawName, out ActiveGesture gesture)
    {
        switch (rawName)
        {
            case "Pointing_Up": gesture = ActiveGesture.Pointing_Up; return true;
            case "Thumb_Down": gesture = ActiveGesture.Thumb_Down; return true;
            case "Thumb_Up": gesture = ActiveGesture.Thumb_Up; return true;
            case "Victory": gesture = ActiveGesture.Victory; return true;
            default: gesture = ActiveGesture.None; return false;
        }
    }

    private Color GetColor(ActiveGesture gesture)
    {
        if (bulletSpawnerRef != null) return bulletSpawnerRef.GetColorForGesture(gesture);

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
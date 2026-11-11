using UnityEngine;
using TMPro;

public class ObstacleTile : MonoBehaviour
{
    [Header("Konfiguracja P³ytki")]
    public ActiveGesture requiredGesture;
    public float moveSpeedZ = 10.0f;

    [Tooltip("Pozycja Z gracza, po przekroczeniu której p³ytka zadaje obra¿enia")]
    public float playerHitZ = 4.5f;

    private TextMeshPro labelText;

    /// <summary>
    /// Inicjalizuje parametry p³ytki, jej kolor oraz tekst z emotikon¹ na przedniej œcianie.
    /// </summary>
    public void Initialize(ActiveGesture gesture, string emojiSymbol, Color tileColor, float speed, Vector2 sizeXY, TMP_FontAsset optionalFont = null)
    {
        this.requiredGesture = gesture;
        this.moveSpeedZ = speed;

        // 1. Skalowanie p³ytki (losowa szerokoœæ i wysokoœæ, sta³a gruboœæ 0.25)
        transform.localScale = new Vector3(sizeXY.x, sizeXY.y, 0.25f);

        // 2. Kolorowanie materia³u p³ytki
        Renderer rend = GetComponent<Renderer>();
        if (rend != null)
        {
            rend.material.color = tileColor;
        }

        // 3. Utworzenie lub pobranie napisu 3D (TextMeshPro) na froncie p³ytki
        labelText = GetComponentInChildren<TextMeshPro>();
        if (labelText == null)
        {
            GameObject textObj = new GameObject("EmojiLabel");
            textObj.transform.SetParent(this.transform);

            // Umieszczamy napis minimalnie przed przedni¹ œciank¹ p³ytki (lokalne Z = -0.55), skierowany w stronê kamery
            textObj.transform.localPosition = new Vector3(0f, 0f, -0.55f);
            textObj.transform.localRotation = Quaternion.identity;

            // Kompensacja skali rodzica, aby tekst nie by³ rozci¹gniêty, gdy p³ytka jest np. szeroka i niska
            textObj.transform.localScale = new Vector3(1f / sizeXY.x, 1f / sizeXY.y, 1f);

            labelText = textObj.AddComponent<TextMeshPro>();
            labelText.alignment = TextAlignmentOptions.Center;
            labelText.fontSize = Mathf.Min(sizeXY.x, sizeXY.y) * 4.5f;
            labelText.color = Color.white;
            labelText.outlineWidth = 0.2f;
            labelText.outlineColor = Color.black;

            // Wy³¹czamy zawijanie wierszy
            labelText.textWrappingMode = TextWrappingModes.NoWrap;
        }

        if (optionalFont != null)
        {
            labelText.font = optionalFont;
        }

        labelText.text = emojiSymbol;
    }

    void Update()
    {
        if (GameManager.Instance != null && GameManager.Instance.IsGameOver) return;

        // Ruch jednostajny prostoliniowy w stronê gracza (ujemna oœ Z)
        transform.position += Vector3.back * (moveSpeedZ * Time.deltaTime);

        // Sprawdzenie, czy p³ytka dolecia³a do gracza (uderzenie = utrata HP)
        if (transform.position.z <= playerHitZ)
        {
            if (GameManager.Instance != null)
            {
                GameManager.Instance.TakeDamage(1);
            }
            Destroy(gameObject);
        }
    }

    /// <summary>
    /// Wywo³ywane przez pocisk w momencie kolizji. Zwraca true, jeœli pocisk pasowa³ i rozbi³ p³ytkê.
    /// </summary>
    public bool TryShatter(ActiveGesture incomingBulletGesture)
    {
        if (incomingBulletGesture == requiredGesture)
        {
            if (GameManager.Instance != null)
            {
                GameManager.Instance.AddScore(1);
            }
            Destroy(gameObject);
            return true;
        }

        // Pocisk mia³ z³y gest - p³ytka nie zostaje zniszczona
        return false;
    }
}
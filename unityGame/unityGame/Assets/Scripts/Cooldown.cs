using UnityEngine;
using UnityEngine.UI;

public class Cooldown : MonoBehaviour
{
    [Header("Konfiguracja Cooldownu")]
    [Tooltip("Czas blokady miêdzy kolejnymi strza³ami w sekundach")]
    public float cooldownDuration = 1.0f;

    [Header("Wizualizacja UI (Opcjonalnie)")]
    [Tooltip("Podepnij UI Image ustawiony na Image Type -> Filled (Radial 360)")]
    public Image cooldownDialImage;

    private float remainingTime = 0f;

    /// <summary> Czy gracz mo¿e w tej klatce wystrzeliæ pocisk? </summary>
    public bool IsReady => remainingTime <= 0f;

    /// <summary> Postêp ³adowania od 0.0 (w³aœnie wystrzelono) do 1.0 (gotowy) </summary>
    public float Progress => Mathf.Clamp01(1f - (remainingTime / cooldownDuration));

    void Update()
    {
        if (remainingTime > 0f)
        {
            remainingTime -= Time.deltaTime;
            if (remainingTime < 0f) remainingTime = 0f;
        }

        if (cooldownDialImage != null)
        {
            cooldownDialImage.fillAmount = Progress;
        }
    }

    /// <summary>
    /// Próbuje zu¿yæ cooldown. Zwraca true i uruchamia zegar, jeœli strza³ by³ mo¿liwy.
    /// </summary>
    public bool TryConsumeCooldown()
    {
        if (!IsReady) return false;

        remainingTime = cooldownDuration;
        return true;
    }
}
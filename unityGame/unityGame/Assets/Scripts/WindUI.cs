using UnityEngine;
using TMPro; // Przestrzeñ nazw dla TextMeshPro

public class WindUI : MonoBehaviour
{
    [Header("Elementy UI do zaktualizowania")]
    [Tooltip("Obiekt ze strza³k¹ (Image), który bêdzie siê obraca³")]
    public RectTransform arrowTransform;

    [Tooltip("Pole tekstowe do wyœwietlania si³y wiatru")]
    public TextMeshProUGUI powerText;

    void Update()
    {
        // Upewniamy siê, ¿e instancja wiatru istnieje
        if (Wind.Instance == null) return;

        // 1. Obrót strza³ki
        // Pobieramy wyliczony ju¿ w skrypcie Wind.cs k¹t wiatru
        float angle = Wind.Instance.WindAngleDegrees;

        // Zmieniamy obrót osi Z, aby strza³ka wskazywa³a kierunek
        if (arrowTransform != null)
        {
            arrowTransform.rotation = Quaternion.Euler(0, 0, angle);
        }

        // 2. Wyœwietlanie si³y wiatru
        // "F1" formatuje liczbê do jednego miejsca po przecinku (np. 3.4 m/s)
        if (powerText != null)
        {
            powerText.text = Wind.Instance.CurrentIntensity.ToString("F1") + " m/s";
        }
    }
}
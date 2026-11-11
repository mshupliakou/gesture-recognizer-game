using System;
using UnityEngine;

public class Wind : MonoBehaviour
{
    public static Wind Instance { get; private set; }

    [Header("Parametry Czasowe")]
    [Tooltip("Czas w sekundach pomiêdzy zmianami kierunku i si³y wiatru")]
    public float changeInterval = 4.0f;

    [Header("Parametry Fizyczne Wiatru")]
    [Tooltip("Minimalna si³a (przyspieszenie) wiatru [m/s^2]")]
    public float minIntensity = 1.5f;
    [Tooltip("Maksymalna si³a (przyspieszenie) wiatru [m/s^2]")]
    public float maxIntensity = 6.0f;
    [Tooltip("Szybkoœæ p³ynnego przejœcia (interpolacji) miêdzy starym a nowym wiatrem")]
    public float transitionSpeed = 3.0f;

    // --- W³aœciwoœci odczytywane przez pociski i UI ---
    /// <summary> Znormalizowany wektor kierunku w p³aszczyŸnie XY (d³ugoœæ = 1) </summary>
    public Vector2 CurrentDirection { get; private set; } = Vector2.zero;

    /// <summary> Aktualne natê¿enie (si³a) wiatru </summary>
    public float CurrentIntensity { get; private set; } = 0f;

    /// <summary> Pe³ny wektor przyspieszenia wiatru w 3D: W = [Wx, Wy, 0] </summary>
    public Vector3 WindForceVector { get; private set; } = Vector3.zero;

    /// <summary> K¹t w stopniach (0-360) do obracania strza³ki UI na ekranie </summary>
    public float WindAngleDegrees => Mathf.Atan2(CurrentDirection.y, CurrentDirection.x) * Mathf.Rad2Deg;

    public event Action<Vector2, float> OnWindChanged;

    private Vector2 targetDirection;
    private float targetIntensity;
    private float timer;

    void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
    }

    void Start()
    {
        GenerateNewWind();
        // Inicjalizacja wartoœci pocz¹tkowych bez czekania na interpolacjê
        CurrentDirection = targetDirection;
        CurrentIntensity = targetIntensity;
        UpdateWindVector();
    }

    void Update()
    {
        timer += Time.deltaTime;
        if (timer >= changeInterval)
        {
            timer = 0f;
            GenerateNewWind();
        }

        // P³ynna interpolacja wektora wiatru, aby unikn¹æ nienaturalnych przeskoków trajektorii
        CurrentDirection = Vector2.Lerp(CurrentDirection, targetDirection, Time.deltaTime * transitionSpeed).normalized;
        CurrentIntensity = Mathf.Lerp(CurrentIntensity, targetIntensity, Time.deltaTime * transitionSpeed);

        UpdateWindVector();
    }

    private void GenerateNewWind()
    {
        // Losowy k¹t w pe³nym zakresie [0, 2*PI] w p³aszczyŸnie pionowo-poziomej (X, Y)
        float randomAngleRad = UnityEngine.Random.Range(0f, Mathf.PI * 2f);
        targetDirection = new Vector2(Mathf.Cos(randomAngleRad), Mathf.Sin(randomAngleRad)).normalized;
        targetIntensity = UnityEngine.Random.Range(minIntensity, maxIntensity);

        OnWindChanged?.Invoke(targetDirection, targetIntensity);
    }

    private void UpdateWindVector()
    {
        WindForceVector = new Vector3(
            CurrentDirection.x * CurrentIntensity,
            CurrentDirection.y * CurrentIntensity,
            0f // Wiatr dzia³a wy³¹cznie prostopadle do osi tunelu (Z)
        );
    }
}
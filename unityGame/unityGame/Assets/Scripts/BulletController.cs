using UnityEngine;

public class BulletController : MonoBehaviour
{
    [Header("Identyfikacja Gestu")]
    [Tooltip("Typ gestu, który stworzy³ ten pocisk (niszczy tylko odpowiadaj¹ce mu przeszkody)")]
    public ActiveGesture bulletType;

    [Header("Parametry Fizyczne Pocisku")]
    [Tooltip("Prêdkoœæ pocz¹tkowa nadawana wzd³u¿ osi Z [m/s]")]
    public float initialSpeedZ = 18.0f;
    [Tooltip("Masa pocisku [kg] - im wiêksza, tym wiêksza bezw³adnoœæ i mniejszy wp³yw wiatru")]
    public float mass = 1.0f;
    [Tooltip("Wspó³czynnik podatnoœci na wiatr (alfa)")]
    public float windSusceptibility = 2.2f;
    [Tooltip("Wspó³czynnik oporu powietrza (t³umienie liniowe w p³aszczyŸnie XY)")]
    public float airDragCoefficient = 0.25f;

    [Header("Cykl ¯ycia Pocisku")]
    [Tooltip("Maksymalna g³êbokoœæ Z (koniec tunelu), po której pocisk znika")]
    public float maxTunnelDepthZ = 55.0f;
    [Tooltip("Maksymalny czas ¿ycia pocisku w sekundach")]
    public float maxLifetime = 5.0f;

    // Chwilowe wektory stanu kinematycznego pocisku
    private Vector3 currentVelocity;
    private Vector3 currentAcceleration;
    private float lifeTimer;

    /// <summary>
    /// Inicjalizuje stan pocz¹tkowy pocisku zaraz po zespawnowaniu.
    /// </summary>
    public void Initialize(ActiveGesture gestureType, Color bulletColor)
    {
        this.bulletType = gestureType;

        // Warunek pocz¹tkowy: V(0) = [0, 0, v_init]
        currentVelocity = new Vector3(0f, 0f, initialSpeedZ);
        currentAcceleration = Vector3.zero;
        lifeTimer = 0f;

        // Przypisanie koloru odpowiadaj¹cego danemu gestowi
        Renderer rend = GetComponent<Renderer>();
        if (rend != null)
        {
            rend.material.color = bulletColor;
        }
    }

    void Update()
    {
        float dt = Time.deltaTime;
        if (dt <= 0f) return;

        // --- KROK 1: Pobranie wektora si³y wiatru z globalnego systemu Wind.cs ---
        Vector3 windVector = (Wind.Instance != null) ? Wind.Instance.WindForceVector : Vector3.zero;

        // --- KROK 2: Wyznaczenie si³ dzia³aj¹cych na pocisk ---
        // 2a. Si³a parcia bocznego wiatru: F_wind = alfa * W
        Vector3 windForce = windVector * windSusceptibility;

        // 2b. Si³a oporu powietrza (dzia³aj¹ca w osiach X i Y, by pocisk nie zwalnia³ biegu przez tunel w osi Z)
        Vector3 lateralVelocity = new Vector3(currentVelocity.x, currentVelocity.y, 0f);
        Vector3 dragForce = -airDragCoefficient * lateralVelocity;

        // 2c. Si³a wypadkowa: F_net = F_wind + F_drag
        Vector3 netForce = windForce + dragForce;

        // --- KROK 3: II Zasada Dynamiki Newtona (a = F_net / m) ---
        currentAcceleration = netForce / Mathf.Max(mass, 0.001f);

        // --- KROK 4: Ca³kowanie numeryczne (Metoda Semi-Implicit Eulera) ---
        // 4a. Aktualizacja prêdkoœci: V(t + dt) = V(t) + a(t) * dt
        currentVelocity += currentAcceleration * dt;

        // 4b. Aktualizacja po³o¿enia: P(t + dt) = P(t) + V(t + dt) * dt
        transform.position += currentVelocity * dt;

        // --- KROK 5: Kontrola opuszczenia obszaru gry (Culling) ---
        lifeTimer += dt;
        if (transform.position.z >= maxTunnelDepthZ || lifeTimer >= maxLifetime)
        {
            Destroy(gameObject);
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        // Sprawdzamy, czy obiekt, w który uderzy³ pocisk, posiada skrypt ObstacleTile
        ObstacleTile obstacle = other.GetComponent<ObstacleTile>();
        if (obstacle != null)
        {
            // Wywo³ujemy próbê rozbicia p³ytki (zwróci true i doda +1 do Score, jeœli gest pocisku == gest p³ytki)
            obstacle.TryShatter(this.bulletType);

            // Pocisk ulega zniszczeniu po zderzeniu z p³ytk¹ (niezale¿nie czy mia³ dobry, czy z³y kolor)
            Destroy(gameObject);
        }
    }
}
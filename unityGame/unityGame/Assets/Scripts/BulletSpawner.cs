using UnityEngine;

public class BulletSpawner : MonoBehaviour
{
    [Header("Referencje do Modułów")]
    public HandOperation handOperation;
    public Cooldown cooldownSystem;

    [Header("Prefab i Rozmiar Pocisku")]
    [Tooltip("Opcjonalny Prefab pocisku. Jeśli puste, skrypt sam stworzy kulę w kodzie.")]
    public GameObject bulletPrefab;
    public float bulletDiameter = 0.45f;

    [Header("Kolory Pocisków dla 4 Gestów")]
    public Color pointingUpColor = new Color(0.1f, 0.85f, 1.0f); // ☝️ Błękitny (Cyjan)
    public Color thumbDownColor = new Color(1.0f, 0.20f, 0.25f); // 👎 Czerwony
    public Color thumbUpColor = new Color(0.2f, 1.00f, 0.35f); // 👍 Zielony
    public Color victoryColor = new Color(1.0f, 0.85f, 0.10f); // ✌️ Żółty / Złoty

    void Update()
    {
        if (handOperation == null || cooldownSystem == null) return;

        // Sprawdzamy, czy dłoń jest widoczna, gest należy do naszych 4 wybranych
        // oraz czy pewność modelu (Score %) jest wystarczająca
        if (handOperation.IsHandVisible &&
            handOperation.CurrentGesture != ActiveGesture.None)
        {
            // Jeśli cooldown minął (1 sekunda), konsumujemy go i wystrzeliwujemy pocisk
            if (cooldownSystem.TryConsumeCooldown())
            {
                SpawnBullet(handOperation.SpawnPositionWorld, handOperation.CurrentGesture);
            }
        }
    }

    private void SpawnBullet(Vector3 spawnPosition, ActiveGesture gesture)
    {
        GameObject bulletObj;

        if (bulletPrefab != null)
        {
            bulletObj = Instantiate(bulletPrefab, spawnPosition, Quaternion.identity);
        }
        else
        {
            // Automatyczne utworzenie kulistego pocisku, jeśli nie podpięto Prefabu
            bulletObj = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            bulletObj.name = $"Bullet_{gesture}";
            bulletObj.transform.position = spawnPosition;
            bulletObj.transform.localScale = Vector3.one * bulletDiameter;

            // Konfiguracja detekcji kolizji (Trigger + Kinematyczny Rigidbody wymagany przez Unity do wywoływania OnTriggerEnter)
            SphereCollider col = bulletObj.GetComponent<SphereCollider>();
            col.isTrigger = true;

            Rigidbody rb = bulletObj.AddComponent<Rigidbody>();
            rb.isKinematic = true;
            rb.useGravity = false;
        }

        BulletController controller = bulletObj.GetComponent<BulletController>();
        if (controller == null)
        {
            controller = bulletObj.AddComponent<BulletController>();
        }

        Color assignedColor = GetColorForGesture(gesture);
        controller.Initialize(gesture, assignedColor);
    }

    public Color GetColorForGesture(ActiveGesture gesture)
    {
        return gesture switch
        {
            ActiveGesture.Pointing_Up => pointingUpColor,
            ActiveGesture.Thumb_Down => thumbDownColor,
            ActiveGesture.Thumb_Up => thumbUpColor,
            ActiveGesture.Victory => victoryColor,
            _ => Color.white
        };
    }
}
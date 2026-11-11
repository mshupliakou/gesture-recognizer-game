using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using TMPro; // Wymagane do obsługi TextMeshPro

#region Struktury JSON (Unikalne nazwy, aby nie kolidowały z HandController.cs)
[Serializable]
public class HandOpPacket
{
    public float t;
    public int trial_id;
    public string spawn_command;
    public HandOpData[] hands;
}

[Serializable]
public class HandOpData
{
    public string hand;
    public string gesture;
    public string raw_gesture;
    public float score;
    public HandOpPoint palm;
    public float pinch;
    public HandOpPoint[] landmarks;
}

[Serializable]
public class HandOpPoint
{
    public float x;
    public float y;
    public float z;
}
#endregion

public enum ActiveGesture
{
    None = 0,
    Pointing_Up = 3, // ☝️
    Thumb_Down = 4,  // 👎
    Thumb_Up = 5,    // 👍
    Victory = 6      // ✌️
}

public class HandOperation : MonoBehaviour
{
    [Header("Ustawienia Sieci (UDP)")]
    public int port = 5050;
    public float handTimeout = 0.35f;

    [Header("Mapowanie Przestrzeni Gry")]
    [Tooltip("Kamera gracza (jeśli puste, użyje Camera.main)")]
    public Camera gameCamera;
    [Tooltip("Odległość szkieletu dłoni przed kamerą (oś Z)")]
    public float distanceFromCamera = 5.0f;
    [Tooltip("Wzmocnienie głębi 3D dla poszczególnych palców")]
    public float depthMultiplier = 2.5f;
    [Tooltip("Szybkość wygładzania ruchu dłoni")]
    public float smoothingSpeed = 18f;

    [Header("Wygląd Szkieletu (Dots & Lines)")]
    public float jointRadius = 0.08f;
    public float boneWidth = 0.03f;
    public Color jointColor = Color.white;
    public Color boneColor = new Color(0f, 0.85f, 1f, 1f);

    [Header("Wizualizacja UI przy Dłoni")]
    [Tooltip("Czcionka z wypieczonymi emotkami (np. SEGUIEMJ SDF)")]
    public TMP_FontAsset emojiFontAsset;
    [Tooltip("Przesunięcie tekstu względem środka dłoni")]
    public Vector3 textOffset = new Vector3(-1.2f, 1.0f, 0f);

    // --- DANE WYJŚCIOWE ---
    public ActiveGesture CurrentGesture { get; private set; } = ActiveGesture.None;
    public string CurrentEmoji { get; private set; } = "";
    public int CurrentScorePercent { get; private set; } = 0;
    public Vector2 SpawnCoordsNormalized { get; private set; } = new Vector2(0.5f, 0.5f);
    public Vector3 SpawnPositionWorld { get; private set; } = Vector3.zero;
    public bool IsHandVisible { get; private set; } = false;
    public int LatestTrialId { get; private set; } = 0;
    public string LatestSpawnCommand { get; private set; } = "";

    // --- Pola prywatne ---
    private UdpClient udpClient;
    private Thread receiveThread;
    private volatile bool running;
    private readonly object lockObj = new object();
    private string latestJson;
    private bool hasNewData;
    private float lastPacketTime;

    private Transform skeletonRoot;
    private Transform[] jointTransforms = new Transform[21];
    private Vector3[] targetJointPositions = new Vector3[21];
    private LineRenderer[] boneLines;
    private Vector3 targetSpawnWorld;

    // Obiekt wyświetlający tekst
    private TextMeshPro handInfoText;

    private static readonly int[,] BoneConnections = new int[,]
    {
        {0, 1}, {1, 2}, {2, 3}, {3, 4},
        {0, 5}, {5, 6}, {6, 7}, {7, 8},
        {5, 9}, {9, 10}, {10, 11}, {11, 12},
        {9, 13}, {13, 14}, {14, 15}, {15, 16},
        {13, 17}, {17, 18}, {18, 19}, {19, 20},
        {0, 17}
    };

    private static readonly Dictionary<string, (ActiveGesture enumVal, string emoji)> AllowedGestures = new()
    {
        { "Pointing_Up", (ActiveGesture.Pointing_Up, "☝") }, // Używamy wersji bazowych znaków dla zgodności z czcionką
        { "Thumb_Down",  (ActiveGesture.Thumb_Down,  "👎") },
        { "Thumb_Up",    (ActiveGesture.Thumb_Up,    "👍") },
        { "Victory",     (ActiveGesture.Victory,     "✌") }
    };

    void Start()
    {
        if (gameCamera == null) gameCamera = Camera.main;

        InitializeSkeletonVisuals();

        running = true;
        receiveThread = new Thread(ReceiveUDPData) { IsBackground = true };
        receiveThread.Start();
    }

    void Update()
    {
        string json = null;
        lock (lockObj)
        {
            if (hasNewData)
            {
                json = latestJson;
                hasNewData = false;
            }
        }

        if (json != null)
        {
            ProcessPacket(json);
        }

        if (Time.time - lastPacketTime > handTimeout)
        {
            SetHandVisibility(false);
        }

        if (IsHandVisible)
        {
            float lerpFactor = 1f - Mathf.Exp(-smoothingSpeed * Time.deltaTime);
            SpawnPositionWorld = Vector3.Lerp(SpawnPositionWorld, targetSpawnWorld, lerpFactor);

            for (int i = 0; i < 21; i++)
            {
                jointTransforms[i].position = Vector3.Lerp(jointTransforms[i].position, targetJointPositions[i], lerpFactor);
            }

            UpdateBonesPositions();
            UpdateHandInfoText();
        }
    }

    private void UpdateHandInfoText()
    {
        if (handInfoText == null) return;

        // Jeśli gest jest prawidłowy (1 z 4)
        if (CurrentGesture != ActiveGesture.None)
        {
            handInfoText.gameObject.SetActive(true);
            handInfoText.text = $"{CurrentEmoji}\n{CurrentScorePercent}%";

            // Pozycjonujemy tekst obok dłoni
            handInfoText.transform.position = SpawnPositionWorld + textOffset;

            // Ustawiamy rotację, aby tekst zawsze patrzył idealnie w stronę kamery
            handInfoText.transform.rotation = Quaternion.LookRotation(handInfoText.transform.position - gameCamera.transform.position);
        }
        else
        {
            // Ukrywamy tekst, jeśli gest to np. "Open_Palm" lub słaby wynik
            handInfoText.gameObject.SetActive(false);
        }
    }

    private void ProcessPacket(string json)
    {
        HandOpPacket packet = JsonUtility.FromJson<HandOpPacket>(json);
        if (packet == null) return;

        // Zapisujemy komendę spawnowania niezależnie od tego, czy dłoń jest obecnie widoczna w kamerze!
        if (packet.trial_id > 0 && !string.IsNullOrEmpty(packet.spawn_command))
        {
            LatestTrialId = packet.trial_id;
            LatestSpawnCommand = packet.spawn_command;
        }

        if (packet.hands == null || packet.hands.Length == 0)
        {
            SetHandVisibility(false);
            return;
        }

        if (packet == null || packet.hands == null || packet.hands.Length == 0)
        {
            SetHandVisibility(false);
            return;
        }

        HandOpData hand = packet.hands[0];
        lastPacketTime = Time.time;

        bool wasVisible = IsHandVisible;
        SetHandVisibility(true);

        if (AllowedGestures.TryGetValue(hand.gesture, out var mappedGesture))
        {
            CurrentGesture = mappedGesture.enumVal;
            CurrentEmoji = mappedGesture.emoji;
            CurrentScorePercent = Mathf.RoundToInt(hand.score * 100f);
        }
        else
        {
            CurrentGesture = ActiveGesture.None;
            CurrentEmoji = "";
            CurrentScorePercent = 0;
        }

        float normX = Mathf.Clamp01(hand.palm.x);
        float normY = Mathf.Clamp01(1f - hand.palm.y);
        SpawnCoordsNormalized = new Vector2(normX, normY);

        targetSpawnWorld = ViewportToWorld(normX, normY, hand.palm.z);
        if (!wasVisible) SpawnPositionWorld = targetSpawnWorld;

        if (hand.landmarks != null && hand.landmarks.Length == 21)
        {
            for (int i = 0; i < 21; i++)
            {
                float lx = Mathf.Clamp01(hand.landmarks[i].x);
                float ly = Mathf.Clamp01(1f - hand.landmarks[i].y);
                float lz = hand.landmarks[i].z;

                Vector3 worldPt = ViewportToWorld(lx, ly, lz);
                targetJointPositions[i] = worldPt;

                if (!wasVisible) jointTransforms[i].position = worldPt;
            }
        }
    }

    private Vector3 ViewportToWorld(float viewportX, float viewportY, float relativeZ)
    {
        float depth = distanceFromCamera + (relativeZ * depthMultiplier);
        return gameCamera.ViewportToWorldPoint(new Vector3(viewportX, viewportY, depth));
    }

    #region Generowanie i Obsługa Wizualizacji Szkieletu
    private void InitializeSkeletonVisuals()
    {
        skeletonRoot = new GameObject("HandSkeleton_Visuals").transform;
        skeletonRoot.SetParent(this.transform);

        Material baseMat = new Material(Shader.Find("Sprites/Default"));

        for (int i = 0; i < 21; i++)
        {
            GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere.name = $"Joint_{i}";
            sphere.transform.SetParent(skeletonRoot);
            float scale = (i == 4 || i == 8 || i == 12 || i == 16 || i == 20) ? jointRadius * 1.3f : jointRadius;
            sphere.transform.localScale = Vector3.one * scale;
            Destroy(sphere.GetComponent<Collider>());
            Renderer r = sphere.GetComponent<Renderer>();
            r.material = new Material(baseMat) { color = jointColor };
            jointTransforms[i] = sphere.transform;
        }

        int boneCount = BoneConnections.GetLength(0);
        boneLines = new LineRenderer[boneCount];

        for (int i = 0; i < boneCount; i++)
        {
            GameObject lineObj = new GameObject($"Bone_{i}");
            lineObj.transform.SetParent(skeletonRoot);
            LineRenderer lr = lineObj.AddComponent<LineRenderer>();
            lr.material = new Material(baseMat) { color = boneColor };
            lr.startColor = boneColor;
            lr.endColor = boneColor;
            lr.startWidth = boneWidth;
            lr.endWidth = boneWidth;
            lr.positionCount = 2;
            lr.numCapVertices = 4;
            boneLines[i] = lr;
        }

        // --- Tworzenie pływającego tekstu z informacjami o geście ---
        GameObject textObj = new GameObject("HandGestureInfoText");
        textObj.transform.SetParent(skeletonRoot);

        handInfoText = textObj.AddComponent<TextMeshPro>();
        handInfoText.alignment = TextAlignmentOptions.Center;
        handInfoText.fontSize = 4f;
        handInfoText.color = Color.white;
        handInfoText.outlineWidth = 0.2f;
        handInfoText.outlineColor = Color.black;
        handInfoText.textWrappingMode = TextWrappingModes.NoWrap;

        if (emojiFontAsset != null)
        {
            handInfoText.font = emojiFontAsset;
        }

        SetHandVisibility(false);
    }

    private void UpdateBonesPositions()
    {
        int boneCount = BoneConnections.GetLength(0);
        for (int i = 0; i < boneCount; i++)
        {
            int idxA = BoneConnections[i, 0];
            int idxB = BoneConnections[i, 1];
            boneLines[i].SetPosition(0, jointTransforms[idxA].position);
            boneLines[i].SetPosition(1, jointTransforms[idxB].position);
        }
    }

    private void SetHandVisibility(bool visible)
    {
        if (IsHandVisible == visible && skeletonRoot.gameObject.activeSelf == visible) return;

        IsHandVisible = visible;
        if (skeletonRoot != null) skeletonRoot.gameObject.SetActive(visible);

        if (!visible)
        {
            CurrentGesture = ActiveGesture.None;
            CurrentEmoji = "";
            CurrentScorePercent = 0;
            if (handInfoText != null) handInfoText.gameObject.SetActive(false);
        }
    }
    #endregion

    #region Wątek UDP
    private void ReceiveUDPData()
    {
        try
        {
            udpClient = new UdpClient(port);
        }
        catch (Exception e)
        {
            Debug.LogError($"[HandOperation] Nie udało się otworzyć portu UDP {port}: {e.Message}");
            return;
        }

        IPEndPoint remote = new IPEndPoint(IPAddress.Any, 0);
        while (running)
        {
            try
            {
                byte[] data = udpClient.Receive(ref remote);
                string text = Encoding.UTF8.GetString(data);
                lock (lockObj)
                {
                    latestJson = text;
                    hasNewData = true;
                }
            }
            catch (SocketException) { if (!running) break; }
            catch (ObjectDisposedException) { break; }
            catch (Exception e) { Debug.LogWarning($"[HandOperation] Błąd UDP: {e.Message}"); }
        }
    }

    void OnDestroy()
    {
        running = false;
        if (udpClient != null) udpClient.Close();
    }
    #endregion
}
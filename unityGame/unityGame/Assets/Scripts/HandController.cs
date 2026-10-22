using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

[Serializable] public class HandPacket { public HandData[] hands; }
[Serializable] public class HandData { public string gesture; public PalmData palm; }
[Serializable] public class PalmData { public float x; public float y; public float z; }

public class HandController : MonoBehaviour
{
	[Header("Сетевые настройки")]
	public int port = 5050;

	[Header("Настройки движения и броска")]
	public float movementMultiplier = 15f;
	public float throwForce = 1f;
	public float handTimeout = 0.5f;   // если данные не приходят дольше — считаем, что руки нет

	private UdpClient udpClient;
	private Thread receiveThread;
	private volatile bool running;
	private readonly object lockObj = new object();
	private string latestJson;
	private bool hasNewData;
	private float lastPacketTime;

	private GameObject hoveredObject;
	private GameObject grabbedObject;
	private Vector3 targetPosition;
	private Vector3 handVelocity;
	private string currentGesture = "None";

	void Start()
	{
		targetPosition = transform.position;
		running = true;
		receiveThread = new Thread(ReceiveData) { IsBackground = true };
		receiveThread.Start();
	}

	private void ReceiveData()
	{
		try
		{
			udpClient = new UdpClient(port);
		}
		catch (Exception e)
		{
			Debug.LogError("Не удалось открыть порт " + port + ": " + e.Message);
			return;
		}

		IPEndPoint remote = new IPEndPoint(IPAddress.Any, 0);
		while (running)
		{
			try
			{
				byte[] data = udpClient.Receive(ref remote);
				lock (lockObj)
				{
					latestJson = Encoding.UTF8.GetString(data);
					hasNewData = true;
				}
			}
			catch (SocketException) { if (!running) break; }   // сокет закрыт в OnDestroy
			catch (ObjectDisposedException) { break; }
			catch (Exception e) { Debug.LogWarning(e.Message); }
		}
	}

	void Update()
	{
		// 1. Забираем новый пакет, если он есть
		string json = null;
		lock (lockObj)
		{
			if (hasNewData) { json = latestJson; hasNewData = false; }
		}
		if (json != null) ProcessPacket(json);

		// Руку потеряли — отпускаем жест
		if (Time.time - lastPacketTime > handTimeout) currentGesture = "None";

		// 2. Плавное движение + скорость руки (каждый кадр)
		Vector3 prev = transform.position;
		transform.position = Vector3.Lerp(prev, targetPosition, 1f - Mathf.Exp(-15f * Time.deltaTime));
		Vector3 instantVel = (transform.position - prev) / Mathf.Max(Time.deltaTime, 0.0001f);
		handVelocity = Vector3.Lerp(handVelocity, instantVel, 0.5f);

		// 3. Логика захвата/броска
		if (currentGesture == "Closed_Fist") TryGrab();
		else if (currentGesture == "Open_Palm") TryRelease();
	}

	private void ProcessPacket(string json)
	{
		HandPacket packet = JsonUtility.FromJson<HandPacket>(json);
		if (packet == null || packet.hands == null || packet.hands.Length == 0) return;

		HandData hand = packet.hands[0];
		lastPacketTime = Time.time;
		currentGesture = hand.gesture;

		float x = (hand.palm.x - 0.5f) * movementMultiplier;
		float y = (0.5f - hand.palm.y) * movementMultiplier;
		targetPosition = new Vector3(x, y, 0f);
	}

	private void OnTriggerEnter(Collider other)
	{
		if (other.CompareTag("Grabbable")) hoveredObject = other.gameObject;
	}

	private void OnTriggerExit(Collider other)
	{
		if (other.gameObject == hoveredObject) hoveredObject = null;
	}

	private void TryGrab()
	{
		if (hoveredObject != null && grabbedObject == null)
		{
			grabbedObject = hoveredObject;
			grabbedObject.GetComponent<Rigidbody>().isKinematic = true;
			grabbedObject.transform.SetParent(transform);
		}
	}

	private void TryRelease()
	{
		if (grabbedObject != null)
		{
			Rigidbody rb = grabbedObject.GetComponent<Rigidbody>();
			grabbedObject.transform.SetParent(null);
			rb.isKinematic = false;
			rb.linearVelocity = handVelocity * throwForce;   // в Unity 6 вместо rb.velocity
			grabbedObject = null;
		}
	}

	void OnDestroy()
	{
		running = false;
		if (udpClient != null) udpClient.Close();   // это прерывает блокирующий Receive
	}
}
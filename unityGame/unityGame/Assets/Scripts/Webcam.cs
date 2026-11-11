using UnityEngine;

public class Webcam : MonoBehaviour
{
    [SerializeField] Material webcamMaterial;
    [SerializeField] string webcamName;

    void Start()
    {
        WebCamDevice[] devices = WebCamTexture.devices;

        // Log out each webcam
        Debug.Log("Available webcams:");
        foreach (WebCamDevice webcam in devices)
        {
            Debug.Log(webcam.name);
        }

        // We get the webcam by name.
        // Can just call new() to get the default.
        WebCamTexture webcamTexture = new(webcamName, 1920, 1080, 30);

        // Set our material texture to be our webcam texture.
        webcamMaterial.mainTexture = webcamTexture;

        // Start the webcam.
        webcamTexture.Play();

    }
}
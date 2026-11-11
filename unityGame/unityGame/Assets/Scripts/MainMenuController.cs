using UnityEngine;
using UnityEngine.SceneManagement;

public class MainMenuController : MonoBehaviour
{
    // Called by the "Play" button
    public void StandardGame()
    {
        // Loads the "Level" scene (Index 3 in your Build Settings)
        SceneManager.LoadScene("Level");
    }

    public void StatsGame()
    {
        SceneManager.LoadScene("Stats");
    }

    // Called by the "Quit" button
    public void QuitGame()
    {
        Debug.Log("Game is exiting...");
        Application.Quit();
    }
}
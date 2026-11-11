using UnityEngine;
using UnityEngine.SceneManagement;
using TMPro;

public class GameManager : MonoBehaviour
{
    public static GameManager Instance { get; private set; }

    [Header("Stan Rozgrywki")]
    public int maxHP = 3;
    public int CurrentHP { get; private set; }
    public int Score { get; private set; } = 0;
    public bool IsGameOver { get; private set; } = false;

    [Header("Elementy UI (Góra Ekranu)")]
    public TextMeshProUGUI scoreText;
    public TextMeshProUGUI hpText;

    [Header("Game Over UI")]
    public GameObject gameOverPanel; // The panel containing "You lost!" and the two buttons

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
        CurrentHP = maxHP;
        Score = 0;
        IsGameOver = false;

        if (gameOverPanel != null)
            gameOverPanel.SetActive(false);

        UpdateUI();
    }

    public void AddScore(int points = 1)
    {
        if (IsGameOver) return;
        Score += points;
        UpdateUI();
    }

    public void TakeDamage(int damage = 1)
    {
        if (IsGameOver) return;
        CurrentHP = Mathf.Max(0, CurrentHP - damage);
        UpdateUI();

        if (CurrentHP <= 0)
        {
            TriggerGameOver();
        }
    }

    private void UpdateUI()
    {
        if (scoreText != null)
            scoreText.text = $"SCORE: {Score}";

        if (hpText != null)
        {
            string hearts = "";
            for (int i = 0; i < maxHP; i++)
            {
                hearts += (i < CurrentHP) ? "♥ " : "";
            }
            hpText.text = hearts.TrimEnd();
        }
    }

    private void TriggerGameOver()
    {
        IsGameOver = true;
        if (gameOverPanel != null)
        {
            gameOverPanel.SetActive(true); // Shows the "You lost!" screen
        }
    }

    // --- BUTTON METHODS FOR GAME OVER SCREEN ---

    // Called by the "Restart" button
    public void RestartLevel()
    {
        // Reloads the current scene ("Level")
        SceneManager.LoadScene(SceneManager.GetActiveScene().buildIndex);
    }

    // Called by the "Main menu" button
    public void LoadMainMenu()
    {
        // Loads the "MainMenu" scene (Index 1 in your Build Settings)
        SceneManager.LoadScene("MainMenu");
    }
}
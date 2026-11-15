import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def generate_benchmark_plots(csv_path=None):
    os.makedirs('benchmarks', exist_ok=True)
    
    if csv_path is None:
        csv_files = glob.glob('benchmarks/*.csv')
        if not csv_files:
            print("Nie znaleziono plików CSV w katalogu benchmarks/.")
            return
        csv_path = max(csv_files, key=os.path.getctime)
        
    print(f"Wczytywanie danych z: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # -------------------------------------------------------------
    # 1. GENEROWANIE STATYSTYK TEKSTOWYCH (DODANE)
    # -------------------------------------------------------------
    total = len(df)
    mp_success = df['mp_detected'].sum()
    custom_success = df['custom_detected'].sum()

    mp_avg_time = df['mp_time_to_detect_s'].mean()
    custom_avg_time = df['custom_time_to_detect_s'].mean()

    mp_false = df['mp_false_activations'].sum()
    custom_false = df['custom_false_activations'].sum()

    winners = df['faster_model'].value_counts().to_dict()

    gesture_stats = df.groupby('target_gesture')[['mp_detected', 'custom_detected']].sum()
    gesture_counts = df['target_gesture'].value_counts()

    stats_output = []
    stats_output.append(f"Zestaw danych: {csv_path}")
    stats_output.append(f"Liczba wszystkich prób: {total}")
    stats_output.append(f"Skuteczność MediaPipe: {mp_success}/{total} ({mp_success/total:.1%})")
    stats_output.append(f"Skuteczność Własnego Modelu: {custom_success}/{total} ({custom_success/total:.1%})")
    stats_output.append(f"Średni czas detekcji MediaPipe: {mp_avg_time:.3f} s")
    stats_output.append(f"Średni czas detekcji Własnego Modelu: {custom_avg_time:.3f} s")
    stats_output.append(f"Błędne aktywacje MediaPipe: {mp_false}")
    stats_output.append(f"Błędne aktywacje Własnego Modelu: {custom_false}")
    stats_output.append("Wyniki rywalizacji (który szybszy):")
    for winner, count in winners.items():
        stats_output.append(f"  - {winner}: {count}")
    stats_output.append("Skuteczność dla poszczególnych gestów:")
    for gest in gesture_counts.index:
        count = gesture_counts[gest]
        mp_det = gesture_stats.loc[gest, 'mp_detected']
        cust_det = gesture_stats.loc[gest, 'custom_detected']
        stats_output.append(f"  - {gest} (N={count}): MP={mp_det/count:.1%} | Custom={cust_det/count:.1%}")
    
    stats_text = "\n".join(stats_output)
    print(stats_text)
    
    # Zapis statystyk do pliku txt
    txt_path = 'benchmarks/benchmark_summary.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(stats_text)
    print(f"Zapisano statystyki tekstowe do: {txt_path}")

    # -------------------------------------------------------------
    # 2. GENEROWANIE WYKRESÓW
    # -------------------------------------------------------------
    df = df[df['faster_model'] != 'None_Detected'].copy()
    
    df['mp_time_to_detect_s'] = df['mp_time_to_detect_s'].fillna(0)
    df['custom_time_to_detect_s'] = df['custom_time_to_detect_s'].fillna(0)
    df['mp_max_score'] = df['mp_max_score'].fillna(0)
    df['custom_max_score'] = df['custom_max_score'].fillna(0)
    
    labels = df['trial_id'].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.35
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 15))
    
    color_mp = '#4C72B0'
    color_cust = '#DD8452'
    
    ax1.bar(x - width/2, df['mp_time_to_detect_s'], width, label='MediaPipe', color=color_mp)
    ax1.bar(x + width/2, df['custom_time_to_detect_s'], width, label='Własny Model', color=color_cust)
    ax1.set_ylabel('Czas detekcji (s)')
    ax1.set_title('Czas potrzebny na poprawne wykrycie zadanego gestu (krótszy = lepiej)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax2.bar(x - width/2, df['mp_max_score'], width, label='MediaPipe', color=color_mp)
    ax2.bar(x + width/2, df['custom_max_score'], width, label='Własny Model', color=color_cust)
    ax2.set_ylabel('Max Pewność (Score)')
    ax2.set_title('Najwyższa zanotowana pewność w 5-sekundowym oknie (wyższy = lepiej)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.set_ylim(0, 1.1)
    
    ax3.bar(x - width/2, df['mp_false_activations'], width, label='MediaPipe', color=color_mp)
    ax3.bar(x + width/2, df['custom_false_activations'], width, label='Własny Model', color=color_cust)
    ax3.set_ylabel('Liczba błędnych aktywacji')
    ax3.set_title('Aktywacje niewłaściwych gestów (innych niż cel) w trakcie okna pomiarowego (mniej = lepiej)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_xlabel('Numer próby (Trial ID)')
    ax3.legend()
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    output_path = 'benchmarks/benchmark_plots.png'
    plt.savefig(output_path, dpi=150)
    print(f"Zapisano wykres do: {output_path}")
    plt.close()

if __name__ == '__main__':
    generate_benchmark_plots()
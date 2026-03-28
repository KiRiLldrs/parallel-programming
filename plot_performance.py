import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import csv

CPP_PROGRAM = "main.exe"
SIZES = [200, 500, 1000, 1500, 2000, 3000]
RESULTS_DIR = "performance_results"

def generate_matrix_file(filename, n):
    matrix = np.random.uniform(0, 10, (n, n))
    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        for row in matrix:
            f.write(" ".join(f"{val:.6f}" for val in row) + "\n")
    return os.path.getsize(filename) / 1024

def run_program(n):
    file_a = "temp_a.txt"
    file_b = "temp_b.txt"
    file_res = "temp_res.txt"
    
    generate_matrix_file(file_a, n)
    generate_matrix_file(file_b, n)
    
    start = time.time()
    result = subprocess.run(
        [CPP_PROGRAM, file_a, file_b, file_res],
        capture_output=True,
        text=True
    )
    end = time.time()
    
    for f in [file_a, file_b, file_res]:
        if os.path.exists(f):
            os.remove(f)
    
    if result.returncode != 0:
        print(f"Ошибка при N={n}: {result.stderr}")
        return None
    
    return end - start

def calculate_data_volume_kb(n):
    elements = n * n
    bytes_per_element = 8
    total_bytes = elements * bytes_per_element * 3
    return total_bytes / 1024

def save_plot(x_data, y_data, xlabel, filename, title='Время выполнения'):
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(x_data, y_data, 'o-', linewidth=2, markersize=8)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel('Время выполнения (сек.)', fontsize=12)
        plt.title(title, fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"График сохранен: {filename}")
        
        plt.close()
        return True
    except Exception as e:
        print(f"Ошибка сохранения графика: {e}")
        return False

def save_results_csv(sizes, volumes_kb, times, filepath):
    """Сохраняет результаты в CSV файл"""
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['N', 'Объем_КБ', 'Время_сек', 'GFLOPS'])
            for i, n in enumerate(sizes):
                flops = 2 * n**3
                gflops = flops / (times[i] * 1e9) if times[i] > 0 else 0
                writer.writerow([n, f"{volumes_kb[i]:.2f}", f"{times[i]:.6f}", f"{gflops:.2f}"])
        print(f"Таблица сохранена: {filepath}")
        return True
    except Exception as e:
        print(f"Ошибка сохранения CSV: {e}")
        return False

def main():
    print("Сбор данных о производительности...")
    print("=" * 50)
    
    sizes = []
    times = []
    volumes_kb = []
    
    for n in SIZES:
        print(f"Тестирование матрицы {n}×{n}...", end=" ")
        
        test_times = []
        for _ in range(3):
            t = run_program(n)
            if t is not None:
                test_times.append(t)
        
        if test_times:
            avg_time = min(test_times)
            sizes.append(n)
            times.append(avg_time)
            volumes_kb.append(calculate_data_volume_kb(n))
            print(f"{avg_time:.6f} сек")
        else:
            print("Ошибка")
    
    if not sizes:
        print("Нет данных для построения графика!")
        return
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("\nПостроение графиков...")
    
    # ===== ГРАФИК 1: Время от объема данных (КБ) =====
    save_plot(
        volumes_kb, 
        times, 
        'Объём данных (Кб)',
        f"{RESULTS_DIR}/time_vs_volume.png"
    )
    
    # ===== ГРАФИК 2: Время от размера матрицы =====
    save_plot(
        sizes, 
        times, 
        'Размер матрицы (N)',
        f"{RESULTS_DIR}/time_vs_size.png"
    )
    
    # ===== ТАБЛИЦА РЕЗУЛЬТАТОВ (консоль) =====
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ:")
    print("=" * 70)
    print(f"{'N':>6} | {'Объем (КБ)':>12} | {'Время (сек)':>12} | {'GFLOPS':>10}")
    print("-" * 70)
    
    for i, n in enumerate(sizes):
        flops = 2 * n**3
        gflops = flops / (times[i] * 1e9) if times[i] > 0 else 0
        print(f"{n:>6} | {volumes_kb[i]:>12.2f} | {times[i]:>12.6f} | {gflops:>10.2f}")
    
    print("=" * 70)
    
    # ===== СОХРАНЕНИЕ В CSV =====
    csv_path = f"{RESULTS_DIR}/results.csv"
    save_results_csv(sizes, volumes_kb, times, csv_path)
    
    print(f"\n✓ Графики и данные сохранены в папку: {RESULTS_DIR}/")
    
    print("\nПроверка файлов:")
    for f in os.listdir(RESULTS_DIR):
        filepath = os.path.join(RESULTS_DIR, f)
        size = os.path.getsize(filepath)
        print(f"  {f} - {size} байт")

if __name__ == "__main__":
    if not os.path.exists(CPP_PROGRAM):
        print(f"Ошибка: программа '{CPP_PROGRAM}' не найдена!")
        print("Скомпилируйте: g++ -O2 -o main.exe matrix_mult.cpp")
    else:
        try:
            main()
        except Exception as e:
            print(f"\nКритическая ошибка: {e}")
            import traceback
            traceback.print_exc()
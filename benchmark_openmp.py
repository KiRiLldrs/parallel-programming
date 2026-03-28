import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import csv

CPP_PROGRAM = "main.exe"
SIZES = [200, 400, 800, 1200]
THREADS = [1, 2, 4, 8]
RESULTS_DIR = "performance_results_openmp"

def generate_matrix_file(filename, n):
    """Генерация файла с матрицей"""
    matrix = np.random.uniform(0, 10, (n, n))
    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        for row in matrix:
            f.write(" ".join(f"{val:.6f}" for val in row) + "\n")
    return os.path.getsize(filename) / 1024

def run_program(n, num_threads=1):
    """Запуск программы с указанным количеством потоков"""
    file_a = "temp_a.txt"
    file_b = "temp_b.txt"
    file_res = "temp_res.txt"
    
    generate_matrix_file(file_a, n)
    generate_matrix_file(file_b, n)
    
    # Запускаем программу
    result = subprocess.run(
        [CPP_PROGRAM, str(num_threads), file_a, file_b, file_res],
        capture_output=True,
        text=True
    )
    
    # Удаляем временные файлы
    for f in [file_a, file_b, file_res]:
        if os.path.exists(f):
            os.remove(f)
    
    if result.returncode != 0:
        print(f"Ошибка при N={n}, threads={num_threads}: {result.stderr}")
        return None
    
    execution_time = None
    gflops = 0.0
    
    for line in result.stdout.split('\n'):
        if 'Execution Time (ms):' in line:
            time_ms = float(line.split(':')[-1].strip())
            execution_time = time_ms / 1000.0
        if 'GFLOPS' in line:
            try:
                gflops = float(line.split(':')[-1].strip().split()[0])
            except:
                pass
    
    if execution_time is None:
        print(f"Не удалось найти время выполнения в выводе программы")
        return None
    
    return execution_time, gflops

def calculate_speedup(times_dict, base_threads=1):
    """Расчёт ускорения относительно 1 потока"""
    speedup = {}
    base_time = times_dict.get(base_threads, 0)
    if base_time > 0:
        for threads, time_val in times_dict.items():
            speedup[threads] = base_time / time_val
    return speedup

def calculate_efficiency(speedup, num_threads):
    """Расчёт эффективности распараллеливания (%)"""
    return (speedup / num_threads) * 100

def save_plot_time_vs_size(all_times, filename):
    """График: Время выполнения от размера матрицы для разного кол-ва потоков"""
    plt.figure(figsize=(12, 7))
    
    colors = {'1': 'green', '2': 'blue', '4': 'red', '8': 'orange'}
    markers = {'1': 's', '2': 's', '4': 's', '8': 's'}
    
    for threads in THREADS:
        sizes = list(all_times[threads].keys())
        times = list(all_times[threads].values())
        plt.plot(sizes, times, 'o-', linewidth=2, markersize=8, 
                color=colors.get(str(threads), 'gray'),
                marker=markers.get(str(threads), 'o'),
                label=f'{threads} поток(ов)')
    
    plt.xlabel('Размер матрицы (n)', fontsize=12)
    plt.ylabel('Время выполнения (сек)', fontsize=12)
    plt.title('Зависимость времени выполнения\nот размера матрицы', fontsize=14, pad=20)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"График сохранен: {filename}")
    plt.close()

def save_plot_speedup(all_times, filename):
    """График: Ускорение от количества потоков"""
    plt.figure(figsize=(12, 7))
    
    # Расчёт ускорения для каждого размера матрицы
    for size in SIZES:
        times_for_size = {threads: all_times[threads].get(size, 0) for threads in THREADS}
        speedup = calculate_speedup(times_for_size)
        
        threads_list = list(speedup.keys())
        speedup_list = list(speedup.values())
        
        plt.plot(threads_list, speedup_list, 'o-', linewidth=2, markersize=8, 
                label=f'{size}×{size}')
    
    ideal_speedup = [1, 2, 4, 8]
    plt.plot(THREADS, ideal_speedup, '--', color='gray', linewidth=2, label='Идеальное ускорение')
    
    plt.xlabel('Количество потоков', fontsize=12)
    plt.ylabel('Ускорение (Speedup)', fontsize=12)
    plt.title('Эффективность распараллеливания', fontsize=14, pad=20)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(THREADS)
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"График сохранен: {filename}")
    plt.close()

def save_plot_efficiency(all_times, filename):
    """График: Эффективность распараллеливания (%)"""
    plt.figure(figsize=(12, 7))
    
    for size in SIZES:
        times_for_size = {threads: all_times[threads].get(size, 0) for threads in THREADS}
        speedup = calculate_speedup(times_for_size)
        
        threads_list = list(speedup.keys())[1:]
        efficiency_list = [calculate_efficiency(speedup[t], t) for t in threads_list]
        
        plt.plot(threads_list, efficiency_list, 'o-', linewidth=2, markersize=8, 
                label=f'{size}×{size}')
    
    plt.axhline(y=100, color='gray', linestyle='--', linewidth=2, label='100% эффективность')
    
    plt.xlabel('Количество потоков', fontsize=12)
    plt.ylabel('Эффективность (%)', fontsize=12)
    plt.title('Эффективность использования потоков', fontsize=14, pad=20)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(THREADS[1:])
    plt.ylim(0, 120)
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"График сохранен: {filename}")
    plt.close()

def save_comprehensive_csv(all_times, all_gflops, filename):
    """Сохранение всех результатов в CSV"""
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['Размер_матрицы', 'Потоки', 'Время_сек', 'GFLOPS', 'Ускорение', 'Эффективность_%'])
            
            for size in SIZES:
                base_time = all_times[1].get(size, 0)
                
                for threads in THREADS:
                    time_val = all_times[threads].get(size, 0)
                    gflops_val = all_gflops[threads].get(size, 0)
                    
                    speedup = base_time / time_val if (base_time > 0 and time_val > 0) else 1.0
                    efficiency = (speedup / threads) * 100 if threads > 0 else 0
                    
                    writer.writerow([
                        size,
                        threads,
                        f"{time_val:.6f}",
                        f"{gflops_val:.2f}",
                        f"{speedup:.2f}",
                        f"{efficiency:.1f}"
                    ])
        
        print(f"Таблица сохранена: {filename}")
        return True
    except Exception as e:
        print(f"Ошибка сохранения CSV: {e}")
        return False

def print_summary_table(all_times, all_gflops):
    """Вывод сводной таблицы в консоль"""
    print("\n" + "=" * 100)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 100)
    print(f"{'Размер':>8} | {'Потоки':>8} | {'Время (сек)':>12} | {'GFLOPS':>10} | {'Ускорение':>10} | {'Эффективность':>12}")
    print("-" * 100)
    
    for size in SIZES:
        base_time = all_times[1].get(size, 0)
        
        for threads in THREADS:
            time_val = all_times[threads].get(size, 0)
            gflops_val = all_gflops[threads].get(size, 0)
            
            speedup = base_time / time_val if (base_time > 0 and time_val > 0) else 1.0
            efficiency = (speedup / threads) * 100 if threads > 0 else 0
            
            print(f"{size:>8} | {threads:>8} | {time_val:>12.6f} | {gflops_val:>10.2f} | {speedup:>10.2f} | {efficiency:>12.1f}%")
        
        print("-" * 100)

def main():
    print("Бенчмарк OpenMP: Тестирование с разным количеством потоков")
    print("=" * 70)
    
    all_times = {threads: {} for threads in THREADS}
    all_gflops = {threads: {} for threads in THREADS}
    
    for threads in THREADS:
        print(f"\n{'='*70}")
        print(f"ТЕСТИРОВАНИЕ С {threads} ПОТОК(ОВ)")
        print('='*70)
        
        for n in SIZES:
            print(f"Матрица {n}×{n}...", end=" ")
            
            test_runs = []
            for run in range(3):
                result = run_program(n, threads)
                if result is not None:
                    exec_time, gflops = result
                    test_runs.append((exec_time, gflops))
            
            if test_runs:
                best_run = min(test_runs, key=lambda x: x[0])
                avg_time, avg_gflops = best_run
                
                all_times[threads][n] = avg_time
                all_gflops[threads][n] = avg_gflops
                
                print(f"{avg_time:.6f} сек, {avg_gflops:.2f} GFLOPS")
            else:
                print("Ошибка")
    
    if not any(all_times[t] for t in THREADS):
        print("Нет данных для построения графиков!")
        return
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("ПОСТРОЕНИЕ ГРАФИКОВ")
    print("=" * 70)
    
    # ===== ГРАФИК 1: Время от размера матрицы =====
    save_plot_time_vs_size(
        all_times,
        f"{RESULTS_DIR}/time_vs_size_openmp.png"
    )
    
    # ===== ГРАФИК 2: Ускорение (Speedup) =====
    save_plot_speedup(
        all_times,
        f"{RESULTS_DIR}/speedup_vs_threads.png"
    )
    
    # ===== ГРАФИК 3: Эффективность =====
    save_plot_efficiency(
        all_times,
        f"{RESULTS_DIR}/efficiency_vs_threads.png"
    )
    
    # ===== ГРАФИК 4: GFLOPS от количества потоков =====
    plt.figure(figsize=(12, 7))
    for size in SIZES:
        gflops_for_size = [all_gflops[threads].get(size, 0) for threads in THREADS]
        plt.plot(THREADS, gflops_for_size, 'o-', linewidth=2, markersize=8, label=f'{size}×{size}')
    
    plt.xlabel('Количество потоков', fontsize=12)
    plt.ylabel('Производительность (GFLOPS)', fontsize=12)
    plt.title('Производительность в зависимости от количества потоков', fontsize=14, pad=20)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(THREADS)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/gflops_vs_threads.png", dpi=300, bbox_inches='tight')
    print(f"График сохранен: {RESULTS_DIR}/gflops_vs_threads.png")
    plt.close()
    
    # ===== ТАБЛИЦА И CSV =====
    print("\n" + "=" * 70)
    print_summary_table(all_times, all_gflops)
    
    csv_path = f"{RESULTS_DIR}/results_openmp.csv"
    save_comprehensive_csv(all_times, all_gflops, csv_path)
    
    print(f"\n✓ Все графики и данные сохранены в папку: {RESULTS_DIR}/")
    
    print("\nПроверка файлов:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        filepath = os.path.join(RESULTS_DIR, f)
        size = os.path.getsize(filepath)
        print(f"  {f} - {size} байт")

if __name__ == "__main__":
    if not os.path.exists(CPP_PROGRAM):
        print(f"Ошибка: программа '{CPP_PROGRAM}' не найдена!")
        print("Скомпилируйте: g++ -fopenmp -O2 -o main.exe main.cpp")
    else:
        try:
            main()
        except Exception as e:
            print(f"\nКритическая ошибка: {e}")
            import traceback
            traceback.print_exc()
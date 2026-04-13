import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import csv

MPI_PROGRAM = "main_mpi.exe"
MPI_EXEC = "C:\\Program Files\\Microsoft MPI\\Bin\\mpiexec.exe"

# Размеры матриц для strong scaling
SIZES = [200, 400, 800, 1200, 1600, 2000]
PROCS = [1, 2, 4, 8]

WEAK_SCALING = [
    (1, 500),
    (2, 700),
    (4, 1000),
    (8, 1400)
]

RESULTS_DIR = "mpi_experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)


def generate_matrix_file(filename, n):
    """Генерация случайной матрицы"""
    matrix = np.random.uniform(0, 10, (n, n))
    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        for row in matrix:
            f.write(" ".join(f"{val:.6f}" for val in row) + "\n")
    return os.path.getsize(filename) / 1024

def run_mpi_test(n, num_procs):
    """Запуск одного теста MPI"""
    file_a = f"temp_a_{n}_{num_procs}.txt"
    file_b = f"temp_b_{n}_{num_procs}.txt"
    file_res = f"temp_res_{n}_{num_procs}.txt"
    
    generate_matrix_file(file_a, n)
    generate_matrix_file(file_b, n)
    
    cmd = [
        MPI_EXEC,
        "-n", str(num_procs),
        MPI_PROGRAM,
        file_a,
        file_b,
        file_res
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: N={n}, P={num_procs}")
        return None
    
    for f in [file_a, file_b, file_res]:
        if os.path.exists(f):
            os.remove(f)
    
    if result.returncode != 0:
        return None
    
    exec_time = None
    gflops = 0.0
    
    for line in result.stdout.split('\n'):
        if 'Execution Time (s):' in line:
            try:
                exec_time = float(line.split(':')[-1].strip())
            except:
                pass
        if 'Performance (GFLOPS):' in line:
            try:
                gflops = float(line.split(':')[-1].strip().split()[0])
            except:
                pass
    
    if exec_time is None:
        return None
    
    return exec_time, gflops

def calculate_metrics(all_times):
    """Расчёт speedup и efficiency"""
    metrics = {}
    
    for size in SIZES:
        metrics[size] = {}
        base_time = all_times[1].get(size, 0)
        
        for procs in PROCS:
            time_val = all_times[procs].get(size, 0)
            if time_val > 0 and base_time > 0:
                speedup = base_time / time_val
                efficiency = (speedup / procs) * 100
            else:
                speedup = 1.0 if procs == 1 else 0
                efficiency = 100.0 if procs == 1 else 0
            
            metrics[size][procs] = {
                'time': time_val,
                'speedup': speedup,
                'efficiency': efficiency
            }
    
    return metrics

def plot_time_vs_procs(all_times, metrics):
    """График 1: Время выполнения от количества процессов"""
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(SIZES)))
    
    for idx, size in enumerate(SIZES):
        times = [all_times[p].get(size, 0) for p in PROCS]
        plt.plot(PROCS, times, 'o-', linewidth=2, markersize=8, 
                color=colors[idx], label=f'N={size}')
    
    plt.xlabel('Количество процессов (P)', fontsize=12)
    plt.ylabel('Время (сек)', fontsize=12)
    plt.title('Зависимость времени выполнения от количества процессов (MPI)', 
              fontsize=14, pad=15)
    plt.legend(title='Размер матрицы', loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xticks(PROCS)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/01_time_vs_procs.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ График 1: Время vs Процессы")

def plot_speedup(metrics):
    """График 2: Ускорение (Speedup)"""
    plt.figure(figsize=(10, 6))
    
    plt.plot(PROCS, PROCS, '--', color='gray', linewidth=2, 
             label='Идеальное ускорение (S=P)', alpha=0.7)
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(SIZES)))
    
    for idx, size in enumerate(SIZES):
        speedups = [metrics[size][p]['speedup'] for p in PROCS]
        plt.plot(PROCS, speedups, 'o-', linewidth=2, markersize=8,
                color=colors[idx], label=f'N={size}')
    
    plt.xlabel('Количество процессов (P)', fontsize=12)
    plt.ylabel('Ускорение (раз)', fontsize=12)
    plt.title('Ускорение вычислений (Speedup) относительно 1 процесса', 
              fontsize=14, pad=15)
    plt.legend(title='Размер матрицы', loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(PROCS)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/02_speedup.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ График 2: Speedup")

def plot_efficiency(metrics):
    """График 3: Эффективность (столбчатая диаграмма)"""
    x = np.arange(len(PROCS))
    width = 0.12
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(SIZES)))
    
    for idx, size in enumerate(SIZES):
        efficiency = [metrics[size][p]['efficiency'] for p in PROCS]
        ax.bar(x + idx * width, efficiency, width, label=f'N={size}',
               color=colors[idx], alpha=0.8)
    
    ax.axhline(y=100, color='red', linestyle='--', linewidth=2, 
               label='100% (идеал)', alpha=0.7)
    
    ax.set_xlabel('Количество процессов (P)', fontsize=12)
    ax.set_ylabel('Эффективность (%)', fontsize=12)
    ax.set_title('Эффективность использования процессов (Efficiency)', 
                 fontsize=14, pad=15)
    ax.set_xticks(x + width * (len(SIZES) - 1) / 2)
    ax.set_xticklabels(PROCS)
    ax.legend(title='Размер матрицы', loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 120)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/03_efficiency.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ График 3: Efficiency")

def plot_weak_scaling(weak_results):
    """График 4: Weak scaling"""
    if not weak_results:
        return
    
    plt.figure(figsize=(10, 6))
    
    procs = [r[0] for r in WEAK_SCALING]
    sizes = [r[1] for r in WEAK_SCALING]
    
    times = [weak_results.get((p, s), 0) for p, s in WEAK_SCALING]
    
    base_time = times[0] if times[0] > 0 else 1
    normalized = [t / base_time if t > 0 else 0 for t in times]
    
    plt.plot(procs, normalized, 's-', linewidth=2, markersize=10, 
             color='green', label='Weak Scaling')
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2, 
                label='Идеальное (1.0)', alpha=0.7)
    
    plt.xlabel('Количество процессов', fontsize=12)
    plt.ylabel('Нормализованное время', fontsize=12)
    plt.title('Слабая масштабируемость (Weak Scaling)\n(размер задачи растёт пропорционально P)', 
              fontsize=14, pad=15)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xticks(procs)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/04_weak_scaling.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ График 4: Weak Scaling")

def plot_gflops(all_gflops):
    """График 5: Производительность (GFLOPS)"""
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(SIZES)))
    
    for idx, size in enumerate(SIZES):
        gflops = [all_gflops[p].get(size, 0) for p in PROCS]
        plt.plot(PROCS, gflops, 'o-', linewidth=2, markersize=8,
                color=colors[idx], label=f'N={size}')
    
    plt.xlabel('Количество процессов (P)', fontsize=12)
    plt.ylabel('Производительность (GFLOPS)', fontsize=12)
    plt.title('Производительность MPI программы', fontsize=14, pad=15)
    plt.legend(title='Размер матрицы', loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(PROCS)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/05_gflops.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ График 5: GFLOPS")

def save_results_csv(all_times, all_gflops, metrics, filename):
    """Сохранение всех результатов в CSV"""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['Размер', 'Процессы', 'Время (сек)', 'GFLOPS', 
                        'Ускорение', 'Эффективность (%)', 'Операций', 'Статус'])
        
        for size in SIZES:
            flops = 2 * size**3
            flops_str = f"{flops/1e6:.0f} млн" if flops < 1e9 else f"{flops/1e9:.2f} млрд"
            
            for procs in PROCS:
                time_val = metrics[size][procs]['time']
                gflops_val = all_gflops[procs].get(size, 0)
                speedup = metrics[size][procs]['speedup']
                efficiency = metrics[size][procs]['efficiency']
                
                status = "✓" if time_val > 0 else "✗"
                
                writer.writerow([
                    size, procs, f"{time_val:.4f}", f"{gflops_val:.2f}",
                    f"{speedup:.2f}", f"{efficiency:.1f}", flops_str, status
                ])
    
    print(f"✓ CSV сохранён: {filename}")

def print_summary_table(metrics):
    """Красивая таблица в консоль"""
    print("\n" + "="*100)
    print(" " * 35 + "РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ")
    print("="*100)
    print(f"{'Размер':>8} | {'Процессы':>10} | {'Время (сек)':>12} | {'Ускорение':>10} | {'Эффективность':>12} | {'Операций':>10} | {'Статус':>6}")
    print("-"*100)
    
    for size in SIZES:
        flops = 2 * size**3
        flops_str = f"{flops/1e6:.0f} млн" if flops < 1e9 else f"{flops/1e9:.2f} млрд"
        
        for procs in PROCS:
            time_val = metrics[size][procs]['time']
            speedup = metrics[size][procs]['speedup']
            efficiency = metrics[size][procs]['efficiency']
            
            status = "✓" if time_val > 0 else "✗"
            
            print(f"{size:>8} | {procs:>10} | {time_val:>12.4f} | {speedup:>10.2f} | {efficiency:>12.1f}% | {flops_str:>10} | {status:>6}")
        
        print("-"*100)

def main():
    print("="*70)
    print(" " * 20 + "MPI БЕНЧМАРК")
    print("="*70)
    
    all_times = {p: {} for p in PROCS}
    all_gflops = {p: {} for p in PROCS}
    weak_results = {}
    
    print("\n[1/2] STRONG SCALING (фиксированный размер задачи)")
    print("-"*70)
    
    for procs in PROCS:
        print(f"\nЗапуск с {procs} процесс(ов)...")
        
        for size in SIZES:
            print(f"  N={size}...", end=" ", flush=True)
            
            best_time = float('inf')
            best_gflops = 0
            
            for attempt in range(3):
                result = run_mpi_test(size, procs)
                if result:
                    exec_time, gflops = result
                    if exec_time < best_time:
                        best_time = exec_time
                        best_gflops = gflops
            
            if best_time < float('inf'):
                all_times[procs][size] = best_time
                all_gflops[procs][size] = best_gflops
                print(f"{best_time:.4f} сек ({best_gflops:.1f} GFLOPS)")
            else:
                print("ОШИБКА")
                all_times[procs][size] = 0
                all_gflops[procs][size] = 0
    
    print("\n[2/2] WEAK SCALING (растущий размер задачи)")
    print("-"*70)
    
    for procs, size in WEAK_SCALING:
        print(f"P={procs}, N={size}...", end=" ", flush=True)
        
        result = run_mpi_test(size, procs)
        if result:
            exec_time, gflops = result
            weak_results[(procs, size)] = exec_time
            print(f"{exec_time:.4f} сек ({gflops:.1f} GFLOPS)")
        else:
            print("ОШИБКА")
            weak_results[(procs, size)] = 0
    
    print("\n" + "="*70)
    print("ОБРАБОТКА РЕЗУЛЬТАТОВ")
    print("="*70)
    
    metrics = calculate_metrics(all_times)
    
    print("\nПОСТРОЕНИЕ ГРАФИКОВ:")
    print("-"*70)
    
    plot_time_vs_procs(all_times, metrics)
    plot_speedup(metrics)
    plot_efficiency(metrics)
    plot_weak_scaling(weak_results)
    plot_gflops(all_gflops)
    
    print_summary_table(metrics)
    
    save_results_csv(all_times, all_gflops, metrics, 
                    f"{RESULTS_DIR}/results_mpi.csv")
    
    print("\n" + "="*70)
    print("ЭКСПЕРИМЕНТ ЗАВЕРШЁН")
    print("="*70)
    print(f"\nРезультаты сохранены в папку: {RESULTS_DIR}/")
    print("\nФайлы:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        size = os.path.getsize(f"{RESULTS_DIR}/{f}")
        print(f"  {f:30} {size:>10} байт")

if __name__ == "__main__":
    if not os.path.exists(MPI_PROGRAM):
        print(f"ОШИБКА: {MPI_PROGRAM} не найден!")
        print("Скомпилируйте программу сначала.")
    elif not os.path.exists(MPI_EXEC):
        print(f"ОШИБКА: mpiexec не найден: {MPI_EXEC}")
    else:
        try:
            main()
        except Exception as e:
            print(f"\nКРИТИЧЕСКАЯ ОШИБКА: {e}")
            import traceback
            traceback.print_exc()
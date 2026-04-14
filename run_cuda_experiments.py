import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import csv
from datetime import datetime
import platform

CUDA_PROGRAM = "main_cuda.exe" if platform.system() == "Windows" else "./main_cuda"
RESULTS_DIR = "cuda_experiments_results"

MATRIX_SIZES = [256, 512, 768, 1024, 1536, 2048]
BLOCK_SIZES = [16, 32]
NUM_RUNS = 3

def generate_matrix_file(filename, n):
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
    matrix = np.random.uniform(1, 10, (n, n)).astype(np.float64)
    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        for row in matrix:
            f.write(" ".join(f"{val:.6f}" for val in row) + "\n")
    return os.path.getsize(filename) / 1024

def calculate_gflops(n, time_sec):
    if time_sec <= 0:
        return 0.0
    flops = 2 * n**3
    return flops / (time_sec * 1e9)

def calculate_data_volume_kb(n):
    elements = n * n
    bytes_per_element = 8
    total_bytes = elements * bytes_per_element * 3
    return total_bytes / 1024

def run_cuda_program(n, block_size=32, timeout=300):
    file_a = f"temp_a_{n}.txt"
    file_b = f"temp_b_{n}.txt"
    file_res = f"temp_res_{n}_{block_size}.txt"
    generate_matrix_file(file_a, n)
    generate_matrix_file(file_b, n)
    cmd = [CUDA_PROGRAM, file_a, file_b, file_res, str(block_size)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        for f in [file_a, file_b, file_res]:
            if os.path.exists(f):
                os.remove(f)
        if result.returncode != 0:
            print(f"Error: {result.stderr[:200]}")
            return None, None
        execution_time_sec = None
        gflops = None
        for line in result.stdout.split('\n'):
            line = line.strip()
            if 'Execution Time:' in line or 'Time (ms):' in line:
                try:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        time_str = parts[-1].strip().replace('ms', '').strip()
                        time_ms = float(time_str)
                        execution_time_sec = time_ms / 1000.0
                except (ValueError, IndexError):
                    pass
            if 'Performance:' in line or 'GFLOPS' in line:
                try:
                    parts = line.split()
                    for part in parts:
                        try:
                            val = float(part)
                            if 0.1 < val < 100000:
                                gflops = val
                                break
                        except ValueError:
                            continue
                except:
                    pass
        if execution_time_sec is None:
            return execution_time_sec, gflops
        if gflops is None:
            gflops = calculate_gflops(n, execution_time_sec)
        return execution_time_sec, gflops
    except subprocess.TimeoutExpired:
        print(f"Timeout ({timeout}s)")
        return None, None
    except Exception as e:
        print(f"Exception: {e}")
        return None, None

def save_plot(x_data, y_data, xlabel, ylabel, title, filename, color='blue', marker='o'):
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'o-', linewidth=2, markersize=8, color=color, marker=marker)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {filename}")

def save_comparison_plot(sizes, times_dict, filename):
    plt.figure(figsize=(12, 7))
    colors = {'16': 'red', '32': 'blue'}
    markers = {'16': 's', '32': 'o'}
    for config, times in times_dict.items():
        if times:
            label = f"Block {config}x{config}"
            plt.plot(sizes, times, 'o-', linewidth=2, markersize=8, 
                    color=colors.get(config, 'gray'), 
                    marker=markers.get(config, 'o'),
                    label=label)
    plt.xlabel('Matrix Size N', fontsize=12)
    plt.ylabel('Execution Time (sec)', fontsize=12)
    plt.title('CUDA Performance: Different Block Sizes', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved: {filename}")

def save_gflops_plot(sizes, gflops_dict, filename):
    plt.figure(figsize=(12, 7))
    colors = {'16': 'red', '32': 'blue'}
    for config, gflops in gflops_dict.items():
        if gflops:
            label = f"Block {config}x{config}"
            plt.plot(sizes, gflops, 's-', linewidth=2, markersize=8, 
                    color=colors.get(config, 'gray'), label=label)
    plt.xlabel('Matrix Size N', fontsize=12)
    plt.ylabel('Performance (GFLOPS)', fontsize=12)
    plt.title('CUDA Matrix Multiplication Performance', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"GFLOPS plot saved: {filename}")

def save_results_csv(all_results, filepath):
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['N', 'Block_Size', 'Time_sec', 'GFLOPS', 'Data_Volume_KB'])
        for r in all_results:
            writer.writerow([
                r['n'],
                r['block_size'],
                f"{r['time']:.6f}" if r['time'] else 'N/A',
                f"{r['gflops']:.2f}" if r['gflops'] else 'N/A',
                f"{r['volume_kb']:.2f}"
            ])
    print(f"Results saved: {filepath}")

def save_summary_table(all_results, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CUDA MATRIX MULTIPLICATION EXPERIMENT RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Platform: {platform.platform()}\n")
        f.write(f"Matrix sizes: {MATRIX_SIZES}\n")
        f.write(f"Block sizes: {BLOCK_SIZES}\n\n")
        f.write("DETAILED RESULTS:\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'N':>6} | {'Block':>5} | {'Time(s)':>12} | {'GFLOPS':>10} | {'Vol(KB)':>10}\n")
        f.write("-" * 80 + "\n")
        for r in all_results:
            if r['time'] and r['time'] > 0:
                f.write(f"{r['n']:>6} | {r['block_size']:>5} | "
                       f"{r['time']:>12.6f} | {r['gflops']:>10.2f} | {r['volume_kb']:>10.2f}\n")
        f.write("=" * 80 + "\n\n")
        f.write("BEST RESULTS (by GFLOPS):\n")
        f.write("-" * 50 + "\n")
        for n in MATRIX_SIZES:
            results_for_n = [r for r in all_results if r['n'] == n and r['time'] and r['time'] > 0]
            if results_for_n:
                best = max(results_for_n, key=lambda x: x['gflops'])
                f.write(f"N={n}: {best['gflops']:.2f} GFLOPS (block={best['block_size']}x{best['block_size']})\n")
    print(f"Summary table saved: {filepath}")

def main():
    print("\n" + "=" * 70)
    print("CUDA MATRIX MULTIPLICATION EXPERIMENTS")
    print("=" * 70 + "\n")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(CUDA_PROGRAM):
        print(f"Error: {CUDA_PROGRAM} not found!")
        print("Compile: nvcc -arch=sm_86 -o main_cuda.exe main_cuda.cu")
        return
    all_results = []
    times_dict = {}
    gflops_dict = {}
    print("RUNNING EXPERIMENTS:")
    print("-" * 70)
    for n in MATRIX_SIZES:
        print(f"\nTesting matrix {n}x{n}...")
        for block_size in BLOCK_SIZES:
            config_key = str(block_size)
            print(f"  Block {block_size}x{block_size}...", end=" ")
            test_times = []
            test_gflops = []
            for run in range(NUM_RUNS):
                t, g = run_cuda_program(n, block_size)
                if t is not None:
                    test_times.append(t)
                    if g is not None:
                        test_gflops.append(g)
            if test_times:
                avg_time = min(test_times)
                avg_gflops = max(test_gflops) if test_gflops else calculate_gflops(n, avg_time)
                volume_kb = calculate_data_volume_kb(n)
                all_results.append({
                    'n': n,
                    'block_size': block_size,
                    'time': avg_time,
                    'gflops': avg_gflops,
                    'volume_kb': volume_kb
                })
                if config_key not in times_dict:
                    times_dict[config_key] = []
                    gflops_dict[config_key] = []
                times_dict[config_key].append(avg_time)
                gflops_dict[config_key].append(avg_gflops)
                print(f"{avg_time:.4f}s ({avg_gflops:.1f} GFLOPS)")
            else:
                print("Error")
    if not all_results:
        print("\nNo data for analysis!")
        return
    print("\n" + "=" * 70)
    print("SAVING RESULTS:")
    print("-" * 70)
    sizes_tested = [r['n'] for r in all_results if r['time'] and r['time'] > 0]
    sizes_tested = sorted(list(set(sizes_tested)))
    save_comparison_plot(sizes_tested, times_dict, f"{RESULTS_DIR}/time_comparison.png")
    save_gflops_plot(sizes_tested, gflops_dict, f"{RESULTS_DIR}/gflops_comparison.png")
    for config_key, times in times_dict.items():
        if times:
            volumes = [calculate_data_volume_kb(n) for n in sizes_tested]
            title = f"Block {config_key}x{config_key}"
            save_plot(volumes, times, 'Data Volume (KB)', 'Time (sec)', 
                     title, f"{RESULTS_DIR}/time_vs_volume_block{config_key}.png")
    csv_path = f"{RESULTS_DIR}/results.csv"
    save_results_csv(all_results, csv_path)
    summary_path = f"{RESULTS_DIR}/summary.txt"
    save_summary_table(all_results, summary_path)
    print("\n" + "=" * 70)
    print("FINAL TABLE:")
    print("=" * 70)
    print(f"{'N':>6} | {'Block':>5} | {'Time(s)':>12} | {'GFLOPS':>10}")
    print("-" * 70)
    for r in all_results:
        if r['time'] and r['time'] > 0:
            print(f"{r['n']:>6} | {r['block_size']:>5} | "
                  f"{r['time']:>12.6f} | {r['gflops']:>10.2f}")
    print("=" * 70)
    print(f"\nAll results saved to: {RESULTS_DIR}/")
    print(f"   - Graphs: time_comparison.png, gflops_comparison.png")
    print(f"   - Tables: results.csv, summary.txt")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nCritical error: {e}")
        import traceback
        traceback.print_exc()
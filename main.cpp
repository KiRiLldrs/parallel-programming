#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <string>

bool readMatrix(const std::string &filename, std::vector<double> &matrix, int &n)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error! Couldn't open the file " << filename << std::endl;
        return false;
    }

    file >> n;
    matrix.resize(n * n);

    for (int i = 0; i < n * n; ++i)
    {
        if (!(file >> matrix[i]))
        {
            std::cerr << "Error! Not enough data in the file " << filename << std::endl;
            return false;
        }
    }
    file.close();
    return true;
}

bool writeMatrix(const std::string &filename, std::vector<double> &matrix, int &n)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error! Couldn't open the file " << filename << std::endl;
        return false;
    }

    file << n << std::endl;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            file << matrix[i * n + j] << " ";
        }
        file << std::endl;
    }
    file.close();
    return true;
}

void multiplyMatrices(const std::vector<double> &A, const std::vector<double> &B, std::vector<double> &C, int n)
{
    C.assign(n * n, 0.0);

    for (int i = 0; i < n; ++i)
    {
        for (int k = 0; k < n; ++k)
        {
            double a_ik = A[i * n + k];
            for (int j = 0; j < n; ++j)
            {
                C[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }
}

int main(int argc, char *argv[])
{
    std::string file_a = "data/matrix_a.txt";
    std::string file_b = "data/matrix_b.txt";
    std::string result_file = "data/matrix_res.txt";

    if (argc >= 4)
    {
        file_a = argv[1];
        file_b = argv[2];
        result_file = argv[3];
    }

    std::vector<double> A, B, C;
    int nA, nB;

    std::cout << "Reading matrix A from file " << file_a << "..." << std::endl;
    if (!readMatrix(file_a, A, nA))
    {
        return 1;
    }

    std::cout << "Reading matrix B from file " << file_b << "..." << std::endl;
    if (!readMatrix(file_b, B, nB))
    {
        return 1;
    }

    if (nA != nB)
    {
        std::cerr << "Error! Matrices should have the save size!" << std::endl;
        std::cerr << "Matrix A: " << nA << "x" << nA << std::endl;
        std::cerr << "Matrix B: " << nB << "x" << nB << std::endl;
        return 1;
    }

    int n = nA;

    // === Замер времени выполнения ===
    auto start = std::chrono::high_resolution_clock::now();
    multiplyMatrices(A, B, C, n);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double time_ms = duration.count() / 1000.0;

    // === Оценка объема вычислений ===
    // Количество операций с плавающей точкой (FLOPs) ≈ 2 * N^3

    long long flops = 2LL * n * n * n;
    double gflops = flops / (time_ms * 1000000.0); // gflops = 10^9 flops

    std::cout << "Writing result to file " << result_file << "..." << std::endl;
    if (!writeMatrix(result_file, C, n))
    {
        return 1;
    }

    std::cout << "Starting verification..." << std::endl;
    system(("python checkMultiply.py " + file_a + " " + file_b + " " + result_file).c_str());

    std::cout << "\n========== REPORT ==========" << std::endl;
    std::cout << "Matrix Size: " << n << "x" << n << std::endl;
    std::cout << "Execution Time (ms): " << time_ms << std::endl;
    std::cout << "Estimated FLOPs: " << flops << std::endl;
    std::cout << "Performance (GFLOPS): " << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;

    std::cout << "============================" << std::endl;

    std::ofstream report("report.txt");
    if (report.is_open())
    {
        report << "Matrix Size: " << n << "x" << n << std::endl;
        report << "Execution Time (ms): " << time_ms << std::endl;
        report << "Estimated FLOPs: " << flops << std::endl;
        report << "Performance (GFLOPS): " << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;

        std::cout << "Report is saved to file report.txt" << std::endl;
    }
}
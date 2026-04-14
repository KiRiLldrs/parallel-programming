#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

__global__ void matrixMulKernel(const double *A, const double *B, double *C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        double sum = 0.0;
        for (int k = 0; k < n; ++k)
        {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

__global__ void matrixMulSharedKernel(const double *A, const double *B, double *C, int n)
{
    __shared__ double As[32][32];
    __shared__ double Bs[32][32];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * 32 + ty;
    int col = bx * 32 + tx;

    double sum = 0.0;
    int tiles = (n + 31) / 32;

    for (int t = 0; t < tiles; ++t)
    {
        if (row < n && t * 32 + tx < n)
            As[ty][tx] = A[row * n + t * 32 + tx];
        else
            As[ty][tx] = 0.0;

        if (t * 32 + ty < n && col < n)
            Bs[ty][tx] = B[(t * 32 + ty) * n + col];
        else
            Bs[ty][tx] = 0.0;

        __syncthreads();

        for (int k = 0; k < 32; ++k)
        {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < n && col < n)
    {
        C[row * n + col] = sum;
    }
}

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
            file << std::fixed << std::setprecision(6) << matrix[i * n + j] << " ";
        }
        file << std::endl;
    }
    return true;
}

void multiplyMatricesCUDA(const std::vector<double> &A, const std::vector<double> &B,
                          std::vector<double> &C, int n,
                          int blockSize = 32, bool useShared = true)
{
    double *d_A, *d_B, *d_C;
    size_t bytes = n * n * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice));

    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim((n + blockSize - 1) / blockSize, (n + blockSize - 1) / blockSize);

    if (useShared && blockSize == 32)
    {
        matrixMulSharedKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
    }
    else
    {
        matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    C.resize(n * n);
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

void printGPUInfo()
{
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    std::cout << "\n=== GPU Information ===" << std::endl;
    for (int i = 0; i < deviceCount; ++i)
    {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max Threads/Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    }
    std::cout << "======================\n"
              << std::endl;
}

int main(int argc, char *argv[])
{
    std::string file_a = "data/matrix_a.txt";
    std::string file_b = "data/matrix_b.txt";
    std::string result_file = "data/matrix_res.txt";
    int blockSize = 32;
    bool useShared = true;

    if (argc >= 4)
    {
        file_a = argv[1];
        file_b = argv[2];
        result_file = argv[3];
    }
    if (argc >= 5)
    {
        blockSize = std::atoi(argv[4]);
        if (blockSize != 16 && blockSize != 32)
        {
            std::cerr << "Warning: blockSize should be 16 or 32 for optimal performance. Using " << blockSize << std::endl;
        }
    }
    if (argc >= 6)
    {
        useShared = (std::string(argv[5]) != "no_shared");
    }

    printGPUInfo();

    std::vector<double> A, B, C;
    int nA, nB;

    std::cout << "Reading matrix A from " << file_a << "..." << std::endl;
    if (!readMatrix(file_a, A, nA))
        return 1;

    std::cout << "Reading matrix B from " << file_b << "..." << std::endl;
    if (!readMatrix(file_b, B, nB))
        return 1;

    if (nA != nB)
    {
        std::cerr << "Error! Matrices should have the same size!" << std::endl;
        return 1;
    }
    int n = nA;

    std::cout << "Matrix size: " << n << "x" << n << std::endl;
    std::cout << "CUDA config: block=" << blockSize << "x" << blockSize
              << ", shared_mem=" << (useShared ? "ON" : "OFF") << std::endl;

    if (n <= 500)
    {
        std::vector<double> C_warmup;
        multiplyMatricesCUDA(A, B, C_warmup, n, blockSize, useShared);
    }

    auto start = std::chrono::high_resolution_clock::now();
    multiplyMatricesCUDA(A, B, C, n, blockSize, useShared);
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double time_ms = duration.count() / 1000.0;

    long long flops = 2LL * n * n * n;
    double gflops = flops / (time_ms * 1e6);

    std::cout << "\nWriting result to " << result_file << "..." << std::endl;
    if (!writeMatrix(result_file, C, n))
        return 1;

    std::cout << "Starting verification..." << std::endl;
    system(("python checkMultiply.py " + file_a + " " + file_b + " " + result_file).c_str());

    std::cout << "\n========== CUDA REPORT ==========" << std::endl;
    std::cout << "Matrix Size: " << n << "x" << n << std::endl;
    std::cout << "Block Size: " << blockSize << "x" << blockSize << std::endl;
    std::cout << "Shared Memory: " << (useShared ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Execution Time: " << std::fixed << std::setprecision(3) << time_ms << " ms" << std::endl;
    std::cout << "Estimated FLOPs: " << flops << std::endl;
    std::cout << "Performance: " << std::setprecision(2) << gflops << " GFLOPS" << std::endl;
    std::cout << "=================================" << std::endl;

    std::ofstream report("report_cuda.txt");
    if (report.is_open())
    {
        report << "Matrix Size: " << n << "x" << n << std::endl;
        report << "Block Size: " << blockSize << "x" << blockSize << std::endl;
        report << "Shared Memory: " << (useShared ? "Enabled" : "Disabled") << std::endl;
        report << "Execution Time (ms): " << time_ms << std::endl;
        report << "Estimated FLOPs: " << flops << std::endl;
        report << "Performance (GFLOPS): " << std::fixed << std::setprecision(2) << gflops << std::endl;
        report.close();
        std::cout << "Report saved to report_cuda.txt" << std::endl;
    }

    return 0;
}
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>

const int SIZES[] = {200, 400, 600, 800, 1000, 1500, 2000, 2500, 3000};
const int NUM_SIZES = 9;

void generateMatrix(std::vector<double> &matrix, int n)
{
    matrix.resize(n * n);
    for (int i = 0; i < n * n; ++i)
    {
        matrix[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

void multiplyPart(const std::vector<double> &A_part, const std::vector<double> &B,
                  std::vector<double> &C_part, int n, int rows)
{
    C_part.assign(rows * n, 0.0);
    for (int i = 0; i < rows; ++i)
        for (int k = 0; k < n; ++k)
        {
            double a_ik = A_part[i * n + k];
            for (int j = 0; j < n; ++j)
                C_part[i * n + j] += a_ik * B[k * n + j];
        }
}

double testMatrixSize(int n, int rank, int size)
{
    std::vector<double> A, B, C;
    
    if (rank == 0)
    {
        generateMatrix(A, n);
        generateMatrix(B, n);
    }
    
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0)
    {
        B.resize(n * n);
    }
    MPI_Bcast(B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    int rows_per_proc = n / size;
    int remainder = n % size;
    int my_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    int my_start_row = rank * rows_per_proc + std::min(rank, remainder);
    
    std::vector<double> A_part(my_rows * n);
    
    if (rank == 0)
    {
        for (int p = 1; p < size; ++p)
        {
            int p_rows = rows_per_proc + (p < remainder ? 1 : 0);
            int p_start = p * rows_per_proc + std::min(p, remainder);
            MPI_Send(A.data() + p_start * n, p_rows * n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
        }
        int my_start = my_start_row * n;
        std::copy(A.begin() + my_start, A.begin() + my_start + my_rows * n, A_part.begin());
    }
    else
    {
        MPI_Recv(A_part.data(), my_rows * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    if (rank == 0) A.clear();
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    std::vector<double> C_part;
    multiplyPart(A_part, B, C_part, n, my_rows);
    
    double end_time = MPI_Wtime();
    double exec_time = end_time - start_time;
    
    if (rank == 0)
    {
        C.resize(n * n);
        for (int i = 0; i < my_rows; ++i)
            for (int j = 0; j < n; ++j)
                C[(my_start_row + i) * n + j] = C_part[i * n + j];
        
        for (int p = 1; p < size; ++p)
        {
            int p_rows = rows_per_proc + (p < remainder ? 1 : 0);
            int p_start = p * rows_per_proc + std::min(p, remainder);
            std::vector<double> recv_buf(p_rows * n);
            MPI_Recv(recv_buf.data(), p_rows * n, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < p_rows; ++i)
                for (int j = 0; j < n; ++j)
                    C[(p_start + i) * n + j] = recv_buf[i * n + j];
        }
    }
    else
    {
        MPI_Send(C_part.data(), my_rows * n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }
    
    return exec_time;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    srand(42 + rank);
    
    if (rank == 0)
    {
        std::cout << "\n=== MPI Matrix Multiplication Benchmark ===" << std::endl;
        std::cout << "Processes: " << size << std::endl;
        std::cout << "Running tests..." << std::endl << std::endl;
    }
    
    std::vector<double> results(NUM_SIZES);
    
    for (int i = 0; i < NUM_SIZES; ++i)
    {
        int n = SIZES[i];
        
        if (rank == 0)
        {
            std::cout << "Testing " << n << "x" << n << "...";
            std::cout.flush();
        }
        
        double time = testMatrixSize(n, rank, size);
        results[i] = time;
        
        if (rank == 0)
        {
            std::cout << " done (" << std::fixed << std::setprecision(2) << time << " s)" << std::endl;
        }
    }
    
    if (rank == 0)
    {
        std::cout << "\n========== RESULTS ==========" << std::endl;
        std::cout << "Matrix Size | Time (s) | Performance (GFLOPS)" << std::endl;
        std::cout << "------------|------------|----------------------" << std::endl;
        
        for (int i = 0; i < NUM_SIZES; ++i)
        {
            int n = SIZES[i];
            double time = results[i];
            long long flops = 2LL * n * n * n;
            double gflops = (time > 0) ? flops / (time * 1e9) : 0;
            
            std::cout << std::setw(10) << n << " | " 
                      << std::setw(9) << std::fixed << std::setprecision(3) << time << " | "
                      << std::setw(15) << std::setprecision(2) << gflops << std::endl;
        }
        
        std::cout << "\n========== CSV FORMAT ==========" << std::endl;
        std::cout << "Size";
        for (int i = 0; i < NUM_SIZES; ++i)
        {
            std::cout << "," << SIZES[i];
        }
        std::cout << std::endl;
        
        std::cout << "Time";
        for (int i = 0; i < NUM_SIZES; ++i)
        {
            std::cout << "," << std::fixed << std::setprecision(2) << results[i];
        }
        std::cout << std::endl;
        
        std::ofstream file("benchmark_results.csv");
        if (file.is_open())
        {
            file << "Size";
            for (int i = 0; i < NUM_SIZES; ++i)
            {
                file << "," << SIZES[i];
            }
            file << std::endl;
            
            file << "Time";
            for (int i = 0; i < NUM_SIZES; ++i)
            {
                file << "," << std::fixed << std::setprecision(2) << results[i];
            }
            file << std::endl;
            file.close();
            std::cout << "\nResults saved to benchmark_results.csv" << std::endl;
        }
        
        std::cout << "\n==============================" << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}

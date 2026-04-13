#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <string>
#include <algorithm>
#include <mpi.h>

// === Чтение матрицы из файла ===
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

bool writeMatrix(const std::string &filename, const std::vector<double> &matrix, int n)
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
            file << std::setprecision(15) << matrix[i * n + j] << " ";
        }
        file << std::endl;
    }
    file.close();
    return true;
}

void multiplyPart(const std::vector<double> &A_part, const std::vector<double> &B,
                  std::vector<double> &C_part, int n, int rows)
{
    C_part.assign(rows * n, 0.0);
    for (int i = 0; i < rows; ++i)
    {
        for (int k = 0; k < n; ++k)
        {
            double a_ik = A_part[i * n + k];
            for (int j = 0; j < n; ++j)
            {
                C_part[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string file_a = "data/matrix_a.txt";
    std::string file_b = "data/matrix_b.txt";
    std::string result_file = "data/matrix_res.txt";

    if (argc >= 4)
    {
        file_a = argv[1];
        file_b = argv[2];
        result_file = argv[3];
    }

    int n = 0;
    std::vector<double> A, B, C;

    if (rank == 0)
    {
        std::cout << "Reading matrices..." << std::endl;
        if (!readMatrix(file_a, A, n))
        {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (!readMatrix(file_b, B, n))
        {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::cout << "Matrix size: " << n << "x" << n << ", Processes: " << size << std::endl;
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
        A_part.assign(A.begin() + my_start, A.begin() + my_start + my_rows * n);
    }
    else
    {
        MPI_Recv(A_part.data(), my_rows * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (rank == 0)
    {
        A.clear();
    }

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
        {
            for (int j = 0; j < n; ++j)
            {
                C[(my_start_row + i) * n + j] = C_part[i * n + j];
            }
        }
        for (int p = 1; p < size; ++p)
        {
            int p_rows = rows_per_proc + (p < remainder ? 1 : 0);
            int p_start = p * rows_per_proc + std::min(p, remainder);
            std::vector<double> recv_buf(p_rows * n);
            MPI_Recv(recv_buf.data(), p_rows * n, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < p_rows; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    C[(p_start + i) * n + j] = recv_buf[i * n + j];
                }
            }
        }

        std::cout << "Writing result to " << result_file << "..." << std::endl;
        writeMatrix(result_file, C, n);

        long long flops = 2LL * n * n * n;
        double gflops = flops / (exec_time * 1e9);

        std::cout << "\n========== MPI REPORT ==========" << std::endl;
        std::cout << "Matrix Size: " << n << "x" << n << std::endl;
        std::cout << "Processes: " << size << std::endl;
        std::cout << "Execution Time (s): " << std::fixed << std::setprecision(6) << exec_time << std::endl;
        std::cout << "Estimated FLOPs: " << flops << std::endl;
        std::cout << "Performance (GFLOPS): " << std::setprecision(2) << gflops << std::endl;
        std::cout << "==============================" << std::endl;

        system(("python checkMultiply.py " + file_a + " " + file_b + " " + result_file).c_str());

        std::ofstream report("report_mpi.txt");
        if (report.is_open())
        {
            report << "Matrix Size: " << n << "x" << n << std::endl;
            report << "Processes: " << size << std::endl;
            report << "Execution Time (s): " << exec_time << std::endl;
            report << "Performance (GFLOPS): " << gflops << std::endl;
            report.close();
            std::cout << "Report saved to report_mpi.txt" << std::endl;
        }
    }
    else
    {
        MPI_Send(C_part.data(), my_rows * n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
#include <vector>

#include "matvecop.h"
#include "matrix.h"

int axpy(
    double a, //constant for multiplying vector x by it
    const std::vector<double>& x, //vector multiplied by a and added with y
    const std::vector<double>& y, //adding vector
    int N,
    std::vector<double>& z //vector to write result
    
) {

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        z[i] = a * x[i] + y[i];
    }

    return 0;
}

double dot(
    const std::vector<double>& x,//first vector
    const std::vector<double>& y,//second vector
    int N,
    double& res
) {
    double re = 0.0;
    #pragma omp parallel for reduction(+:re)
    for (int i = 0; i < N; ++i) {
        re += x[i] * y[i];
    }
    res = re;
    return 0.0;
}

int SpMV (
    const CSRMatrix& A, //matrix in CSR format
    std::vector<double>& x, //vector multiplying matrix A by it
    std::vector<double>& y //vector to write result
) {
    int N = A.n;

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        const int jb = A.rowPtr[i], je = A.rowPtr[i+1];
        for (int j = jb; j < je; ++j) {
            sum += A.values[j] * x[A.colIndex[j]];
        }
        y[i] = sum;
    }

    return 0;
}

int matmul (
    std::vector<double>& x,//first vector
    std::vector<double>& y,//second vector
    std::vector<double>& z//vector to write result
) {

    #pragma omp parallel for
    for (int i = 0; i < x.size(); ++i) {
        z[i] = x[i] * y[i];
    }

    return 0;
}

int find_antidiag (
    const CSRMatrix& A, //matrix in CSR format whose diagonal^(-1) is being searched for
    std::vector<double>& antidiag//vector to write result
) {
    int N = A.n;

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        int jb = A.rowPtr[i], je = A.rowPtr[i+1] - 1;

        while(jb <= je) {
            int mid = (jb + je) / 2;
            if (i == A.colIndex[mid]) {
                antidiag[i] = 1 / A.values[mid];
                break;
            }
            if (A.colIndex[mid] > i) {
                je = mid - 1;
            } else {
                jb = mid + 1;
            }
        }
    }

    return 0;
}
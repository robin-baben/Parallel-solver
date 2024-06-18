#ifndef __matvecop_h__
#define __matvecop_h__

#include <vector>
#include "matrix.h"

int axpy(
    double a, //constant for multiplying vector x by it
    const std::vector<double>& x, //vector multiplied by a and added with y
    const std::vector<double>& y, //adding vector
    int N,
    std::vector<double>& z//vector to write result
);

double dot(
    const std::vector<double>& x,//first vector
    const std::vector<double>& y,//second vector
    int N,
    double& res
);

int SpMV (
    const CSRMatrix& A, //matrix in CSR format
    std::vector<double>& x, //vector multiplying matrix A by it
    std::vector<double>& y //vector to write result
);

int matmul (
    std::vector<double>& x,//first vector
    std::vector<double>& y,//second vector
    std::vector<double>& z//vector to write result
);

int find_antidiag (
    const CSRMatrix& A, //matrix in CSR format whose diagonal^(-1) is being searched for
    std::vector<double>& antidiag//vector to write result
);

#endif


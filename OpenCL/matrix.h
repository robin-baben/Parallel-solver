#ifndef __matrix_h__
#define __matrix_h__

#include <vector>

struct CSRMatrix_rltn
{
    int n; // number of rows
    int m; // number of columns
    int nz; // number of non-zero elements
    std::vector<int> colIndex; // column indices
    std::vector<int> rowPtr; // row ptr
};

struct CSRMatrix : CSRMatrix_rltn
{
    std::vector<double> values; //non-zero values

    // Default constructor
    CSRMatrix() {

    }

    // Initialize CSRMAtrix with CSRMatrix_rltn portrait
    CSRMatrix(CSRMatrix_rltn& A) {
        n = A.n;
        m = A.m;
        nz = A.nz;
        colIndex = A.colIndex;
        rowPtr = A.rowPtr;
    }
};



int sparse_transpose(
    const CSRMatrix_rltn& input, // matrix to transpose
    CSRMatrix_rltn& res // matrix to result
);

void print_CSRMatrix_rltn(
    const CSRMatrix_rltn& A
);

void print_CSRMAtrix(
    const CSRMatrix& A
);

void print_vector(
    const std::vector<double> &b
);

int generator_EN(
    CSRMatrix_rltn &EN, // matrix adjaction elements to nodes
    int Ny, // number of nodes vertically
    int Nx, // number of nodes horizontally 
    int k3, // the number of shared cells to the number of non-shared cells (shared:non_shared)
    int k4
);

int generator_NeN(
    const CSRMatrix_rltn& NE, // matrix adjaction nodes to elements
    const CSRMatrix_rltn& EN, // matrix adjaction elements to nodes
    CSRMatrix_rltn& res // result, matrix adjaction nodes from element
);

int fill_Matrix(
    CSRMatrix& A, //matrix to fill
    int T //number of threads
);

int find_antidiag (
    const CSRMatrix& A, //matrix in CSR format whose diagonal^(-1) is being searched for
    std::vector<double>& antidiag//vector to write result
);

#endif
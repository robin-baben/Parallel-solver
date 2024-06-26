#include <vector>
#include <iostream>
#include <set>
#include <cmath>
#include <omp.h>

#include "matrix.h"



int sparse_transpose(
    const CSRMatrix_rltn& input, // matrix to transpose
    CSRMatrix_rltn& res // matrix to result
) {
    
    res.n = input.m;
    res.m = input.n;
    res.nz = input.nz;
    res.colIndex = std::vector<int>(input.nz, 0);
    res.rowPtr = std::vector<int>(input.m + 2, 0); // one extra

    // count per column
    for (int i = 0; i < input.nz; ++i) {
        res.rowPtr[input.colIndex[i] + 2] += 1; 
    }
    

    // from count per column generate new rowPtr (but shifted)
    for (int i = 2; i < (int)res.rowPtr.size(); ++i) {
        // create incremental sum
        res.rowPtr[i] += res.rowPtr[i - 1];
    }

    // perform the main part
    for (int i = 0; i < input.n; ++i) {
        for (int j = input.rowPtr[i]; j < input.rowPtr[i + 1]; ++j) {
            // calculate index to transposed matrix at which we should place current element, and at the same time build final rowPtr
            const int new_index = res.rowPtr[input.colIndex[j] + 1]++;
            res.colIndex[new_index] = i;
        }
    }
    res.rowPtr.pop_back(); // pop that one extra

    return (res.colIndex.capacity()+res.rowPtr.capacity()) * sizeof(int);
}

void print_CSRMatrix_rltn(
    const CSRMatrix_rltn& A
) {
    for (int i = 0; i < A.n; ++i) {
        for (int j = A.rowPtr[i]; j < A.rowPtr[i+1]; ++j) {
            std::cout << A.colIndex[j] << "\t";
        }
        std::cout << std::endl;
    }
}

void print_CSRMAtrix(
    const CSRMatrix& A
) {
    for (int i = 0; i < A.n; ++i) {
        int k = 0;
        for (int j = A.rowPtr[i]; j < A.rowPtr[i+1]; ++j) {
            for (int l = k; l < A.colIndex[j]; ++l) {
                printf("0.000\t");
            }
            printf("%5.3f\t", A.values[j]);
            k = A.colIndex[j] + 1;
        }
        while (k < A.m) {
            printf("0.000\t");
            k++;
        }
        std::cout << std::endl;
    }
}

void print_vector(
    const std::vector<double> &b
) {
    int N = b.size();
    for (int i = 0; i < N; ++i) {
        printf("%f ", b[i]);
    }
    printf("\n");
}

int generator_EN(
    CSRMatrix_rltn &EN, // matrix adjaction elements to nodes
    int N_y, // number of nodes vertically
    int N_x, // number of nodes horizontally 
    int shared, // the number of shared cells to the number of non-shared cells (shared:non_shared)
    int non_shared
) {
    

    int num_of_shared = ((N_x - 1) * (N_y - 1) / (non_shared + shared)) * shared;
    
    num_of_shared += std::min((N_x - 1) * (N_y - 1) % (non_shared + shared), shared);
    

    int num_of_elems = (N_x - 1) * (N_y - 1) + num_of_shared; // number of elements
    int num_of_nodes = N_x * N_y; // number of nodes

    EN.n = num_of_elems;

    EN.m = num_of_nodes;
    // EN.nz = 6 * num_of_shared + 4 * (num_of_elems - 2 * num_of_shared);
    EN.nz = 4 * num_of_elems - 2 * num_of_shared;
    EN.colIndex = std::vector<int>(EN.nz, 0);
    EN.rowPtr = std::vector<int>(EN.n + 1, 0);

    int k = 0; // counter of elements
    int flag = 0; //shared from left-up to right-down(0) or left-down to right-up(1)

    for (int i = 0; i < (N_x - 1) * (N_y - 1); ++i) {
        int add_coef = i / (N_x - 1);

        int begin = EN.rowPtr[k];

        if (i % (non_shared + shared) < shared) {
            
            EN.rowPtr[k+1] = begin + 3;
            EN.rowPtr[k+2] = begin + 6;

            if (flag) {
                EN.colIndex[begin] = i + add_coef;
                EN.colIndex[begin + 1] = i + 1 + add_coef;
                EN.colIndex[begin + 2] = i + N_x + add_coef;
                EN.colIndex[begin + 3] = i + 1 + add_coef;
                EN.colIndex[begin + 4] = i + N_x + add_coef;
                EN.colIndex[begin + 5] = i + N_x + 1 + add_coef;

                flag = 0;
            } else {
                EN.colIndex[begin] = i + add_coef;
                EN.colIndex[begin + 1] = i + N_x + add_coef;
                EN.colIndex[begin + 2] = i + N_x + 1 + add_coef;
                EN.colIndex[begin + 3] = i + add_coef;
                EN.colIndex[begin + 4] = i + 1 + add_coef;
                EN.colIndex[begin + 5] = i + N_x + 1 + add_coef;

                flag = 1;
            }
            k += 2;
        } else {
            EN.rowPtr[k+1] = begin + 4;

            EN.colIndex[begin] = i + add_coef;
            EN.colIndex[begin + 1] = i + 1 + add_coef;
            EN.colIndex[begin + 2] = i + N_x + add_coef;
            EN.colIndex[begin + 3] = i + N_x + 1 + add_coef;

            ++k;
        }
    }

    //std::cout << 0 << std::endl;
    return (EN.colIndex.capacity()+EN.rowPtr.capacity()) * sizeof(int);
}


int generator_NeN(
    const CSRMatrix_rltn& NE, // matrix adjaction nodes to elements
    const CSRMatrix_rltn& EN, // matrix adjaction elements to nodes
    CSRMatrix_rltn& res // result, matrix adjaction nodes from element
) {
    res.n = NE.n;
    res.m = NE.n;
    res.rowPtr = std::vector<int>(res.n + 1, 0);

    int max_size_set = 0;

    for (int node = 0; node < NE.n; ++node) {

        std::set<int> adj_nodes;
        for (int i = NE.rowPtr[node]; i < NE.rowPtr[node + 1]; ++i) {
            for (int j = EN.rowPtr[NE.colIndex[i]]; j < EN.rowPtr[NE.colIndex[i] + 1]; ++j) {
                adj_nodes.insert(EN.colIndex[j]);
            }
        }

        //adj_nodes.erase(node);

        if (max_size_set < (int)adj_nodes.size()) {
            max_size_set = adj_nodes.size();
        }

        res.rowPtr[node + 1] = res.rowPtr[node] + adj_nodes.size();
        
        res.colIndex.insert(res.colIndex.end(), adj_nodes.begin(), adj_nodes.end());
    }
    res.nz = res.rowPtr[res.n];

    return (res.colIndex.capacity()+res.rowPtr.capacity() + max_size_set) * sizeof(int);
}


int fill_Matrix(
    CSRMatrix& A, //matrix to fill
    int T //number of threads
) {
    #pragma omp parallel for schedule(dynamic, 1000)
    for (int i = 0; i < A.n; ++i) {
        double aii = 0.0;
        double num_aii = 0.0;
        for (int j = A.rowPtr[i]; j < A.rowPtr[i+1]; ++j) {
            if (i != A.colIndex[j]) {
                double elem = cos(i * A.colIndex[j] + i + A.colIndex[j]);
                A.values[j] = elem;
                aii += fabs(elem);
            } else {
                num_aii = j;
            }
        }
        A.values[num_aii] = aii * 1.234;
    }
    return A.values.capacity() * sizeof(double);
}

int find_antidiag (
    const CSRMatrix& A, //matrix in CSR format whose diagonal^(-1) is being searched for
    std::vector<double>& antidiag//vector to write result
) {
    int N = antidiag.size();

    #pragma omp parallel for schedule(dynamic, 1000)
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
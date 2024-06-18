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
    int Ny, // number of nodes vertically
    int Nx, // number of nodes horizontally 
    int k3, // the number of shared cells to the number of non-shared cells (shared:non_shared)
    int k4,
    int p,
    int MyID,
    std::vector<int> &L2G, // local to global indexes
    SR &SendRec,
    Lengths &Len
) {
    int y = MyID / p; // y coordinate of subgraph
    int x = MyID % p; // x coordinate of subgraph

    int Nx_local = Nx / p + (x < (Nx % p)? 1 : 0); // local + interface nodes
    int Ny_local = Ny / p + (y < (Ny % p) ? 1 : 0); //local + interface
        
    int Nx_elem = Nx_local + 1; // number of elements
    if (x == 0) {
        Nx_elem--;
    }
    if (x == p-1) {
        Nx_elem--;
    }

    int Ny_elem = Ny_local + 1;
    if (y == 0) {
        Ny_elem--;
    }
    if (y == p-1) {
        Ny_elem--;
    }

    L2G = std::vector<int>((Nx_elem+1)*(Ny_elem+1));

    int gbl_ind_first_node = x * Nx_local - ((x == 0) ? 0 : 1) + std::min(x, Nx % p) +
                                (y * Ny_local - (y == 0 ? 0 : 1)) * Nx + std::min(y, Ny % p) * Nx;
    int gbl_shift = gbl_ind_first_node / Nx;
    int gbl_ind_first_elem = gbl_ind_first_node - gbl_shift;

    EN.rowPtr.push_back(0);

    int k = 0;

    for (int i = 0; i < Ny_elem; ++i) { // записывать будем по строчкам
        for (int j = 0; j < Nx_elem; ++j) {
            int ind = i * Nx_elem + j;

            int begin = EN.rowPtr.back();


            if ((gbl_ind_first_elem + j) % (k4 + k3) < k3) {

                k++;

                EN.rowPtr.push_back(begin + 3);
                EN.rowPtr.push_back(begin + 6);


                int gbl_ind = gbl_ind_first_elem + j; // глобальный индекс элемента
                int already_shared = gbl_ind / (k4 + k3) + std::min(k3, gbl_ind % (k4 + k3));

                if (already_shared % 2 == 0) {

                    EN.colIndex.push_back(ind + i);
                    EN.colIndex.push_back(ind + Nx_elem + i + 1);
                    EN.colIndex.push_back(ind + Nx_elem + i + 2);

                    EN.colIndex.push_back(ind + i);
                    EN.colIndex.push_back(ind + i + 1);
                    EN.colIndex.push_back(ind + Nx_elem + i + 2);
                } else {

                    EN.colIndex.push_back(ind + i);
                    EN.colIndex.push_back(ind + i + 1);
                    EN.colIndex.push_back(ind + Nx_elem + i + 1);

                    EN.colIndex.push_back(ind + i + 1);
                    EN.colIndex.push_back(ind + Nx_elem + i + 1);
                    EN.colIndex.push_back(ind + Nx_elem + i + 2);
                }

                L2G[ind+i] = gbl_ind_first_node + j;
                L2G[ind+i+1] = gbl_ind_first_node + j + 1;
                L2G[ind + Nx_elem + 1 + i] = gbl_ind_first_node + j + Nx;
                L2G[ind + Nx_elem + 2 + i] = gbl_ind_first_node + j + Nx + 1;

            } else {
                EN.rowPtr.push_back(begin + 4);

                EN.colIndex.push_back(ind + i);
                EN.colIndex.push_back(ind + 1 + i);
                EN.colIndex.push_back(ind + Nx_elem + i + 1);
                EN.colIndex.push_back(ind + Nx_elem + 2 + i);

                L2G[ind + i] = gbl_ind_first_node + j;
                L2G[ind + i + 1] = gbl_ind_first_node + j + 1;
                L2G[ind + Nx_elem + 1 + i] = gbl_ind_first_node + j + Nx;
                L2G[ind + Nx_elem + 2 + i] = gbl_ind_first_node + j + Nx + 1;
            }
        }

        gbl_ind_first_elem += Nx-1; //обновляем глобальные индексы для следующей строки
        gbl_ind_first_node += Nx;
    }

    EN.nz = EN.rowPtr.back();
    EN.n = Nx_elem * Ny_elem + k;
    EN.m = (Nx_elem + 1) * (Ny_elem + 1);


    


    std::vector<int> interface;
    std::vector<int> selff;
    std::vector<std::vector<int> > halo(p * p);

    int up = (y == 0 ? 0 : 1);
    int down = (y == p-1 ? 0 : 1);
    int left = (x == 0 ? 0 : 1);
    int right = (x == p-1 ? 0 : 1);

    int num_rows = Ny_elem+1;
    int num_cols = Nx_elem+1;

    int count = 0;

    if (up) {
        if (left) {
            halo[MyID-p-1].push_back(count);
            SendRec.neighbours[MyID-p-1]++;
            count++;
        }
        for (int i = 0; i < num_cols - left - right; ++i) {
            halo[MyID-p].push_back(count);
            SendRec.neighbours[MyID-p]++;
            count++;
        }
        if (right) {
            halo[MyID-p+1].push_back(count);
            SendRec.neighbours[MyID-p+1]++;
            count++;
        }
        
        if (left) {
            halo[MyID-1].push_back(count);
            SendRec.neighbours[MyID-1]++;
            count++;

            interface.push_back(count);
            SendRec.send[MyID-p-1].push_back(count);
            SendRec.send[MyID-p].push_back(count);
            SendRec.send[MyID-1].push_back(count);
            count++;
        }
        for (int i = 0; i < num_cols - 2*left - 2*right; ++i) {
            interface.push_back(count);
            SendRec.send[MyID-p].push_back(count);
            count++;
        }
        if (right) {
            interface.push_back(count);
            SendRec.send[MyID-p+1].push_back(count);
            SendRec.send[MyID-p].push_back(count);
            SendRec.send[MyID+1].push_back(count);
            count++;

            halo[MyID+1].push_back(count);
            SendRec.neighbours[MyID+1]++;
            count++;
        }
    }
    for (int i = 0; i < num_rows - 2*up - 2*down; ++i) {
        if (left) {
            halo[MyID-1].push_back(count);
            SendRec.neighbours[MyID-1]++;
            count++;

            interface.push_back(count);
            SendRec.send[MyID-1].push_back(count);
            count++;
        }
        for (int i = 0; i < num_cols - 2*left - 2*right; ++i) {
            selff.push_back(count);
            count++;
        }
        if (right) {
            interface.push_back(count);
            SendRec.send[MyID+1].push_back(count);
            count++;

            halo[MyID+1].push_back(count);
            SendRec.neighbours[MyID+1]++;
            count++;
        }
    }
    if (down) {
        if (left) {
            halo[MyID-1].push_back(count);
            SendRec.neighbours[MyID-1]++;
            count++;

            interface.push_back(count);
            SendRec.send[MyID+p-1].push_back(count);
            SendRec.send[MyID+p].push_back(count);
            SendRec.send[MyID-1].push_back(count);
            count++;
        }
        for (int i = 0; i < num_cols - 2*left - 2*right; ++i) {
            interface.push_back(count);
            SendRec.send[MyID+p].push_back(count);
            count++;
        }
        if (right) {
            interface.push_back(count);
            SendRec.send[MyID+p+1].push_back(count);
            SendRec.send[MyID+p].push_back(count);
            SendRec.send[MyID+1].push_back(count);
            count++;

            halo[MyID+1].push_back(count);
            SendRec.neighbours[MyID+1]++;
            count++;
        }

        if (left) {
            halo[MyID+p-1].push_back(count);
            SendRec.neighbours[MyID+p-1]++;
            count++;
        }
        for (int i = 0; i < num_cols - left - right; ++i) {
            halo[MyID+p].push_back(count);
            SendRec.neighbours[MyID+p]++;
            count++;
        }
        if (right) {
            halo[MyID+p+1].push_back(count);
            SendRec.neighbours[MyID+p+1]++;
            count++;
        }
    }

    Len.len_self = selff.size();
    Len.len_inter = interface.size();

    selff.insert(selff.end(), interface.begin(), interface.end());
    for (int i = 0; i < halo.size(); ++i) {
        selff.insert(selff.end(), halo[i].begin(), halo[i].end()); 
    }
    
    //получили переход от новой нумерации к старой
    // теперь надо поменять L2G, colindexes, send, сделав массив O2N

    std::vector<int> O2N(selff.size());
    
    for (int i = 0; i < selff.size(); ++i) {
        O2N[selff[i]] = i;
    }

    std::vector<int> newL2G(L2G.size());

    for (int i = 0; i < L2G.size(); ++i) {
        newL2G[i] = L2G[selff[i]];
    }

    L2G = newL2G;

    for (int i = 0; i < EN.colIndex.size(); ++i) {
        EN.colIndex[i] = O2N[EN.colIndex[i]];
    }

    for (int i = 0; i < SendRec.send.size(); ++i) {
        for (int j = 0; j < SendRec.send[i].size(); ++j) {
            SendRec.send[i][j] = O2N[SendRec.send[i][j]];
        }
    }

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
    std::vector<int> L2G, // local to global indexes
    int T //number of threads
) {
    #pragma omp parallel for schedule(dynamic, 100)
    for (int i = 0; i < A.n; ++i) {
        double aii = 0.0;
        double num_aii = 0.0;
        for (int j = A.rowPtr[i]; j < A.rowPtr[i+1]; ++j) {
            if (i != A.colIndex[j]) {
                double elem = cos(L2G[i] * L2G[A.colIndex[j]] + L2G[i] + L2G[A.colIndex[j]]);
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
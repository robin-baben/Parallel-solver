#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <omp.h>
#include <string.h>
#include <map>
#include <cmath>
#include <utility>
#include <unistd.h>

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
    CSRMatrix_rltn& res, // matrix to result
    int T // number of threads
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

int generator_EN(
    CSRMatrix_rltn &EN, // matrix adjaction elements to nodes
    int N_y, // number of nodes vertically
    int N_x, // number of nodes horizontally 
    int shared, // the number of shared cells to the number of non-shared cells (shared:non_shared)
    int non_shared,
    int T // number of threads
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
    CSRMatrix_rltn& res, // result, matrix adjaction nodes from elements
    int T // number of threads
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

int axpy(
    double a, //constant for multiplying vector x by it
    const std::vector<double>& x, //vector multiplied by a and added with y
    const std::vector<double>& y, //adding vector
    std::vector<double>& z //vector to write result
) {
    int N = x.size();

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        z[i] = a * x[i] + y[i];
    }

    return 0;
}

double dot(
    const std::vector<double>& x,//first vector
    const std::vector<double>& y,//second vector
    double& res
) {
    int N = x.size();
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
    int N = x.size();

    #pragma omp parallel for schedule(dynamic, 1000)
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
    int N = x.size();

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        z[i] = x[i] * y[i];
    }

    return 0;
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


std::vector<std::string> solver(
    int T, //number of threads
    const CSRMatrix& A, //matrix of system, size N*N
    const std::vector<double>& b, //vector of right part, size N
    double eps, // stopping criteria
    int maxit, // maximum number of iterations
    std::vector<double>& x, //vector for writing solution, size N
    int& n, //number of operations performed
    double& res //L2 discrepancy rate
) { 
    std::vector<std::string> retur(4);

    int N = b.size();
    std::vector<double> antidiag = std::vector<double>(N);
    
    double t_diag = omp_get_wtime();
    find_antidiag(A, antidiag);
    retur[0] = std::to_string(omp_get_wtime() - t_diag);

    std::vector<double> r_k = b;
    x = std::vector<double>(N, 0.0);

    std::vector<double> p_k = std::vector<double>(N);
    std::vector<double> z_k = std::vector<double>(N);
    std::vector<double> q_k = std::vector<double>(N);

    int k = 0;
    double rho_old = 0;
    double rho_new = 0;

    double t_solve = omp_get_wtime();
    do
    {
        
        k++;
        matmul(r_k, antidiag, z_k);
        

        rho_old = rho_new;
        dot(r_k, z_k, rho_new);

            
        printf("%f\n", rho_new);
            
        
        if (k == 1) {
            p_k = z_k;
        } else {
            double beta_k = rho_new / rho_old;
            
            axpy(beta_k, p_k, z_k, p_k);
        }

        SpMV(A, p_k, q_k);
        
        double pq;
        dot(p_k, q_k, pq);
        double alfa = rho_new / pq;
        
        axpy(alfa, p_k, x, x);
        axpy(-alfa, q_k, r_k, r_k);
    }
    while ((rho_new > eps * eps) && (k < maxit));
    retur[1] = std::to_string(omp_get_wtime() - t_solve);

    
    n = k;
    

    std::vector<double> r = std::vector<double>(N);

    double t_res = omp_get_wtime();
    SpMV(A, x, r);
    axpy(-1.0, b, r, r);
    dot(r, r, res);
    res = sqrt(res);
    retur[2] = std::to_string(omp_get_wtime() - t_res);

    int memory = 5 * N * sizeof(double);
    retur[3] = std::to_string(memory);

    return retur;
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


static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << " <option(s)> \n"
            << "Options:\n"
            << "\t--help\t\tShow this help message.\n"
            << "\tMust-have paraneters:\n"
            << "\t-Nx <int1>\tNumber of nodes horizontally, the value is integer int1,\n"
            << "\t\t\tint1 is greater then 1.\n"
            << "\t-Ny <int2>\tNumber of nodes vertically, the value is integer int2,\n"
            << "\t\t\tint2 is greater then 1.\n"
            << "\t-k3 <int3>\tNumber of shared cells, the value is a non-negative integer int3.\n"
            << "\t-k4 <int4>\tNumber of non-shared cells, the value is a non-negative integer int4.\n"
            << "\tOptional paraneters, comes after must-have parameters:\n"
            << "\t-T <int5>\tNumber of threads, by default 1, the value is non-negative integer no more than 8.\n"
            << "\t-eps <double1>\taccuracy, the value is positive real number double1.\n"
            << "\t--debug\t\tTurn on debug mode with printing matrices, can only be the last parameter\n"
            << std::endl;
}

int parcing_args(
    int argc, //number of arguments
    char** argv, // arguments
    int& Nx,
    int& Ny,
    int& k3,
    int& k4,
    int& debug,
    double& eps,
    int& T
) {
    if (argc == 1) {
        printf("no arguments were passed to the function\n help: --help\n");
        return 1;
    } else if (argc == 2) { // check help
        if (strcmp(argv[1], "--help") == 0) {
            show_usage(argv[0]+2);
            return 1;
        } else {
            printf("Incorrect arguments\n help: --help\n");
            return 1;
        }
    } else if ((8 < argc) && (argc < 15)) {
        std::set<std::string> check {"-Nx", "-Ny", "-k3", "-k4"};
        
        std::set<std::string> test {argv[1], argv[3], argv[5], argv[7]};

        if (check == test) { // check right arguments
            std::map<std::string, int> products;

            for (int i = 1; i < 9; i+=2) {
                char* it = argv[i+1];
                while (*it != '\0' && std::isdigit(*it)) ++it; // check integer
                if (*it == '\0') {
                    products[argv[i]] = atoi(argv[i+1]);
                } else {
                    printf("Incorrect value of argument %s\n help: --help\n", argv[i]);
                    return 1;
                }
            }

            int i = 9;
            while(i < argc) {
                if (argc == (i + 1)) {
                    if (strcmp(argv[i],"--debug") == 0) {
                        debug = 1;
                    } else if (strcmp(argv[i],"-T") == 0){
                        printf("the parameter -T value was not passed, the default value was used\n help: --help\n\n");
                    } else if (strcmp(argv[i],"-eps") == 0){
                        printf("the parameter -eps value was not passed, the default value was used\n help: --help\n\n");
                    } else {
                        printf("unknown last argument\n help: --help\n");
                        return 1;
                    }

                    i++;
                } else {
                    if (strcmp(argv[i],"-T") == 0){
                        if (atoi(argv[i+1]) > 0 && atoi(argv[i+1]) <= 8) {
                            T = atoi(argv[i+1]);
                        } else {
                            printf("Incorrect value of argument -T\n help: --help\n");
                            return 1;
                        }
                    } else if (strcmp(argv[i],"-eps") == 0) {
                        double e = atof(argv[i+1]);
                        if (e > 0.0) {
                            eps = e;
                        } else {
                            printf("Incorrect value of argument -eps\n help: --help\n");
                            return 1;
                        }
                    } else {
                        printf("Incorrect argument %s\n help: --help\n", argv[i]);
                        return 1;
                    }

                    i += 2;
                }
            }

            Nx = products["-Nx"];
            Ny = products["-Ny"];
            k3 = products["-k3"];
            k4 = products["-k4"];

            if (Nx < 2) {
                printf("Argument -Nx must be greater then 1\n help: --help\n");
                return 1;
            } else if (Ny < 2) {
                printf("Argument -Ny must be greater then 1\n help: --help\n");
                return 1;
            } else if (k3 + k4 < 1) {
                printf("arguments -k3 and -k4 cannot be equal to 0 at the same time\n help: --help\n");
            } else {
                if (debug && (Nx > 5 || Ny > 5)) {
                    printf("The debug mode is only for -Nx and -Ny values less than 5\n help: --help\n\n");
                    debug = 0;
                }
                return 0;
            }
        } else {
            printf("Incorrect names of arguments\n help: --help\n");
            return 1;
        }
    } else {
        printf("Incorrect number of arguments\n help: --help\n");
        return 1;
    }
    return 1;
}

int main(int argc, char* argv[]) {
    int Nx, Ny, k3, k4;
    int debug = 0;
    double eps = 1e-3;
    int maxit = 1000;
    int T = 1;

    if(!parcing_args(argc, argv, Nx, Ny, k3, k4, debug, eps, T)) {
        omp_set_num_threads(T);

        printf("Generating EN matrix...\n");
        double t_gener = omp_get_wtime();
        CSRMatrix_rltn EN;
        int memory_gener = generator_EN(EN, Nx, Ny, k3, k4, T);
        t_gener = omp_get_wtime() - t_gener;
    
        if (debug) {
            std::cout << "Matrix EN\n";
            print_CSRMatrix_rltn(EN);
            std::cout << std::endl;
        }
        printf("\n");
    
        printf("Transpose EN matrix, receiving NE matrix...\n");
        double t_transpose = omp_get_wtime();
        CSRMatrix_rltn NE;
        int memory_transpose = sparse_transpose(EN, NE, T);
        t_transpose = omp_get_wtime() - t_transpose;

        if (debug) {
            std::cout << "Matrix NE\n";
            print_CSRMatrix_rltn(NE);
            std::cout << std::endl;
        }
        printf("\n");
    
        printf("Receiving NeN matrix...\n");
        double t_recieve = omp_get_wtime();
        CSRMatrix_rltn NeN;
        int memory_recieve = generator_NeN(NE, EN, NeN, T);
        t_recieve = omp_get_wtime() - t_recieve;

        if (debug) {
            std::cout << "Matrix NeN\n";
            print_CSRMatrix_rltn(NeN);
            std::cout << std::endl;
        }
        printf("\n");

        CSRMatrix A(NeN);
        A.values = std::vector<double>(A.nz);

        printf("Filling the matrix A...\n");
        double t_fill_A = omp_get_wtime();
        int memory_fill_A = fill_Matrix(A, T);
        t_fill_A = omp_get_wtime() - t_fill_A;

        if (debug) {
            printf("Matrix A:\n");
            print_CSRMAtrix(A);
        }
        printf("\n");


        std::vector<double> b(A.n);
        printf("Filling the vector b...\n");
        double t_fill_b = omp_get_wtime();
        #pragma omp parallel for
        for (int i = 0; i < A.n; ++i) {
            b[i] = sin(i);
        }
        t_fill_b = omp_get_wtime() - t_fill_b;
        int memory_fill_b = A.n * sizeof(double);

        if (debug) {
            printf("Vector b:\n");
            print_vector(b);
        }
        printf("\n");

        std::vector<double> x;
        double res;
        int n;

        std::vector<std::string> t_m_solve;
        
        printf("Solving system Ax = b ...\n");
        double t_solve = omp_get_wtime();
        t_m_solve = solver(T, A, b, eps, maxit, x, n, res);
        t_solve = omp_get_wtime() - t_solve;
        printf("the process was completed in %d iterations with precision %f\n", n, res);
        if (debug) {
            printf("Vector x:\n");
            print_vector(x);

            std::vector<double> Ax(x.size());
            SpMV(A, x, Ax);

            printf("Vector Ax:\n");
            print_vector(Ax);

            printf("Vector b:\n");
            print_vector(b);
        }
        printf("\n");

        printf("\tProcess\t\tTime\t\tMemory\n");
        printf("\tGenerating En\t%f s\t%d bytes\n", t_gener, memory_gener);
        printf("\tTranspose En\t%f s\t%d bytes\n", t_transpose, memory_transpose);
        printf("\tReciving NeN\t%f s\t%d bytes\n", t_recieve, memory_recieve);
        printf("\tFilling A\t%f s\t%d bytes\n", t_fill_A, memory_fill_A);
        printf("\tFilling b\t%f s\t%d bytes\n", t_fill_b, memory_fill_b);
        printf("\tSolving Ax = b\t%f s\t%d bytes\n", t_solve, std::stoi(t_m_solve[3]));
        printf("\t-- Find diag A\t%f s\t\n", std::stof(t_m_solve[0]));
        printf("\t--- SGM method\t%f s\t\n", std::stof(t_m_solve[1]));
        printf("\t-- Compute res\t%f s\t\n", std::stof(t_m_solve[2]));
        return 0;
    }

}
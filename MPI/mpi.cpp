#include <mpi.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <unistd.h>
#include <set>
#include <map>
#include <omp.h>
#include <algorithm>
#include <cassert>

#include "matrix.h"
#include "matvecop.h"
// -----------------------------------------------------------------------------------------------------------
// Global data
// -----------------------------------------------------------------------------------------------------------
int mpi_initialized = 0; // MPI initialization flag
int MyID = 0; // process ID
int NumProc = 1; // number of processes 
int MASTER_ID = 0; // master process ID
MPI_Comm MCW = MPI_COMM_WORLD; // default communicator

// -----------------------------------------------------------------------------------------------------------
// Utility functions 
// -----------------------------------------------------------------------------------------------------------

#define LINE_SEPARATOR  "-------------------------------------------------------------------------------\n" 
#define crash(...) exit(Crash( __VA_ARGS__)) // via exit macro, so a static analyzer knows that it is an exit point
static int Crash(const char *fmt,...){ // termination of the program in case of error
    va_list ap;
    if(mpi_initialized) fprintf(stderr,"\nEpic fail: MyID = %d\n",MyID);
    else fprintf(stderr,"\nEpic fail: \n");

    va_start(ap,fmt);
    vfprintf(stderr,fmt,ap);
    va_end(ap);

    fprintf(stderr,"\n");
    fflush(stderr);
    
    if(mpi_initialized) MPI_Abort(MPI_COMM_WORLD,-1);
    return -1;
}


// wrapper for MPI barrier
static inline void barrier(){
    if(MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS) crash("Base lib barrier: MPI_Barrier failed! \n");
}

// Write to stdout from Master process only
// Write to stdout from Master process only
int printf0(const char *fmt,...){ 
    int r = 0;
    va_list ap;
    if(MyID==MASTER_ID){
        va_start(ap,fmt);  
        r=vfprintf(stdout,fmt,ap);  
        va_end(ap);
    }
    fflush(stdout);
    return(r);
}

int printf_my_id(int my_id, const char *fmt,...) {
    int r = 0;
    va_list ap;
    if(MyID==my_id){
        va_start(ap,fmt);  
        r=vfprintf(stdout,fmt,ap);  
        va_end(ap);
    }
    fflush(stdout);
    return(r);
}

// Debug synchronous printf - Write to stdout + flush + barrier
int pprintf(const char *fmt,...){ 
    int r = 0;
    fflush(stdout);
    barrier();
    for(int p=0; p<NumProc; p++){
        barrier();
        if(MyID != p) continue; 
        fprintf(stdout,"%3d: ",MyID);
        va_list ap;
        //stdout
        va_start(ap,fmt);  
        r=vfprintf(stdout,fmt,ap);
        va_end(ap);
        fflush(stdout);
    }
    fflush(stdout);
    barrier();
    return(r);
}

// exchange

struct tCommScheme
{
    std::vector<int> Send; // список ячеек на отправку по всем соседям
    std::vector<int> Recv; // список ячеек на прием по всем соседям
    std::vector<int> SendOffset; // смещения списков по каждому соседу на отправку
    std::vector<int> RecvOffset; // смещения списков по каждому соседу на прием
    std::vector<int> Neigbours; // номера процессов соседей
    MPI_Comm MyComm; // коммуникатор для данной группы (MPI_COMM_WORLD)
    int B; // число соседей

    const int GetNumOfNeighbours() {return B;} // число соседей
    std::vector <int>& GetSendList() {return Send;} // список ячеек на отправку по всем соседям
    std::vector <int>& GetRecvList() {return Recv;} // список ячеек на прием по всем соседям
    std::vector <int>& GetSendOffset() {return SendOffset;} // смещения списков по каждому соседу на отправку
    std::vector <int>& GetRecvOffset() {return RecvOffset;} // смещения списков по каждому соседу на прием
    std::vector <int>& GetListOfNeighbours() {return Neigbours;}  // номера процессов соседей
};

template <typename VarType /* тип значений */>
void Update(std::vector<VarType> &V, // Входной массив значений в вершинах/ячейках, который надо обновить
            tCommScheme &CS/*какая-то структура, описывающая схему обменов*/){
                
                
    const int B = CS.GetNumOfNeighbours(); // число соседей
    if(B==0) return; // нет соседей - нет проблем

    // приведем все к POD типам и неймингу, как было в тексте выше
    std::vector<int> Send = CS.GetSendList(); // список ячеек на отправку по всем соседям
    std::vector<int> Recv = CS.GetRecvList(); // список ячеек на прием по всем соседям
    std::vector<int> SendOffset = CS.GetSendOffset(); // смещения списков по каждому соседу на отправку
    std::vector<int> RecvOffset = CS.GetRecvOffset(); // смещения списков по каждому соседу на прием
    std::vector<int> Neighbours = CS.GetListOfNeighbours(); // номера процессов соседей
    
    int sendCount=SendOffset[B]; // размер общего списка на отправку по всем соседям
    int recvCount=RecvOffset[B]; // размер общего списка на прием по всем соседям

    // MPI данные - сделаем статиками, поскольку это высокочастотная функция,
    // чтобы каждый раз не реаллокать (так делать вовсе не обязательно).
    static std::vector<double> SENDBUF, RECVBUF; // буферы на отправку и прием по всем соседям
    static std::vector<MPI_Request> REQ; // реквесты для неблокирующих обменов
    static std::vector<MPI_Status> STS; // статусы для неблокирующих обменов

    // ресайзим, если надо
    if(2*B > (int)REQ.size()){ REQ.resize(2*B); STS.resize(2*B); }
    if(sendCount>(int)SENDBUF.size()) SENDBUF.resize(sendCount);
    if(recvCount>(int)RECVBUF.size()) RECVBUF.resize(recvCount);

    int nreq=0; // сквозной счетчик реквестов сообщений

    // инициируем получение сообщений
    for(int p=0; p<B; p++){
        int SZ = (RecvOffset[p+1]-RecvOffset[p]);//*sizeof(VarType); // размер сообщения
        if(SZ<=0) continue; // если нечего слать - пропускаем соседа
        int NB_ID = Neighbours[p]; // узнаем номер процесса данного соседа
        int mpires = MPI_Irecv(&RECVBUF[RecvOffset[p]],//*sizeof(VarType)],
                                //SZ, MPI_CHAR, NB_ID, 0, MCW, &(REQ[nreq]));
                                SZ, MPI_DOUBLE, NB_ID, 0, MCW, &(REQ[nreq]));
        assert(mpires == MPI_SUCCESS);
        //~ ASSERT(mpires==MPI_SUCCESS, "MPI_Irecv failed");
        //ASSERT - какой-то макрос проверки-авоста, замените на ваш способ проверки
        nreq++;
    }

    // пакуем данные с интерфейса по единому списку сразу по всем соседям
    #pragma omp parallel for // в параллельном режиме с целью ускорения (К.О.)
    for(int i=0; i<sendCount; ++i) SENDBUF[i] = V[Send[i]/*номер ячейки на отправку*/];

    // инициируем отправку сообщений
    for(int p=0; p<B; p++){
        int SZ =(SendOffset[p+1]-SendOffset[p]);//*sizeof(VarType); // размер сообщения
        if(SZ<=0) continue; // если нечего принимать - пропускаем соседа
        int NB_ID = Neighbours[p]; // узнаем номер процесса данного соседа
        int mpires = MPI_Isend(&SENDBUF[SendOffset[p]],//*sizeof(VarType)],
                                //SZ, MPI_CHAR, NB_ID, 0, MCW, &(REQ[nreq]));
                                SZ, MPI_DOUBLE, NB_ID, 0, MCW, &(REQ[nreq]));
        assert(mpires == MPI_SUCCESS);
        //~ ASSERT(mpires==MPI_SUCCESS, "MPI_Isend failed");
        nreq++;
    }

    if(nreq>0){ // ждем завершения всех обменов
        int mpires = MPI_Waitall(nreq, &REQ[0], &STS[0]);
        assert(mpires == MPI_SUCCESS);
        //~ ASSERT(mpires==MPI_SUCCESS, "MPI_Waitall failed");
    }

    // разбираем данные с гало ячеек по единому списку сразу по всем соседям
    #pragma omp parallel for
    for(int i=0; i<recvCount; ++i) V[Recv[i]/*номер ячейки на прием*/] = RECVBUF[i];
}




static void show_usage(
    std::string name
) {
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
        printf0("no arguments were passed to the function\n help: --help\n");
        return 1;
    } else if (argc == 2) { // check help
        if (strcmp(argv[1], "--help") == 0) {
            if (MyID == MASTER_ID) {
                show_usage(argv[0]+2);
            }
            return 1;
        } else {
            printf0("Incorrect arguments\n help: --help\n");
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
                    printf0("Incorrect value of argument %s\n help: --help\n", argv[i]);
                    return 1;
                }
            }

            int i = 9;
            while(i < argc) {
                if (argc == (i + 1)) {
                    if (strcmp(argv[i],"--debug") == 0) {
                        debug = 1;
                    } else if (strcmp(argv[i],"-T") == 0){
                        printf0("the parameter -T value was not passed, the default value was used\n help: --help\n\n");
                    } else if (strcmp(argv[i],"-eps") == 0){
                        printf0("the parameter -eps value was not passed, the default value was used\n help: --help\n\n");
                    } else {
                        printf0("unknown last argument\n help: --help\n");
                        return 1;
                    }

                    i++;
                } else {
                    if (strcmp(argv[i],"-T") == 0){
                        if (atoi(argv[i+1]) > 0 && atoi(argv[i+1]) <= 8) {
                            T = atoi(argv[i+1]);
                        } else {
                            printf0("Incorrect value of argument -T\n help: --help\n");
                            return 1;
                        }
                    } else if (strcmp(argv[i],"-eps") == 0) {
                        double e = atof(argv[i+1]);
                        if (e > 0.0) {
                            eps = e;
                        } else {
                            printf0("Incorrect value of argument -eps\n help: --help\n");
                            return 1;
                        }
                    } else {
                        printf0("Incorrect argument %s\n help: --help\n", argv[i]);
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
                printf0("Argument -Nx must be greater then 1\n help: --help\n");
                return 1;
            } else if (Ny < 2) {
                printf0("Argument -Ny must be greater then 1\n help: --help\n");
                return 1;
            } else if (k3 + k4 < 1) {
                printf0("arguments -k3 and -k4 cannot be equal to 0 at the same time\n help: --help\n");
            } else {
                if (debug && (Nx > 5 || Ny > 5)) {
                    printf0("The debug mode is only for -Nx and -Ny values less than 5\n help: --help\n\n");
                    debug = 0;
                }
                return 0;
            }
        } else {
            printf0("Incorrect names of arguments\n help: --help\n");
            return 1;
        }
    } else {
        printf0("Incorrect number of arguments\n help: --help\n");
        return 1;
    }
    return 1;
}

struct data
{
    int memory;
    std::vector<double> times;
};


data solver(
    int T, //number of threads
    const CSRMatrix& A, //matrix of system, size N*N
    int begin_halo, // number of local elements
    const std::vector<double>& b, //vector of right part, size N
    double eps, // stopping criteria
    int maxit, // maximum number of iterations
    std::vector<double>& x, //vector for writing solution, size N
    int& n, //number of operations performed
    double& res, //L2 discrepancy rate
    tCommScheme &CS /*структура, описывающая схему обменов*/
) { 
    data D;

    int N = b.size();
    std::vector<double> antidiag = std::vector<double>(N);
    
    
    MPI_Barrier(CS.MyComm);
    double t_diag = MPI_Wtime();
    find_antidiag(A, antidiag);
    MPI_Barrier(CS.MyComm);
    D.times.push_back(MPI_Wtime() - t_diag);
    
    
    Update(antidiag, CS);
    
    

    std::vector<double> r_k = b;
    x = std::vector<double>(N, 0.0);

    std::vector<double> p_k = std::vector<double>(N);
    std::vector<double> z_k = std::vector<double>(N);
    std::vector<double> q_k = std::vector<double>(N);

    int k = 0;
    double rho_old = 0;
    double rho_new = 0;

    MPI_Barrier(CS.MyComm);
    double t_solve = MPI_Wtime();
    do
    {
        
        k++;
        Update(r_k, CS);
        matmul(r_k, antidiag, z_k);
        

        rho_old = rho_new;
        dot(r_k, z_k, begin_halo, rho_new);

        MPI_Allreduce(MPI_IN_PLACE, &rho_new, 1, MPI_DOUBLE, MPI_SUM, CS.MyComm);

        printf0("%f\n", rho_new);
            
        
        if (k == 1) {
            p_k = z_k;
        } else {
            double beta_k = rho_new / rho_old;
            
            axpy(beta_k, p_k, z_k, begin_halo, p_k);
        }

        Update(p_k, CS);

        SpMV(A, p_k, q_k);
        
        double pq;
        dot(p_k, q_k, begin_halo, pq);
        MPI_Allreduce(MPI_IN_PLACE, &pq, 1, MPI_DOUBLE, MPI_SUM, CS.MyComm);
        double alfa = rho_new / pq;
        
        axpy(alfa, p_k, x, begin_halo, x);
        axpy(-alfa, q_k, r_k, begin_halo, r_k);
    }
    while ((rho_new > eps * eps) && (k < maxit));
    MPI_Barrier(CS.MyComm);
    D.times.push_back(MPI_Wtime() - t_solve);

    n = k;
    
    std::vector<double> r = std::vector<double>(N);

    Update(x, CS);

    MPI_Barrier(CS.MyComm);
    double t_res = MPI_Wtime();
    SpMV(A, x, r);
    axpy(-1.0, b, r, begin_halo, r);
    dot(r, r, begin_halo, res);
    MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_DOUBLE, MPI_SUM, CS.MyComm);

    res = sqrt(res);
    MPI_Barrier(CS.MyComm);
    D.times.push_back(MPI_Wtime() - t_res);

    

    D.memory = 5 * N * sizeof(double);
    MPI_Allreduce(MPI_IN_PLACE, &D.memory, 1, MPI_DOUBLE, MPI_SUM, CS.MyComm);
    return D;
}

// -----------------------------------------------------------------------------------------------------------
// Entry point 
// -----------------------------------------------------------------------------------------------------------
int main(int argc, char **argv){

    int mpi_res; 

    mpi_res = MPI_Init(&argc, &argv);
    if(mpi_res!= MPI_SUCCESS) 
        crash("MPI_Init failed (code %d)\n", mpi_res);

    mpi_res = MPI_Comm_rank(MCW,&MyID); 
    if(mpi_res!= MPI_SUCCESS) 
        crash("MPI_Comm_rank failed (code %d)\n", mpi_res);

    mpi_res = MPI_Comm_size(MCW,&NumProc);
    if(mpi_res!= MPI_SUCCESS) 
        crash("MPI_Comm_size failed (code %d)\n", mpi_res);

    char proc_name[128]; 
    proc_name[0]=0;
    int ll; 
    mpi_res= MPI_Get_processor_name(proc_name, &ll);
    if(mpi_res!= MPI_SUCCESS) 
        crash("MPI_Get_processor_name failed (code %d)\n", mpi_res);

    mpi_initialized = 1;



    int Nx, Ny, k3, k4, p;
    int debug = 0;
    double eps = 1e-3;
    int maxit = 1000;
    int T = 1;

    p = int(std::sqrt(NumProc));

    if(!parcing_args(argc, argv, Nx, Ny, k3, k4, debug, eps, T)) {
        omp_set_num_threads(T);

        printf0("Generating EN matrix...\n");
        barrier();
        double t_gener = MPI_Wtime();
        CSRMatrix_rltn EN;
        std::vector<int> L2G;
        Lengths Len;

        SR SendRec(p);

        int memory_gener = generator_EN(EN, Ny, Nx, k3, k4, p, MyID, L2G, SendRec, Len);
        barrier();
        t_gener = MPI_Wtime() - t_gener;

        for (int i = 0; i < SendRec.send.size(); ++i) {
            std::sort(SendRec.send[i].begin(), SendRec.send[i].end());
        }

        tCommScheme CS; // заполним все для обменов
        CS.B = 0;
        CS.RecvOffset.push_back(0);
        CS.SendOffset.push_back(0);
        for (int i = 0; i < SendRec.neighbours.size(); ++i) {
            if (SendRec.neighbours[i] != 0) {
                CS.B++;
                CS.Neigbours.push_back(i);
                CS.RecvOffset.push_back(CS.RecvOffset.back() + SendRec.neighbours[i]);
                CS.Send.insert(CS.Send.begin(), SendRec.send[i].begin(), SendRec.send[i].end());
                CS.SendOffset.push_back(CS.SendOffset.back()+SendRec.send[i].size());
            }
        }
        int begin_halo = Len.len_inter + Len.len_self;
        CS.Recv = std::vector<int>(EN.m - begin_halo);
        for (int i = 0; i < CS.Recv.size(); ++i) {
            CS.Recv[i] = begin_halo + i;
        } 
        CS.MyComm = MCW;


        printf0("Transpose EN matrix, receiving NE matrix...\n");
        barrier();
        double t_transpose = MPI_Wtime();
        CSRMatrix_rltn NE;
        int memory_transpose = sparse_transpose(EN, NE);
        barrier();
        t_transpose = MPI_Wtime() - t_transpose;

        printf0("Receiving NeN matrix...\n");
        barrier();
        double t_recieve = MPI_Wtime();
        CSRMatrix_rltn NeN;
        int memory_recieve = generator_NeN(NE, EN, NeN);
        barrier();
        t_recieve = MPI_Wtime() - t_recieve;

        printf0("Filling the matrix A...\n");
        barrier();
        double t_fill_A = MPI_Wtime();
        CSRMatrix A(NeN);
        A.values = std::vector<double>(A.nz);

        int memory_fill_A = fill_Matrix(A, L2G, T);
        barrier();
        t_fill_A = MPI_Wtime() - t_fill_A;



        printf0("Filling the vector b...\n");
        barrier();
        double t_fill_b = MPI_Wtime();
        std::vector<double> b(A.n);
        #pragma omp parallel for
        for (int i = 0; i < A.n; ++i) {
            b[i] = sin(L2G[i]);
        }
        barrier();
        t_fill_b = MPI_Wtime() - t_fill_b;
        int memory_fill_b = A.n * sizeof(double);

        Update(b, CS);
        
        std::vector<double> x;
        double res;
        int n;
        
        printf0("Solving system Ax = b ...\n");
        barrier();
        double t_solve = MPI_Wtime();
        data D = solver(T, A, begin_halo, b, eps, maxit, x, n, res, CS);
        barrier();
        t_solve = MPI_Wtime() - t_solve;


        printf0(LINE_SEPARATOR);
        printf0("the process was completed in %d iterations with precision %f\n", n, res);

        printf0("\tProcess\t\tTime\t\tMemory\n");
        printf0("\tGenerating En\t%f s\t%d bytes\n", t_gener, memory_gener);
        printf0("\tTranspose En\t%f s\t%d bytes\n", t_transpose, memory_transpose);
        printf0("\tReciving NeN\t%f s\t%d bytes\n", t_recieve, memory_recieve);
        printf0("\tFilling A\t%f s\t%d bytes\n", t_fill_A, memory_fill_A);
        printf0("\tFilling b\t%f s\t%d bytes\n", t_fill_b, memory_fill_b);
        printf0("\tSolving Ax = b\t%f s\t%d bytes\n", t_solve, D.memory);
        printf0("\t-- Find diag A\t%f s\t\n", D.times[0]);
        printf0("\t--- SGM method\t%f s\t\n", D.times[1]);
        printf0("\t-- Compute res\t%f s\t\n", D.times[2]);
        
        MPI_Finalize();
    }
}



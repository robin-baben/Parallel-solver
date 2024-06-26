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
#include <CL/cl.h>

#include "matrix.h"

#define PLATFORM_NAME "NVIDIA"
#define REDUCTION_LWS 256 // размер рабочей группы для суммы
#define REDUCTION_ITEM 8 // сколько элементов будет суммировать один work-item (нить)

void ocl_init(
    const cl_int devID, // Номер нужного девайса
    const char *platformName, //Нужная платформа // "Intel(R) OpenCL" // "NVIDIA CUDA"
    cl_context &clContext, // OpenCL контекст
    cl_command_queue &clQueue, // OpenCL очередь команд
    cl_program &clProgram, // OpenCL программа
    cl_int &clErr 
) {
    {// Инициализация OpenCL 
        printf("OpenCL initialization\n");

        //PLATFORM
        // узнаем число платформ в системе
        cl_uint platformCount = 0;
        clErr = clGetPlatformIDs(0, 0, &platformCount);
        if(clErr != CL_SUCCESS){ printf("clGetPlatformIDs error %d\n", clErr); exit(-1); }
        if(platformCount <= 0){ printf("No platforms found\n"); exit(-1); }
        printf("clGetPlatformIDs: %d platforms\n", platformCount);

        // запрашиваем список платформ
        cl_platform_id *platformList = new cl_platform_id[platformCount];
        clErr = clGetPlatformIDs(platformCount, platformList, 0);
        if(clErr != CL_SUCCESS){ printf("clGetPlatformIDs error %d\n", clErr); exit(-1); }

        // ищем нужную платформу
        #define STR_SIZE 1024 // размерчик буфера для названия платформ
        char nameBuf[STR_SIZE] = "\0"; // буфер для названия платформы
        cl_int platform_id = -1;
        for(cl_uint i = 0; i<platformCount; i++){
            clErr = clGetPlatformInfo(platformList[i], CL_PLATFORM_NAME, STR_SIZE, nameBuf, 0);
            if(clErr != CL_SUCCESS){ printf("clGetPlatformInfo error %d\n", clErr); exit(-1); }
            printf("  Platform %d: %s\n", i, nameBuf);
            if(strstr(nameBuf, platformName)) { platform_id = i; }// found
        }
        if(platform_id<0){ printf("Can't find platform\n"); exit(-1); }
        printf("Platform %d selected\n", platform_id);

        // DEVICE
        // узнаем число девайсов у выбранной платформы
        int deviceCount = 0;
        clErr = clGetDeviceIDs(platformList[platform_id], CL_DEVICE_TYPE_ALL,
                               0, NULL, (cl_uint *)&deviceCount);
        if(clErr != CL_SUCCESS){ printf("clGetDeviceIDs error %d\n", clErr); exit(-1); }
        printf("%d devices found\n", deviceCount);
        if(devID >= deviceCount){ printf("Wrong device selected: %d!\n", devID); exit(-1); }

        // запрашиваем список девайсов у выбранной платформы
        cl_device_id *deviceList = new cl_device_id[deviceCount]; // list of devices
        clErr = clGetDeviceIDs(platformList[platform_id], CL_DEVICE_TYPE_ALL,
            (cl_uint)deviceCount, deviceList, NULL);
        if(clErr != CL_SUCCESS){ printf("clGetDeviceIDs error %d\n", clErr); exit(-1); }
        delete[] platformList; // больше не нужно

        // печатаем девайсы платформы
        for(int i = 0; i<deviceCount; i++){
            clErr = clGetDeviceInfo(deviceList[i], CL_DEVICE_NAME, STR_SIZE, nameBuf, 0);
            if(clErr != CL_SUCCESS){ printf("clGetDeviceInfo error %d\n", clErr); exit(-1); }
            printf("  Device %d: %s \n", i, nameBuf);
        }

        // CONTEXT
        clContext = clCreateContext(NULL, 1, &deviceList[devID], 0, 0, &clErr);
        if(clErr != CL_SUCCESS){ printf("clCreateContext error %d\n", clErr);  exit(-1); }

        // COMMAND QUEUE
        clQueue = clCreateCommandQueueWithProperties(clContext, deviceList[devID], 0, &clErr);
        if(clErr != CL_SUCCESS){ printf("clCreateCommandQueue %d\n", clErr);  exit(-1); }

        // PROGRAM
        const char *cPathAndName = "kernel.cl";  // файл с исходным кодом кернелов 
        printf("Loading program from %s\n", cPathAndName);

        // сюда можно напихать каких нужно дефайнов, чтобы они подставились в программу
        const char *cDefines = " /* add your defines */ ";

        char * cSourceCL = NULL; // буфер для исходного кода 
        { // читаем файл 
            FILE *f = fopen(cPathAndName, "rb");
            if(!f){ printf("Can't open program file %s!\n", cPathAndName); exit(-1); }
            fseek(f, 0, SEEK_END); // считаем размер
            size_t fileSize = ftell(f);
            rewind(f);
            size_t codeSize = fileSize + strlen(cDefines); // считаем общий размер: код + дефайны 
            cSourceCL = new char[codeSize + 1/*zero-terminated*/]; // выделяем буфер
            memcpy(cSourceCL, cDefines, strlen(cDefines)); // подставляем дефайны
            size_t nd = fread(cSourceCL+strlen(cDefines), 1, fileSize, f); // читаем
            if(nd != fileSize){ printf("Failed to read program %s!\n", cPathAndName); exit(-1); }
            cSourceCL[codeSize] = 0; // заканчиваем строку нулем!
        }
        if(cSourceCL == NULL){ printf("Can't get program from %s!\n", cPathAndName); exit(-1); }

        // сдаем исходники в OpenCL
        size_t szKernelLength = strlen(cSourceCL);
        clProgram = clCreateProgramWithSource(clContext, 1, (const char **)&cSourceCL,
                                              &szKernelLength, &clErr);
        if(clErr != CL_SUCCESS){ printf("clCreateProgramWithSource error %d\n", clErr); exit(-1); }

        // компилим кернел-программу
        printf("clBuildProgram... ");
        clErr = clBuildProgram(clProgram, 0, NULL, "-cl-mad-enable", NULL, NULL);
        printf("done\n");

        // запрашиваем размер лога компиляции
        size_t LOG_S = 0;
        clErr = clGetProgramBuildInfo(clProgram, deviceList[devID], CL_PROGRAM_BUILD_LOG,
                                      0, NULL, (size_t*)&LOG_S);
        if(clErr != CL_SUCCESS){ printf("clGetProgramBuildInfo error %d\n", clErr); exit(-1); }
        if(LOG_S>8){ // если там не пусто - печатаем лог
            char *programLog = new char[LOG_S + 1];
            clErr = clGetProgramBuildInfo(clProgram, deviceList[devID], CL_PROGRAM_BUILD_LOG,
                                          LOG_S, programLog, 0);
            if(clErr != CL_SUCCESS){ printf("clGetProgramBuildInfo error %d\n", clErr); exit(-1); }
            printf("%s\n", programLog);
            delete[] programLog;
        }
        if(clErr != CL_SUCCESS){ printf("Compilation failed with error: %d\n", clErr); exit(-1); }
        delete[] cSourceCL;
    }
}


struct data
{
    int memory;
    std::vector<double> times;
};

data solver(
    int T, //number of threads
    const CSRMatrix& A, //matrix of system, size N*N
    const std::vector<double>& b, //vector of right part, size N
    double eps, // stopping criteria
    int maxit, // maximum number of iterations
    std::vector<double>& x, //vector for writing solution, size N
    int& n, //number of operations performed
    double& res //L2 discrepancy rate
) { 
    data D;

    int N = b.size();

    double init_time = omp_get_wtime();
    // Инициализация OpenCL
    const cl_int devID = 0; // Номер нужного девайса
    const char *platformName = PLATFORM_NAME; //Нужная платформа // "Intel(R) OpenCL" // "NVIDIA CUDA"
    // OpenCL переменные
    cl_context clContext; // OpenCL контекст
    cl_command_queue clQueue; // OpenCL очередь команд
    cl_program clProgram; // OpenCL программа
    cl_int clErr; // код возврата из OpenCL функций
    ocl_init(devID, platformName, clContext, clQueue, clProgram, clErr);

    // KERNELS
    cl_kernel knlSPMV_CSR;
    knlSPMV_CSR = clCreateKernel(clProgram, "knlSPMV_CSR", &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateKernel knlSPMV_CSR error: %d\n",clErr); exit(1); }

    cl_kernel knlDOT;
    knlDOT = clCreateKernel(clProgram, "knlDOT", &clErr); // создаем кернел
    if(clErr != CL_SUCCESS){ printf("clCreateKernel knlSUM error: %d\n",clErr); exit(1); }

    cl_kernel knlAXPBY;
    knlAXPBY = clCreateKernel(clProgram, "knlAXPBY", &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateKernel knlAXPBY error: %d\n",clErr); exit(1); }

    cl_kernel knlMATMUL;
    knlMATMUL = clCreateKernel(clProgram, "knlMATMUL", &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateKernel knlMATMUL error: %d\n",clErr); exit(1); }

    cl_kernel knlAssig;
    knlAssig = clCreateKernel(clProgram, "knlAssig", &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateKernel knlAssig error: %d\n",clErr); exit(1); }

    
    //Buffers
    printf("Creating opencl buffers\n");

    // для предобуславливателя
    cl_mem clAntidiag = clCreateBuffer(clContext, CL_MEM_READ_WRITE, N*sizeof(double), NULL, &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateBuffer clAntidiag error %d\n",clErr); exit(1); }

    // для rowPtr
    cl_mem clRowPtr = clCreateBuffer(clContext, CL_MEM_READ_WRITE, A.rowPtr.size()*sizeof(int), NULL, &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateBuffer clRowPtr error %d\n",clErr); exit(1); }

    // для colInd
    cl_mem clColInd = clCreateBuffer(clContext, CL_MEM_READ_WRITE, A.colIndex.size()*sizeof(int), NULL, &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateBuffer clColIndr %d\n",clErr); exit(1); }

    // для значений матрицы(A.values)
    cl_mem clAvalues = clCreateBuffer(clContext, CL_MEM_READ_WRITE, A.colIndex.size()*sizeof(double), NULL, &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateBuffer clAvalues error %d\n",clErr); exit(1); }

    // для вектора r
    cl_mem clR = clCreateBuffer(clContext, CL_MEM_READ_WRITE, N*sizeof(double),NULL, &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateBuffer clR error %d\n",clErr); exit(1); }

    // для вектора z
    cl_mem clZ = clCreateBuffer(clContext, CL_MEM_READ_WRITE, N*sizeof(double),NULL, &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateBuffer clZ error %d\n",clErr); exit(1); }

    // для вектора p
    cl_mem clP = clCreateBuffer(clContext, CL_MEM_READ_WRITE, N*sizeof(double),NULL, &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateBuffer clP error %d\n",clErr); exit(1); }

    // для вектора q
    cl_mem clQ = clCreateBuffer(clContext, CL_MEM_READ_WRITE, N*sizeof(double),NULL, &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateBuffer clQ error %d\n",clErr); exit(1); }

    // для вектора x
    cl_mem clX = clCreateBuffer(clContext, CL_MEM_READ_WRITE, N*sizeof(double),NULL, &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateBuffer clX error %d\n",clErr); exit(1); }

    // для результата частичного скалярного произведения
    // размер буфера частичных сумм рабочих групп
    int REDUCTION_BUFSIZE = ((N/REDUCTION_ITEM)/REDUCTION_LWS) + ((N/REDUCTION_ITEM)%REDUCTION_LWS>0);
    // буфер под частичные суммы рабочих групп
    cl_mem clSum = clCreateBuffer(clContext, CL_MEM_READ_WRITE, REDUCTION_BUFSIZE*sizeof(double), NULL, &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateBuffer clSum error %d\n",clErr); exit(1); }
    // Выделим для частичных сумм место на хосте
    std::vector <double> Sum(REDUCTION_BUFSIZE, 0);

    printf("Init done\n");
    D.times.push_back(omp_get_wtime()-init_time);


    double t_diag = omp_get_wtime();
    std::vector<double> antidiag = std::vector<double>(N);
    find_antidiag(A, antidiag);
    D.times.push_back(omp_get_wtime()-t_diag);

    std::vector<double> r_k = b;
    x = std::vector<double>(N, 0.0);

    
    double time_copy = omp_get_wtime();
    // копируем вектора на девайс
    clErr = clEnqueueWriteBuffer(clQueue, clX, CL_TRUE, 0, N*sizeof(double), &x[0], 0, NULL, NULL);
    if(clErr != CL_SUCCESS){ printf("clEnqueueWriteBuffer clX error %d\n", clErr); exit(1); }

    clErr = clEnqueueWriteBuffer(clQueue, clR, CL_TRUE, 0, N*sizeof(double), &r_k[0], 0, NULL, NULL);
    if(clErr != CL_SUCCESS){ printf("clEnqueueWriteBuffer clR error %d\n", clErr); exit(1); }

    clErr = clEnqueueWriteBuffer(clQueue, clAntidiag, CL_TRUE, 0, N*sizeof(double), &antidiag[0], 0, NULL, NULL);
    if(clErr != CL_SUCCESS){ printf("clEnqueueWriteBuffer clAntidiag error %d\n", clErr); exit(1); }

    clErr = clEnqueueWriteBuffer(clQueue, clRowPtr, CL_TRUE, 0, A.rowPtr.size()*sizeof(int), &A.rowPtr[0], 0, NULL, NULL);
    if(clErr != CL_SUCCESS){ printf("clEnqueueWriteBuffer clRowPtr error %d\n", clErr); exit(1); }

    clErr = clEnqueueWriteBuffer(clQueue, clColInd, CL_TRUE, 0, A.colIndex.size()*sizeof(int), &A.colIndex[0], 0, NULL, NULL);
    if(clErr != CL_SUCCESS){ printf("clEnqueueWriteBuffer clColInd error %d\n", clErr); exit(1); }

    clErr = clEnqueueWriteBuffer(clQueue, clAvalues, CL_TRUE, 0, A.values.size()*sizeof(double), &A.values[0], 0, NULL, NULL);
    if(clErr != CL_SUCCESS){ printf("clEnqueueWriteBuffer clColInd error %d\n", clErr); exit(1); }
    D.times.push_back(omp_get_wtime()-time_copy);

    int k = 0;
    double rho_old = 0;
    double rho_new = 0;

    double time_solve = omp_get_wtime();

    do
    {
        ++k;
        // Запуск kernelSPMV с предобуславливателем...
        // ...выставляем параметры запуска
        size_t lws = 128; // размер рабочей группы
        size_t gws = N; // общее число заданий
        if(gws%lws>0) gws += lws-gws%lws; // делаем кратное lws
        // ...выставляем аргументы кернелу
        clSetKernelArg(knlMATMUL, 0, sizeof(int), &N);
        clSetKernelArg(knlMATMUL, 1, sizeof(cl_mem), &clAntidiag);
        clSetKernelArg(knlMATMUL, 2, sizeof(cl_mem), &clR);
        clSetKernelArg(knlMATMUL, 3, sizeof(cl_mem), &clZ);
        // ...отправляем на исполнение
        clErr= clEnqueueNDRangeKernel(clQueue, knlMATMUL, 1, NULL, &gws, &lws, 0, NULL, NULL);
        if(clErr != CL_SUCCESS){ printf("clEnqueueNDRangeKernel error %d\n",clErr); exit(1); }

        // Запуск кернела DOT...
        // ...выставляем параметры запуска
        lws = REDUCTION_LWS; // размер рабочей группы
        gws = (N/REDUCTION_ITEM); // общее число заданий
        if(gws%lws>0) gws += lws-gws%lws; // делаем кратное lws
        // ...выставляем аргументы кернелу
        clSetKernelArg(knlDOT, 0, sizeof(int), &N);
        clSetKernelArg(knlDOT, 1, sizeof(cl_mem), &clR);
        clSetKernelArg(knlDOT, 2, sizeof(cl_mem), &clZ);
        clSetKernelArg(knlDOT, 3, sizeof(cl_mem), &clSum);

        clFinish(clQueue); // ждем завершения работы предыдущего ядра
        // ...отправляем на исполнение
        clErr= clEnqueueNDRangeKernel(clQueue, knlDOT, 1, NULL, &gws, &lws, 0, NULL, NULL);
        if(clErr != CL_SUCCESS){ printf("clEnqueueNDRangeKernel error %d\n",clErr); exit(1); }
        clFinish(clQueue); // ждем завершения
        // ...забираем результат с девайса
        clErr = clEnqueueReadBuffer(clQueue, clSum, CL_TRUE, 0, REDUCTION_BUFSIZE*sizeof(double), &Sum[0], 0, NULL, NULL);
        if(clErr != CL_SUCCESS){printf("clEnqueueReadBuffer clSum error %d\n", clErr); exit(1);}
        clFinish(clQueue);
        // ...досуммируем результат на хосте, там осталось где-то 0.05% от общего объема работы
        rho_old = rho_new;
        rho_new = 0.0;
        #pragma omp parallel for reduction(+:rho_new)
        for(int i=0; i<REDUCTION_BUFSIZE; ++i) rho_new += Sum[i];

        printf("%f\n", rho_new);

        double a, b;
        if (k == 1) {           
            // Для копирования вектора используем кернел Assig
            lws = 128; // размер рабочей группы
            gws = N; // общее число заданий
            if(gws%lws>0) gws += lws-gws%lws; // делаем кратное lws
            // ...выставляем аргументы кернелу
            clSetKernelArg(knlAssig, 0, sizeof(int), &N);
            clSetKernelArg(knlAssig, 1, sizeof(cl_mem), &clZ);
            clSetKernelArg(knlAssig, 2, sizeof(cl_mem), &clP);
             // ...отправляем на исполнение
            clErr= clEnqueueNDRangeKernel(clQueue, knlAssig, 1, NULL, &gws, &lws, 0, NULL, NULL);
            if(clErr != CL_SUCCESS){ printf("clEnqueueNDRangeKernel error %d\n",clErr); exit(1); }
        } else {
            double beta = rho_new / rho_old; 
            //Запускаем кернел AXPBY
            lws = 128; // размер рабочей группы
            gws = N; // общее число заданий
            if(gws%lws>0) gws += lws-gws%lws; // делаем кратное lws
            // ...выставляем аргументы кернелу
            clSetKernelArg(knlAXPBY, 0, sizeof(int), &N);
            clSetKernelArg(knlAXPBY, 1, sizeof(cl_mem), &clZ);
            clSetKernelArg(knlAXPBY, 2, sizeof(cl_mem), &clP);
            clSetKernelArg(knlAXPBY, 3, sizeof(cl_mem), &clP);
            a = 1.;
            clSetKernelArg(knlAXPBY, 4, sizeof(double), &a);
            clSetKernelArg(knlAXPBY, 5, sizeof(double), &beta);
            // ...отправляем на исполнение
            clErr= clEnqueueNDRangeKernel(clQueue, knlAXPBY, 1, NULL, &gws, &lws, 0, NULL, NULL);
            if(clErr != CL_SUCCESS){ printf("clEnqueueNDRangeKernel error %d\n",clErr); exit(1); }
        }
        // Запуск кернела SPMV с матрицей A
        lws = 128; // размер рабочей группы
        gws = N; // общее число заданий
        if(gws%lws>0) gws += lws-gws%lws; // делаем кратное lws
        // ...выставляем аргументы кернелу
        clSetKernelArg(knlSPMV_CSR, 0, sizeof(int), &N);
        clSetKernelArg(knlSPMV_CSR, 1, sizeof(cl_mem), &clAvalues);
        clSetKernelArg(knlSPMV_CSR, 2, sizeof(cl_mem), &clRowPtr);
        clSetKernelArg(knlSPMV_CSR, 3, sizeof(cl_mem), &clColInd);
        clSetKernelArg(knlSPMV_CSR, 4, sizeof(cl_mem), &clP);
        clSetKernelArg(knlSPMV_CSR, 5, sizeof(cl_mem), &clQ);

        clFinish(clQueue); // ждем завершения предыдущего ядра
        // ...отправляем на исполнение
        clErr= clEnqueueNDRangeKernel(clQueue, knlSPMV_CSR, 1, NULL, &gws, &lws, 0, NULL, NULL);
        if(clErr != CL_SUCCESS){ printf("clEnqueueNDRangeKernel error %d\n",clErr); exit(1); }

        // Запуск кернела DOT...
        // ...выставляем параметры запуска
        lws = REDUCTION_LWS; // размер рабочей группы
        gws = (N/REDUCTION_ITEM); // общее число заданий
        if(gws%lws>0) gws += lws-gws%lws; // делаем кратное lws
        // ...выставляем аргументы кернелу
        clSetKernelArg(knlDOT, 0, sizeof(int), &N);
        clSetKernelArg(knlDOT, 1, sizeof(cl_mem), &clP);
        clSetKernelArg(knlDOT, 2, sizeof(cl_mem), &clQ);
        clSetKernelArg(knlDOT, 3, sizeof(cl_mem), &clSum);

        clFinish(clQueue); // ждем завершения предыдущего ядра
        // ...отправляем на исполнение
        clErr= clEnqueueNDRangeKernel(clQueue, knlDOT, 1, NULL, &gws, &lws, 0, NULL, NULL);
        if(clErr != CL_SUCCESS){ printf("clEnqueueNDRangeKernel error %d\n",clErr); exit(1); }
        clFinish(clQueue); // ждем завершения
        // ...забираем результат с девайса
        clErr = clEnqueueReadBuffer(clQueue, clSum, CL_TRUE, 0, REDUCTION_BUFSIZE*sizeof(double), &Sum[0], 0, NULL, NULL);
        if(clErr != CL_SUCCESS){printf("clEnqueueReadBuffer clSum error %d\n", clErr); exit(1);}
        clFinish(clQueue);
        // ...досуммируем результат на хосте, там осталось где-то 0.05% от общего объема работы
        double dot_p_q = 0.0;
        #pragma omp parallel for reduction(+:dot_p_q)
        for(int i=0; i<REDUCTION_BUFSIZE; ++i) dot_p_q += Sum[i];
        double alpha = rho_new / dot_p_q;  
        //Запускаем кернел AXPBY
        lws = 128; // размер рабочей группы
        gws = N; // общее число заданий
        if(gws%lws>0) gws += lws-gws%lws; // делаем кратное lws
        // ...выставляем аргументы кернелу
        clSetKernelArg(knlAXPBY, 0, sizeof(int), &N);
        clSetKernelArg(knlAXPBY, 1, sizeof(cl_mem), &clX);
        clSetKernelArg(knlAXPBY, 2, sizeof(cl_mem), &clP);
        clSetKernelArg(knlAXPBY, 3, sizeof(cl_mem), &clX);
        a = 1.;
        clSetKernelArg(knlAXPBY, 4, sizeof(double), &a);
        clSetKernelArg(knlAXPBY, 5, sizeof(double), &alpha);
        // ...отправляем на исполнение
        clErr= clEnqueueNDRangeKernel(clQueue, knlAXPBY, 1, NULL, &gws, &lws, 0, NULL, NULL);
        if(clErr != CL_SUCCESS){ printf("clEnqueueNDRangeKernel error %d\n",clErr); exit(1); }
        // ...выставляем аргументы кернелу
        clSetKernelArg(knlAXPBY, 0, sizeof(int), &N);
        clSetKernelArg(knlAXPBY, 1, sizeof(cl_mem), &clR);
        clSetKernelArg(knlAXPBY, 2, sizeof(cl_mem), &clQ);
        clSetKernelArg(knlAXPBY, 3, sizeof(cl_mem), &clR);
        a = 1.;
        clSetKernelArg(knlAXPBY, 4, sizeof(double), &a);
        b = -alpha;
        clSetKernelArg(knlAXPBY, 5, sizeof(double), &b);
        // ...отправляем на исполнение
        clErr= clEnqueueNDRangeKernel(clQueue, knlAXPBY, 1, NULL, &gws, &lws, 0, NULL, NULL);
        if(clErr != CL_SUCCESS){ printf("clEnqueueNDRangeKernel error %d\n",clErr); exit(1); }
        clFinish(clQueue); // ждем завершения
    }
    while ((rho_new > eps * eps) && (k < maxit));

    D.times.push_back(omp_get_wtime()-time_solve);
    

    
    n = k;
    res = sqrt(rho_new);
    

    D.memory = 5 * N * sizeof(double);

    // Забираем x с девайса
    clErr = clEnqueueReadBuffer(clQueue, clX, CL_TRUE, 0, N*sizeof(double), &x[0], 0, NULL, NULL);
    if(clErr != CL_SUCCESS){printf("clEnqueueReadBuffer clSum error %d\n", clErr); exit(1);}
    clFinish(clQueue);

    //Чистим
    clErr = clReleaseMemObject(clQ);
    if(clErr != CL_SUCCESS){ printf("clReleaseMemObject clQ error %d\n", clErr); exit(-1); }
    clErr = clReleaseMemObject(clP);
    if(clErr != CL_SUCCESS){ printf("clReleaseMemObject clP error %d\n", clErr); exit(-1); }
    clErr = clReleaseMemObject(clZ);
    if(clErr != CL_SUCCESS){ printf("clReleaseMemObject clZ error %d\n", clErr); exit(-1); }
    clErr = clReleaseMemObject(clR);
    if(clErr != CL_SUCCESS){ printf("clReleaseMemObject clR error %d\n", clErr); exit(-1); }
    clErr = clReleaseMemObject(clRowPtr);
    if(clErr != CL_SUCCESS){ printf("clReleaseMemObject clRowPtr error %d\n", clErr); exit(-1); }
    clErr = clReleaseMemObject(clColInd);
    if(clErr != CL_SUCCESS){ printf("clReleaseMemObject clColind error %d\n", clErr); exit(-1); }
    clErr = clReleaseMemObject(clAvalues);
    if(clErr != CL_SUCCESS){ printf("clReleaseMemObject clAvalues error %d\n", clErr); exit(-1); }
    clErr = clReleaseMemObject(clAntidiag);
    if(clErr != CL_SUCCESS){ printf("clReleaseMemObject clAntidiag error %d\n", clErr); exit(-1); }
    clErr = clReleaseMemObject(clX);
    if(clErr != CL_SUCCESS){ printf("clReleaseMemObject clX error %d\n", clErr); exit(-1); }
    clErr = clReleaseMemObject(clSum);
    if(clErr != CL_SUCCESS){ printf("clReleaseMemObject clSum error %d\n", clErr); exit(-1); }

    return D;
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
        int memory_gener = generator_EN(EN, Nx, Ny, k3, k4);
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
        int memory_transpose = sparse_transpose(EN, NE);
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
        int memory_recieve = generator_NeN(NE, EN, NeN);
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


        // Вот до сюда все оставляем

    
        printf("Solving system Ax = b ...\n");
        double t_solve = omp_get_wtime();
        data D = solver(T, A, b, eps, maxit, x, n, res);
        t_solve = omp_get_wtime() - t_solve;
        printf("the process was completed in %d iterations with precision %f\n", n, res);
        

        printf("\tProcess\t\t\tTime\t\tMemory\n");
        printf("\tGenerating En\t\t%f s\t%d bytes\n", t_gener, memory_gener);
        printf("\tTranspose En\t\t%f s\t%d bytes\n", t_transpose, memory_transpose);
        printf("\tReciving NeN\t\t%f s\t%d bytes\n", t_recieve, memory_recieve);
        printf("\tFilling A\t\t%f s\t%d bytes\n", t_fill_A, memory_fill_A);
        printf("\tFilling b\t\t%f s\t%d bytes\n", t_fill_b, memory_fill_b);
        printf("\tsolver's work\t\t%f s\t%d bytes\n", t_solve, D.memory);
        printf("\t-- Init OpenCL\t\t%f s\t\n", D.times[0]);
        printf("\t- Find antidiag\t\t%f s\t\n", D.times[1]);
        printf("\t- Copy data on GPU\t%f s\t\n", D.times[2]);
        printf("\t-- Solve Ax=b\t\t%f s\t\n", D.times[3]);
        return 0;
    }

}
#pragma OPENCL EXTENSION cl_khr_fp64 : enable // включаем поддержку даблов

// y = ax+y
__kernel void knlAXPBY(
    int n, 
    __global double* x, 
    __global double *y, 
    __global double *res, 
    double a, 
    double b
){
 const int gid = get_global_id(0);
 if(gid>=n) return;
 res[gid] = a*x[gid] + b*y[gid];
}

// z = x @ y, поэлементоное произведение
__kernel void knlMATMUL(
    int n,
    __global double* x, 
    __global double *y, 
    __global double *z
) {
    const int gid = get_global_id(0);
    if(gid>=n) return;
    z[gid] = x[gid] * y[gid];
}

#define REDUCTION_LWS 256 //reduction workgroup size - fixed!
__kernel void knlDOT(
    int n, 
    __global double* x, 
    __global double* y, 
    __global double *res
) {
    const int gid = get_global_id(0); // глобальный номер задания
    const int lid = get_local_id(0); // номер внутри рабочей группы
    const int gsz = get_global_size(0); // общее число нитей
    __local double sdata[REDUCTION_LWS]; // сюда будем собирать сумму группы

    sdata[lid]=0.0; // зануляем свою позицию

    for(int i=gid; i<n; i+=gsz) sdata[lid] += x[i]*y[i]; // находим сумму данной нити
    barrier(CLK_LOCAL_MEM_FENCE); // барьер, чтобы вся группа нашла свою локальную сумму

    // делаем сборку сдваиванием
    if(lid <128) sdata[lid] += sdata[lid +128]; barrier(CLK_LOCAL_MEM_FENCE);
    if(lid < 64) sdata[lid] += sdata[lid + 64]; barrier(CLK_LOCAL_MEM_FENCE);
    if(lid < 32) sdata[lid] += sdata[lid + 32]; barrier(CLK_LOCAL_MEM_FENCE);
    if(lid < 16) sdata[lid] += sdata[lid + 16]; barrier(CLK_LOCAL_MEM_FENCE);
    if(lid < 8) sdata[lid] += sdata[lid + 8]; barrier(CLK_LOCAL_MEM_FENCE);
    if(lid < 4) sdata[lid] += sdata[lid + 4]; barrier(CLK_LOCAL_MEM_FENCE);
    if(lid < 2) sdata[lid] += sdata[lid + 2]; barrier(CLK_LOCAL_MEM_FENCE);
    if(lid < 1) sdata[lid] += sdata[lid + 1]; barrier(CLK_LOCAL_MEM_FENCE);
    // записываем результат группы
    if(lid == 0) res[get_group_id(0)] = sdata[0];
}

__kernel void knlSPMV_CSR(
    int N,
    __global double* A, 
    __global int* IA, 
    __global int* JA,
    __global double* x, 
    __global double* y
){
    const int gid = get_global_id(0); 
    if(gid >= N) return;
    double sum = 0.0;
    int jbeg = IA[gid];
    int jend = IA[gid+1];
    for(int _j=jbeg; _j<jend; ++_j) {
        sum += A[_j]*x[JA[_j]];
    }
    y[gid] = sum;
}

__kernel void knlAssig(
    int N,
    __global double* x,
    __global double* y
) {
    const int gid = get_global_id(0); 
    if(gid >= N) return;
    y[gid] = x[gid];
}

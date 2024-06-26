g++ -c matrix.cpp -fopenmp -o matrix.o
g++ -o ocl.exe ocl.cpp matrix.o -fopenmp -Wall -I "C:/OCLSDK/include/" -L "C:/OCLSDK/lib/x86/" -lOpenCL
./ocl.exe -Nx 1000 -Ny 1000 -k3 1 -k4 1
	

	

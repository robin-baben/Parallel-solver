run: ocl
	./ocl -Nx 1000 -Ny 1000 -k3 1 -k4 1
ocl: ocl.cpp matrix.o
	g++ -o ocl ocl.cpp matrix.o -fopenmp -O3 -I "C:/OCLSDK/include/" -L "C:/OCLSDK/lib/x86/" -lOpenCL
matrix.o:
	g++ -c matrix.cpp -fopenmp -o matrix.o


	

	

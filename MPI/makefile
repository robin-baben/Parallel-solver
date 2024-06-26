run: mpi
	mpiexec -n 4 ./mpi -Nx 1000 -Ny 1000 -k3 1 -k4 1
mpi: mpi.cpp matrix.o matvecop.o
	mpic++ mpi.cpp matrix.o matvecop.o -fopenmp -O3 -o mpi
matrix.o: matrix.cpp matrix.h
	g++ -c matrix.cpp -fopenmp -o matrix.o
matvec.o: matvec.cpp matvec.h
	g++ -c matvecop.cpp -fopenmp -o matvecop.o
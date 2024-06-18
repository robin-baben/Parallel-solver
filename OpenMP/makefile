run: parallel_copy
	./parallel_copy -Nx 1000 -Ny 1000 -k3 1 -k4 1 -T 4 -eps 0.001
parallel_copy: parallel_copy.cpp
	g++ parallel_copy.cpp -fopenmp -o parallel_copy
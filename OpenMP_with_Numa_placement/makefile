run: with_numa
	./with_numa -Nx 1000 -Ny 1000 -k3 1 -k4 1 -T 4 -eps 0.001
with_numa: with_numa.cpp
	g++ with_numa.cpp -fopenmp -o with_numa
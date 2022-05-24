mpi_overlap_spmv:
	mpicc -o mpi_overlap_spmv mpi_overlap_spmv.c -lmetis -lm -O3
mpi_combal_spmv:
	mpicc -o mpi_combal_spmv mpi_combal_spmv.c -lmetis -lm -O3
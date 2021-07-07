# Distributed-SpMV
分布式稀疏矩阵-向量乘（c/MPI+OpenMP）

purempi_spmv_withouthp.c - MPI分布式SpMV（未使用超图分割）

purempi_spmv.c - MPI分布式SpMV（使用超图分割）

spmv.c - MPI+OpenMP分布式SpMV

编译：

mpicc spmv.c -fopenmp -lmetis -lm -O3

运行:

mpirun -n 进程数 ./a.out 矩阵名 线程数

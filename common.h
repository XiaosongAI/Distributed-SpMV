#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <sys/time.h>

#include <mm_malloc.h>
#include <x86intrin.h>
#include <immintrin.h>
#include <nmmintrin.h>

#include <omp.h>
//#include "mkl.h"
//#include "mkl_spblas.h"

#ifndef MKL_INT
#define MKL_INT int
#endif

#ifndef MAT_VAL_TYPE
#define MAT_VAL_TYPE float
#endif

#ifndef MAT_PTR_TYPE
#define MAT_PTR_TYPE int
#endif

// SpMV 200, SpTRSV 200, SpGEMM 3
// SpMM 20, SpTRSV 10
#ifndef BENCH_REPEAT
#define BENCH_REPEAT 20
#endif

// ---------------------------------------
// ---       MKL regular routines      ---
// CSR-SpMV and COO-SpMV, use 1t 2t 4t 8t 16t 32t 64t, set NTHREADS_MAX = 64
// CSC-SpMV, CSR-SpTRSV, COO-SpTRSV and CSC-SpTRSV, use 1t, set NTHREADS_MAX = 1
// CSR-SpADD and CSR-SpGEMM, use 1t 2t 4t 8t 16t 32t 64t, set NTHREADS_MAX = 64
//
// --- MKL inspector-executor routines ---
// CSR-SpMV, COO-SpMV, CSR-SpTRSV and CSC-SpTRSV, use 1t 2t 4t 8t 16t 32t 64t, set NTHREADS_MAX = 64
// CSC-SpMV, COO-SpTRSV, use 1t, set NTHREADS_MAX = 1
// CSR-SpADD and CSR-SpGEMM, use 1t 2t 4t 8t 16t 32t 64t, set NTHREADS_MAX = 64
// ---------------------------------------
#ifndef NTHREADS_MAX
#define NTHREADS_MAX 1
#endif

#ifndef NCOL_RHS
#define NCOL_RHS 8
#endif

#ifndef TEST_SPMV
#define TEST_SPMV 0
#endif

#ifndef TEST_SPTRSV
#define TEST_SPTRSV 1
#endif

#ifndef TEST_SPGEMM
#define TEST_SPGEMM 0
#endif


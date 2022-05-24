#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>
#include <metis.h>
#include "mmio_highlevel.h"
#define VALUE_TYPE double
#define NTIMES 100

struct MATRIX
{
    int rownum;
    int colnum;
    int nnznum;
    int *rowptr;
    int *colidx;
    VALUE_TYPE *val;
};

struct COMMUN
{
    int infocount;
    int *recvid;
    int *index;
    int *flag;
};

int i, j, k;

int initmtx(struct MATRIX *matrix)
{
    matrix->rowptr = (int *)malloc(sizeof(int) * (matrix->rownum+1));
    matrix->colidx = (int *)malloc(sizeof(int) * matrix->nnznum);
    matrix->val = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * matrix->nnznum);

    return 1;
}

void spmv_serial(int m, int *RowPtr, int *ColIdx, VALUE_TYPE *Val, VALUE_TYPE *Y, VALUE_TYPE *X)
{
    for (i = 0; i < m; i++) 
    {
        Y[i] = 0;
        for (j = RowPtr[i]; j < RowPtr[i+1]; j++)
        {
            Y[i] += Val[j] * X[ColIdx[j]];
        }
    }
}

int readmtx(char *filename, struct MATRIX *sourcematrix)
{
    int isSymmetric;
    printf ("filename = %s\n", filename);
    mmio_info(&sourcematrix->rownum, &sourcematrix->colnum, &sourcematrix->nnznum, &isSymmetric, filename);
    sourcematrix->rowptr = (int *)malloc((sourcematrix->rownum+1) * sizeof(int));
    sourcematrix->colidx = (int *)malloc(sourcematrix->nnznum * sizeof(int));
    sourcematrix->val = (VALUE_TYPE *)malloc(sourcematrix->nnznum * sizeof(VALUE_TYPE));
    mmio_data(sourcematrix->rowptr, sourcematrix->colidx, sourcematrix->val, filename);
    printf("Matrix A is %i by %i, #nonzeros = %i\n", sourcematrix->rownum, sourcematrix->colnum, sourcematrix->nnznum);

    return 1;
}

int mtx_display (struct MATRIX *matrix, char *filename)
{
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "%d %d %d\n", matrix->rownum, matrix->colnum, matrix->nnznum);
    for (i = 0; i < matrix->rownum; i++)
        for (j = matrix->rowptr[i]; j < matrix->rowptr[i+1]; j++)
            fprintf(fp, "%d %d 1\n", i, matrix->colidx[j]);
    fclose(fp);
    return 0;
}

int mtx_reorder (struct MATRIX *matrix, int *reorder_res, int num_parts)
{
    struct MATRIX reordered_matrix;
    reordered_matrix.rownum = matrix->rownum;
    reordered_matrix.colnum = matrix->colnum;
    reordered_matrix.nnznum = matrix->nnznum;
    reordered_matrix.rowptr = (int *)malloc(sizeof(int) * (reordered_matrix.rownum+1));
    int * temp_rowptr = (int *)malloc(sizeof(int) * (reordered_matrix.rownum+1));
    reordered_matrix.colidx = (int *)malloc(sizeof(int) * reordered_matrix.nnznum);
    for (i = 0; i <= reordered_matrix.rownum; i++)
    {
        reordered_matrix.rowptr[i] = 0;
        temp_rowptr[i] = 0;
    }
    for (i = 0; i < reordered_matrix.nnznum; i++) reordered_matrix.colidx[i] = 0;
    reordered_matrix.val = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * reordered_matrix.nnznum);
    int *cnt = (int *)malloc(sizeof(int)*(num_parts+1));
    for (i = 0; i <= num_parts; i++) cnt[i] = 0;
    int *index = (int *)malloc(sizeof(int)*num_parts);
    for (i = 0; i < num_parts; i++) index[i] = 0;
    int *temp_reorder = (int *)malloc(sizeof(int)*matrix->rownum);
    for (i = 0; i < matrix->rownum; i++) 
    {
        cnt[reorder_res[i]]++;
        temp_reorder[i] = 0;
    }
    exclusive_scan(cnt, num_parts+1);
    for (i = 0; i < num_parts; i++) index[i] = 0;
    for (i = 0; i < matrix->rownum; i++)
    {
        temp_reorder[i] = cnt[reorder_res[i]] + index[reorder_res[i]];
        index[reorder_res[i]] = index[reorder_res[i]] + 1;
    }
    
    //row reorder
    for (i = 0; i < matrix->rownum; i++)
        reordered_matrix.rowptr[temp_reorder[i]] = matrix->rowptr[i+1] - matrix->rowptr[i];
    exclusive_scan(reordered_matrix.rowptr, (matrix->rownum+1));
    
    for (i = 0; i < matrix->rownum; i++)
    {
        int nnz_cnt = 0;
        for (j = matrix->rowptr[i]; j < matrix->rowptr[i+1]; j++)
        {
            reordered_matrix.colidx[reordered_matrix.rowptr[temp_reorder[i]]+nnz_cnt] = matrix->colidx[j];
            nnz_cnt += 1;
        }
    }
    
    //colnum reorder
    for (i = 0; i < reordered_matrix.nnznum; i++)
        reordered_matrix.colidx[i] = temp_reorder[reordered_matrix.colidx[i]];
    for (i = 0; i < reordered_matrix.nnznum; i++) reordered_matrix.val[i] = 1.0;


    *matrix = reordered_matrix;
    free(cnt);
    free(index);
    free(temp_reorder);
    free(temp_rowptr);

    return 1;
}

int *mtx_partition (struct MATRIX *matrix, int part)
{
    idx_t rownum = matrix->rownum;
    idx_t *rowptr = (idx_t *)malloc(sizeof(idx_t)*(matrix->rownum+1));
    for (i = 0; i <= rownum; i++) rowptr[i] = matrix->rowptr[i];
    idx_t *colidx = (idx_t *)malloc(sizeof(idx_t)*matrix->nnznum);
    for (i = 0; i < matrix->nnznum; i++) colidx[i] = matrix->colidx[i];
    idx_t weights = 1;
    idx_t *reorder = (idx_t *)malloc(sizeof(idx_t)*matrix->rownum);
    idx_t objval;
    idx_t parts = part;
    METIS_PartGraphKway(&rownum, &weights, rowptr, colidx, NULL, NULL, NULL, &parts, NULL, NULL, NULL, &objval, reorder);
    
    int *order = (int *)malloc(sizeof(int) * matrix->rownum);
    for (i = 0; i < matrix->rownum; i++) order[i] = reorder[i];
    return order;
}

int vec_reorder (VALUE_TYPE *vec, int *reorder, int num_parts, int length)
{
    int *cnt = (int *)malloc(sizeof(int)*(num_parts+1));
    for (i = 0; i <= num_parts; i++) cnt[i] = 0;
    int *index = (int *)malloc(sizeof(int)*num_parts);
    for (i = 0; i < num_parts; i++) index[i] = 0;
    int *temp_reorder = (int *)malloc(sizeof(int)*length);
    VALUE_TYPE *temp_vec = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE )*length);
    for (i = 0; i < length; i++) 
    {
        cnt[reorder[i]]++;
        temp_reorder[i] = 0;
        temp_vec[i] = 0;
    }
    exclusive_scan(cnt, num_parts+1);
    for (i = 0; i < num_parts; i++) index[i] = 0;
    for (i = 0; i < length; i++)
    {
        temp_reorder[i] = cnt[reorder[i]] + index[reorder[i]];
        index[reorder[i]] = index[reorder[i]] + 1;
    }
    for (i = 0; i < length; i++)
        temp_vec[temp_reorder[i]] = vec[i];
    for (i = 0; i < length; i++)
        vec[i] = temp_vec[i];
    return 0;
}

int dividematrixbyrow(struct MATRIX sourcematrix, struct MATRIX *mtx, int partnum)
{
    int *subrowadd = (int *)malloc((partnum+1) * sizeof(int));
    for (i = 0; i < partnum; i++)
    {
        subrowadd[i] = floor(sourcematrix.rownum/partnum*i);
    }
    subrowadd[partnum] = sourcematrix.rownum;
    for (i = 0; i < partnum; i++)
    {
        mtx[i].rownum = subrowadd[i+1] - subrowadd[i];
        mtx[i].colnum = sourcematrix.colnum;
    }
    int start, end;
    for (i = 0; i < partnum; i++)
    {
        start = subrowadd[i];
        end = start + mtx[i].rownum;
        mtx[i].nnznum = sourcematrix.rowptr[end] - sourcematrix.rowptr[start];
    }
    
    int offset = 0;
    for(i = 0; i < partnum; i ++)
    {
        mtx[i].rowptr = (int *)malloc(sizeof(int) * (mtx[i].rownum+1));
        memset(mtx[i].rowptr, 0, sizeof(int) * (mtx[i].rownum+1));
        mtx[i].colidx = (int *)malloc(sizeof(int) * mtx[i].nnznum);
        memset(mtx[i].colidx, 0, sizeof(int) * mtx[i].nnznum);
        mtx[i].val = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * mtx[i].nnznum);
        memset(mtx[i].val, 0, sizeof(VALUE_TYPE) * mtx[i].nnznum);
        int temp;
        if(i == 0)
            temp = 0;
        else
            temp = sourcematrix.rowptr[subrowadd[i]];

        for(j = 0; j < mtx[i].rownum+1; j++)
        {
            mtx[i].rowptr[j] = sourcematrix.rowptr[subrowadd[i]+j] - temp;
        }
        
        for(j = 0; j < mtx[i].nnznum; j++)
        {
            mtx[i].colidx[j] = sourcematrix.colidx[offset+j];
            mtx[i].val[j] = sourcematrix.val[offset+j];
        }
        offset += mtx[i].nnznum;
    }
    
    free(subrowadd);

    return 1;
}

int dividevector(VALUE_TYPE *sourcevector, VALUE_TYPE **vector, int partnum, int length)
{

    for (i = 0; i < partnum; i ++)
    {
        if (i != (partnum-1))
        {    
            for (j = 0; j < length/partnum; j++)
            {
                vector[i][j+i*(length/partnum)] = sourcevector[j+i*(length/partnum)];
            }
        }
        if (i == (partnum-1))
        {
            for (j = 0; j < (length-(length/partnum*(partnum-1))); j++)
            {
                vector[i][j+length-(length-(length/partnum*(partnum-1)))] = sourcevector[j+length-(length-(length/partnum*(partnum-1)))];
            }
        }
    }

    return 1;
}

int binary_search_right_boundary_kernel(const int *row_pointer,
                                        const int  key_input,
                                        const int  size)
{
    int start = 0;
    int stop  = size - 1;
    int median;
    int key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;

        key_median = row_pointer[median];

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }

    return start;
}

void writeresults(char *filename_res, char *filename, int m, int n, int nnzR, double time, double GFlops, int processors, int nthreads)
{
    FILE *fres = fopen(filename_res, "a");
    if (fres == NULL) printf("Writing results fails.\n");
    fprintf(fres, "%s, %i, %i, %d, %lf ms, %lf GFlops, %i, %i\n", filename, m, n, nnzR, time, GFlops, processors, nthreads);
    fclose(fres);
}

int main(int argc, char ** argv)
{
    int p, id, parts, *reorder;
    VALUE_TYPE *x, *sourcevector;
    char *filename;
    struct MATRIX sourcematrix;
    struct timeval t1, t2, t3, t4;
    double comtime;

    //Init MPI enviorment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Status status;
    MPI_Request request;
    struct MATRIX submatrix;
    struct COMMUN vec_commun;

    int *subm = (int *)malloc(sizeof(int) * p);
    int *submadd = (int *)malloc(sizeof(int) * (p+1));
    memset(subm, 0, sizeof(int) * p);
    memset(submadd, 0, sizeof(int) * (p+1));

    if (id == 0)
    {
        filename = argv[1];
        //master id read mtx
        readmtx(filename, &sourcematrix);

        //create input vector
        sourcevector = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * sourcematrix.colnum);
        for (i = 0; i < sourcematrix.colnum; i++) sourcevector[i] = rand()%10+1;//1.0;

        parts = p;
        reorder = (int *)malloc(sizeof(int)*sourcematrix.rownum);
        reorder = mtx_partition(&sourcematrix, parts);
        
        mtx_reorder(&sourcematrix, reorder, parts);
        vec_reorder(sourcevector, reorder, parts, sourcematrix.colnum);

        //create mtx array
        struct MATRIX *mtx = (struct MATRIX *)malloc(p * sizeof(struct MATRIX));

        //divide matrix by row
        dividematrixbyrow(sourcematrix, mtx, p);
        
        //compute subm and subadd
        for (i = 0; i < p; i++)
        {
            subm[i] = mtx[i].rownum;
            submadd[i] = i * mtx[0].rownum;
        }
        submadd[p] = sourcematrix.rownum;
        
        VALUE_TYPE **vector = (VALUE_TYPE **)malloc(sizeof(VALUE_TYPE *) * p);
        for(i = 0; i < p; i++)
        {
            vector[i] = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * sourcematrix.colnum);
            memset(vector[i], 0, sizeof(VALUE_TYPE)*sourcematrix.colnum);
        }
        
        //divide vector by row
        dividevector(sourcevector, vector, p, sourcematrix.colnum);

        //compute vector index those need to be communicated
        int xidx[p+1];
        for (i = 0; i < p; i++)
            xidx[i] = i * floor(sourcematrix.colnum/p);
        xidx[p] = sourcematrix.colnum;

        //create comm struct
        struct COMMUN *comm = (struct COMMUN *)malloc(p * sizeof(struct COMMUN));
        for (i = 0; i < p; i++)
        {
            comm[i].infocount = 0;
            comm[i].recvid = (int *)malloc(sizeof(int) * sourcematrix.colnum);
            comm[i].index = (int *)malloc(sizeof(int) * sourcematrix.colnum);
            comm[i].flag = (int *)malloc(sizeof(int) * sourcematrix.colnum);
            memset(comm[i].flag, 0, sourcematrix.colnum);
        }
        
        int communicate_total_num = 0;
        int *send_num = (int *)malloc(sizeof(int) * p);
        memset(send_num, 0, sizeof(int) * p);
        int *recv_num = (int *)malloc(sizeof(int) * p);
        memset(recv_num, 0, sizeof(int) * p);
        for (i = 0; i < p; i++)
        { 
            for (j = 0; j < mtx[i].rownum; j++)
            {
                for (k = mtx[i].rowptr[j]; k < mtx[i].rowptr[j+1]; k++)
                {
                    int templength = floor(sourcematrix.colnum/p);
                    if (mtx[i].colidx[k] < xidx[i] || mtx[i].colidx[k] >= xidx[i+1])
                    {
                        int need1 = (mtx[i].colidx[k]/templength) > (p-1) ? (p-1) : (mtx[i].colidx[k]/templength);
                        if (comm[i].flag[mtx[i].colidx[k]] == 0)
                        {
                            int need2 = mtx[i].colidx[k] - need1*templength;
                            comm[need1].recvid[comm[need1].infocount] = i;
                            comm[need1].index[comm[need1].infocount] = mtx[i].colidx[k];
                            comm[i].flag[mtx[i].colidx[k]] = 1;
                            comm[need1].infocount++;
                            communicate_total_num++;
                            send_num[need1]++;
                            recv_num[i]++;
                        }
                    }
                }
            }
        }
        printf("communicate_total_num %d\n", communicate_total_num);

        submatrix.rownum = mtx[id].rownum;
        submatrix.colnum = mtx[id].colnum;
        submatrix.nnznum = mtx[id].nnznum;
        initmtx(&submatrix);
        submatrix.rowptr = mtx[id].rowptr;
        submatrix.colidx = mtx[id].colidx;
        submatrix.val = mtx[id].val;
        vec_commun.infocount = comm[0].infocount;
        vec_commun.recvid = (int *)malloc(sizeof(int) * vec_commun.infocount);
        vec_commun.index = (int *)malloc(sizeof(int) * vec_commun.infocount);
        vec_commun.recvid = comm[0].recvid;
        vec_commun.index = comm[0].index;

        x = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * submatrix.colnum);
        x = vector[id];

        //scatter matrix and vector
        for(i = 1; i < p; i++)
        {
            MPI_Send(&mtx[i].rownum, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(&mtx[i].colnum, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
            MPI_Send(&mtx[i].nnznum, 1, MPI_INT, i, 3, MPI_COMM_WORLD);
            MPI_Send(mtx[i].rowptr, mtx[i].rownum+1, MPI_INT, i, 4, MPI_COMM_WORLD);
            MPI_Send(mtx[i].colidx, mtx[i].nnznum, MPI_INT, i, 5, MPI_COMM_WORLD);
            MPI_Send(mtx[i].val, mtx[i].nnznum, MPI_FLOAT, i, 6, MPI_COMM_WORLD);
            MPI_Send(vector[i], mtx[i].colnum, MPI_FLOAT, i, 7, MPI_COMM_WORLD);
            MPI_Send(&comm[i].infocount, 1, MPI_INT, i, 8, MPI_COMM_WORLD);
            MPI_Send(comm[i].recvid, comm[i].infocount, MPI_INT, i, 9, MPI_COMM_WORLD);
            MPI_Send(comm[i].index, comm[i].infocount, MPI_INT, i, 10, MPI_COMM_WORLD);
        }
    } 
    else
    {
        //receive matrix and vector
        MPI_Recv(&submatrix.rownum, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&submatrix.colnum, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
        MPI_Recv(&submatrix.nnznum, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, &status);
        initmtx(&submatrix);
        MPI_Recv(submatrix.rowptr, submatrix.rownum+1, MPI_INT, 0, 4, MPI_COMM_WORLD, &status);
        MPI_Recv(submatrix.colidx, submatrix.nnznum, MPI_INT, 0, 5, MPI_COMM_WORLD, &status);
        MPI_Recv(submatrix.val, submatrix.nnznum, MPI_FLOAT, 0, 6, MPI_COMM_WORLD, &status);
        x = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * submatrix.colnum);
        MPI_Recv(x, submatrix.colnum, MPI_FLOAT, 0, 7, MPI_COMM_WORLD, &status);
        MPI_Recv(&vec_commun.infocount, 1, MPI_INT, 0, 8, MPI_COMM_WORLD, &status);
        vec_commun.recvid = (int *)malloc(sizeof(int) * vec_commun.infocount);
        vec_commun.index = (int *)malloc(sizeof(int) * vec_commun.infocount);
        MPI_Recv(vec_commun.recvid, vec_commun.infocount, MPI_INT, 0, 9, MPI_COMM_WORLD, &status);
        MPI_Recv(vec_commun.index, vec_commun.infocount, MPI_INT, 0, 10, MPI_COMM_WORLD, &status);
    }

    //start communicate
    int length[p], temp[p];
    for (i = 0; i < p; i++)
    {
        length[i] = 0;
        temp[i] = 0;
    }
    VALUE_TYPE **commun_vec = (VALUE_TYPE **)malloc(sizeof(VALUE_TYPE *) * p);
    for (i = 0; i < p; i++) commun_vec[i] = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * vec_commun.infocount);
    int **commun_vec_index = (int **)malloc(sizeof(int *) * p);
    for (i = 0; i < p; i++) commun_vec_index[i] = (int *)malloc(sizeof(int) * vec_commun.infocount);
    for (j = 0; j < vec_commun.infocount; j++)
    {
        commun_vec[vec_commun.recvid[j]][temp[vec_commun.recvid[j]]] = x[vec_commun.index[j]];
        commun_vec_index[vec_commun.recvid[j]][temp[vec_commun.recvid[j]]] = vec_commun.index[j];
        temp[vec_commun.recvid[j]]++;
    }
    int **final_index = (int **)malloc(sizeof(int *) * p);
    VALUE_TYPE **final_value = (VALUE_TYPE **)malloc(sizeof(VALUE_TYPE *) * p);
    //recv vector

    if (id == 0) gettimeofday(&t1, NULL);
    for (int r = 0; r < NTIMES; r++)
    {
        for (i = 0; i < p; i++)
        {
            if (i != id)
            {
                //printf("id %d send %d number to id%d\n", id, temp[i], i);
                MPI_Send(&temp[i], 1, MPI_INT, i, 11, MPI_COMM_WORLD);
                MPI_Isend(commun_vec_index[i], temp[i], MPI_INT, i, 12, MPI_COMM_WORLD, &request);
                MPI_Isend(commun_vec[i], temp[i], MPI_FLOAT, i, 13, MPI_COMM_WORLD, &request);
                //printf("id %d %d %4.2f\n", id, commun_vec_index[i][0], commun_vec[i][0]);
            }
        }
        for (i = 0; i < p; i++)
        {
            if (i != id)
            {
                MPI_Recv(&length[i], 1, MPI_INT, i, 11, MPI_COMM_WORLD, &status);
                //printf("id%d recv i%d %d\n", id, i, length[i]);
                final_index[i] = (int *)malloc(sizeof(int) * length[i]);
                memset(final_index[i], 0, length[i]);
                final_value[i] = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * length[i]);
                memset(final_value[i], 0, length[i]);
                MPI_Recv(final_index[i], length[i], MPI_INT, i, 12, MPI_COMM_WORLD, &status);
                MPI_Recv(final_value[i], length[i], MPI_FLOAT, i, 13, MPI_COMM_WORLD, &status);
            }
        }
    }
    if (id == 0) 
    {
        gettimeofday(&t2, NULL);
	    comtime = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / NTIMES;
        printf("communicate time %4.2f ms\n", comtime);
    }
    for (i = 0; i < p; i++)
        for (j = 0; j < length[i]; j++)
            if (id != i)
                x[final_index[i][j]] = final_value[i][j];

    
    VALUE_TYPE *y = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * submatrix.rownum);
    memset(y, 0, sizeof(VALUE_TYPE) * submatrix.rownum);
    
    int nthreads = atoi(argv[2]);
    omp_set_num_threads(nthreads);
    
    // find balanced points
    int *csrSplitter = (int *)malloc((nthreads+1) * sizeof(int));
    int stridennz = ceil((double)submatrix.nnznum/(double)nthreads);
    #pragma omp parallel for
    for (int tid = 0; tid <= nthreads; tid++)
    {
        int boundary = tid * stridennz;
        boundary = boundary > submatrix.nnznum ? submatrix.nnznum : boundary;
        csrSplitter[tid] = binary_search_right_boundary_kernel(submatrix.rowptr, boundary, submatrix.rownum + 1) - 1;
    }

    gettimeofday(&t1, NULL);
    for (int q = 0; q < NTIMES; q++)
    {
        #pragma omp parallel for
        for (int tid = 0; tid < nthreads; tid++)
        {
            for (int u = csrSplitter[tid]; u < csrSplitter[tid+1]; u++)
            {
                y[u] = 0;
                for (int h = submatrix.rowptr[u]; h < submatrix.rowptr[u+1]; h++)
                {
                    y[u] += submatrix.val[h] * x[submatrix.colidx[h]];
                }
            }
        }
    }
    gettimeofday(&t2, NULL);
    if (id == 0)
    {
        double distime = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / NTIMES;
        double GFlops = 2 * sourcematrix.nnznum / (distime+comtime) / pow(10,6);
        printf("spmv time %4.2f ms  %4.2f GFlops\n", distime+comtime, GFlops);
        // writeresults("mpi+omp.csv", filename, sourcematrix.rownum, sourcematrix.colnum, sourcematrix.nnznum, distime, GFlops, p, nthreads);
    }
    
    MPI_Bcast(subm, p, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(submadd, p+1, MPI_INT, 0, MPI_COMM_WORLD);
    VALUE_TYPE *result = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * sourcematrix.rownum);
    if (id == 0) gettimeofday(&t1, NULL);
    MPI_Gatherv(y, submatrix.rownum, MPI_FLOAT, result, subm, submadd, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (id == 0)
    {
        int errorcnt = 0;
        VALUE_TYPE *y_golden = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * sourcematrix.rownum);
        memset(y_golden, 0, sizeof(VALUE_TYPE) * sourcematrix.rownum);
        spmv_serial(sourcematrix.rownum, sourcematrix.rowptr, sourcematrix.colidx, sourcematrix.val, y_golden, sourcevector);

        for (i = 0; i < sourcematrix.rownum; i++)
        {
            if (result[i] != y_golden[i]) 
            {
                errorcnt++;
            }
        }
        printf("error count %d \n", errorcnt);
    }

    MPI_Finalize();
}


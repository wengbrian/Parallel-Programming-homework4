#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>

const int INF = 1000000000;
const int V = 20010;
void input(char *inFileName);
void output(char *outFileName);
int *d_ptr[2];
size_t pitch[2];
int size = 3*32*32*sizeof(int);
dim3 block(32,32);

void block_FW(int B);
int ceil(int a, int b);
__global__ void cal(int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int *d_ptr, size_t pitch);
void kernel(int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int *d_ptr, size_t pitch);
void test2();

int n, m;	// Number of vertices, edges
static int Dist[V][V];

int main(int argc, char* argv[])
{
	input(argv[1]);
	int B = atoi(argv[3]);
    //test2();
	block_FW(B);
    //test2();
	output(argv[2]);

	return 0;
}

void input(char *inFileName)
{
	FILE *infile = fopen(inFileName, "r");
	fscanf(infile, "%d %d", &n, &m);

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (i == j)	Dist[i][j] = 0;
			else		Dist[i][j] = INF;
		}
	}

	while (--m >= 0) {
		int a, b, v;
		fscanf(infile, "%d %d %d", &a, &b, &v);
		Dist[a][b] = v;
	}
    fclose(infile);
}

void output(char *outFileName)
{
	FILE *outfile = fopen(outFileName, "w");
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			// if (Dist[i][j] >= INF)	fprintf(outfile, "INF ");
			// else					fprintf(outfile, "%d ", Dist[i][j]);
            if (Dist[i][j] >= INF)
                Dist[i][j] = INF;
		}
		fwrite(Dist[i], sizeof(int), n, outfile);
	}
    fclose(outfile);
}

int ceil(int a, int b)
{
	return (a + b -1)/b;
}

int floor(int a, int b)
{
    return a/b;
}

__global__
void test(int *d_ptr, size_t pitch, int n){
    printf("\ndevice:\n");
    for(int i = 0; i < n; i++){
        int *row = (int*) ((char*)d_ptr + i * pitch);
        for(int j = 0; j < n; j++){
            printf("%3d", row[j]);
        }
        printf("\n");
    }
    printf("\n");
}

void test2(){
    printf("\nhost:\n");
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            printf("%3d", Dist[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}


void kernel(int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int *d_ptr, size_t pitch){
    int num_block_width = ceil(32*block_width,32);
    int num_block_height = ceil(32*block_height,32);
    dim3 grid(num_block_width, num_block_height);
    cal<<<grid,block,size>>>(Round, block_start_x, block_start_y, block_width,block_height, n, d_ptr, pitch);
}

void QQ(int tid, char *str){
    cudaError_t cudaerr = cudaThreadSynchronize();
    if (cudaerr != cudaSuccess)
        printf("[%d,%s]kernel launch failed with error \"%s\".\n", tid, str, cudaGetErrorString(cudaerr));
}

__global__
void test3(int tid, int* d_ptr){
    printf("[%d]:%d\n", tid, d_ptr);
}
void block_FW(int B)
{

    // num round
	int round = ceil(n, B);
    cudaError_t cudaerr;
    omp_set_num_threads(2);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        cudaSetDevice(tid);
        // copy memory to host
        cudaerr = cudaMallocPitch(&(d_ptr[tid]), &(pitch[tid]), n * sizeof(int), n);
        if (cudaerr != cudaSuccess)
            printf("[%d,%s]kernel launch failed with error \"%s\".\n", tid, "malloc2d", cudaGetErrorString(cudaerr));
        cudaerr = cudaMemcpy2D(d_ptr[tid], pitch[tid], &Dist, V * sizeof(int), n * sizeof(int), n, cudaMemcpyHostToDevice);
        if (cudaerr != cudaSuccess)
            printf("[%d,%s]kernel launch failed with error \"%s\".\n", tid, "memcpy2d", cudaGetErrorString(cudaerr));
        //test3<<<1,1>>>(tid, d_ptr[tid]);
        //cudaerr = cudaThreadSynchronize();
        //if (cudaerr != cudaSuccess)
        //    printf("[%d,%s]kernel launch failed with error \"%s\".\n", tid, "print", cudaGetErrorString(cudaerr));
        int offset;
        for (int r = 0; r < round; ++r) {
            int width[4] = {floor(r,2), ceil(r,2), ceil(round-r-1,2), floor(round-r-1,2)};
            int idx[2] = {floor(r,2), r+1+width[2]};
            if(tid == 0){
                // outer
                kernel(r, r, r, 1, 1, n, d_ptr[tid], pitch[tid]);
                //QQ(tid,"(2,2)");
                //cudaerr = cudaDeviceSynchronize();
                //if (cudaerr != cudaSuccess)
                //    printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));
                if(width[0] > 0){
                    kernel(r, r, 0, 1, width[0], n, d_ptr[tid], pitch[tid]); // outer-up (3,1)
                    //QQ(tid, "(2,0)");
                    //printf("host: outer-up\n");
                }
                if(width[3] > 0){
                    kernel(r, r, idx[1], 1, width[3], n, d_ptr[tid], pitch[tid]); // outer-down (3,5)
                    //QQ(tid, "(2,4)");
                    //printf("host: outer-down\n");
                }
                if(width[0] > 0){
                    kernel(r, 0, r, width[0], 1, n, d_ptr[tid], pitch[tid]); // outer-left (1,3)
                    //QQ(tid, "(0,2)");
                    //printf("host: outer-left\n");
                }
                if(width[3] > 0){
                    kernel(r, idx[1], r, width[3], 1, n, d_ptr[tid], pitch[tid]); // outer-right (5,3)
                    //QQ(tid, "(4,2)");
                    //printf("host: outer-right\n");
                }

                // TODO
                // wait cal so can async memory copy
                cudaerr = cudaThreadSynchronize();
                if (cudaerr != cudaSuccess)
                    printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));

                offset = r*32;
                if(width[0] > 0){
                    //cudaMemcpy2DAsync(d_ptr[1]+offset, pitch[1], d_ptr[0]+offset, pitch[0], 1*32*sizeof(int), width[0]*32, cudaMemcpyDeviceToDevice, 0);   
                    cudaMemcpy2D(d_ptr[1]+offset, pitch[1], d_ptr[0]+offset, pitch[0], 1*32*sizeof(int), width[0]*32, cudaMemcpyDeviceToDevice);   
                }
                offset = idx[1]*32*pitch[0] + r*32;
                if(width[3] > 0){
                    //cudaMemcpy2DAsync(d_ptr[1]+offset, pitch[1], d_ptr[0]+offset, pitch[0], 1*32*sizeof(int), width[3]*32, cudaMemcpyDeviceToDevice, 0);   
                    cudaMemcpy2D(d_ptr[1]+offset, pitch[1], d_ptr[0]+offset, pitch[0], 1*32*sizeof(int), width[3]*32, cudaMemcpyDeviceToDevice);   
                }
                offset = r*32*pitch[0];
                if(width[0] > 0){
                    //cudaMemcpy2DAsync(d_ptr[1]+offset, pitch[1], d_ptr[0]+offset, pitch[0], width[0]*32*sizeof(int), 1*32, cudaMemcpyDeviceToDevice, 0);   
                    cudaMemcpy2D(d_ptr[1]+offset, pitch[1], d_ptr[0]+offset, pitch[0], width[0]*32*sizeof(int), 1*32, cudaMemcpyDeviceToDevice);   
                }
                offset = r*32*pitch[0] + idx[1]*32;
                if(width[3] > 0){
                    //cudaMemcpy2DAsync(d_ptr[1]+offset, pitch[1], d_ptr[0]+offset, pitch[0], width[3]*32*sizeof(int), 1*32, cudaMemcpyDeviceToDevice, 0);   
                    cudaMemcpy2D(d_ptr[1]+offset, pitch[1], d_ptr[0]+offset, pitch[0], width[3]*32*sizeof(int), 1*32, cudaMemcpyDeviceToDevice);   
                }

                if(width[0] > 0){
                    kernel(r, 0, 0, width[0], width[0], n, d_ptr[tid], pitch[tid]); // outer-left-up (0,0)
                    //QQ(tid, "(0,0)");
                }
                if((width[0] > 0) && (width[3] > 0)){
                    kernel(r, 0, idx[1], width[0], width[3], n, d_ptr[tid], pitch[tid]); // outer-left-down (4,0)
                    //QQ(tid, "(4,0)");
                }
                if((width[0] > 0) && (width[3] > 0)){
                    kernel(r, idx[1], 0, width[3], width[0], n, d_ptr[tid], pitch[tid]); // outer-right-up (0,4)
                    //QQ(tid, "(0,4)");
                }
                if(width[3] > 0){
                    kernel(r, idx[1], idx[1], width[3], width[3], n, d_ptr[tid], pitch[tid]); // right-down (4,4)
                    //QQ(tid, "(4,4)");
                }

                // TODO
                // wait for memory copy
                // wait for sending
                cudaerr = cudaThreadSynchronize();
                if (cudaerr != cudaSuccess)
                    printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));
                #pragma omp barrier
                // wait for receiving

                if((width[0] > 0) && (width[1] > 0)){
                    kernel(r, 0, idx[0], width[0], width[1], n, d_ptr[tid], pitch[tid]); // outer-left-up-grid (1,0)
                    //QQ(tid, "(1,0)");
                }
                if((width[0] > 0) && (width[2] > 0)){
                    kernel(r, 0, r+1, width[0], width[2], n, d_ptr[tid], pitch[tid]); // outer-left-down-grid (3,0)
                    //QQ(tid, "(3,0)");
                }
                if((width[1] > 0) && (width[3] > 0)){
                    kernel(r, idx[1], idx[0], width[3], width[1], n, d_ptr[tid], pitch[tid]); // outer-right-up-grid (1,4)
                    //QQ(tid, "(1,4)");
                }
                if((width[2] > 0) && (width[3] > 0)){
                    kernel(r, idx[1], r+1, width[3], width[2], n, d_ptr[tid], pitch[tid]); // outer-right-down-grid (3,4)
                    //QQ(tid, "(3,4)");
                }
                offset = 0;
                //if(r != round-1)
                if(width[0] > 0){
                    cudaMemcpy2D(d_ptr[1]+offset, pitch[1], d_ptr[0]+offset, pitch[0], width[0]*32*sizeof(int), n, cudaMemcpyDeviceToDevice);   
                //else
                    cudaMemcpy2D(((int*)&Dist)+offset, V*sizeof(int), d_ptr[0]+offset, pitch[0], width[0]*32*sizeof(int), n, cudaMemcpyDeviceToHost);
                }
                offset = idx[1]*32;
                //if(r != round-1)
                if(width[3]){
                    cudaMemcpy2D(d_ptr[1]+offset, pitch[1], d_ptr[0]+offset, pitch[0], width[3]*32*sizeof(int), n, cudaMemcpyDeviceToDevice);   
                //else
                    cudaMemcpy2D(((int*)&Dist)+offset, V*sizeof(int), d_ptr[0]+offset, pitch[0], width[3]*32*sizeof(int), n, cudaMemcpyDeviceToHost);
                }
                #pragma omp barrier
            }else if(tid == 1){
                // inner 
                kernel(r, r, r, 1, 1, n, d_ptr[tid], pitch[tid]);
                //QQ(tid,"(2,2)");
                //cudaerr = cudaDeviceSynchronize();
                //if (cudaerr != cudaSuccess)
                //    printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));

                if(width[1] > 0){
                    kernel(r, r, idx[0], 1, width[1], n, d_ptr[tid], pitch[tid]); // inner-up (2,1)
                    //QQ(tid, "(2,1)");
                    //printf("host: inner-up\n");
                }
                if(width[2] > 0){
                    kernel(r, r, r+1, 1, width[2], n, d_ptr[tid], pitch[tid]); // inner-down (2,3)
                    //QQ(tid, "(2,3)");
                    //printf("host: inner-down\n");
                }
                if(width[1] > 0){
                    kernel(r, idx[0], r, width[1], 1, n, d_ptr[tid], pitch[tid]); // inner-left (1,2)
                    //QQ(tid, "(1,2)");
                    //printf("host: inner-left\n");
                }
                if(width[2] > 0){
                    kernel(r, r+1, r, width[2], 1, n, d_ptr[tid], pitch[tid]); // inner-right (3,2)
                    //QQ(tid, "(3,2)");
                    //printf("host: inner-right\n");
                }

                // TODO
                // wait cal so can async memory copy
                cudaerr = cudaThreadSynchronize();
                if (cudaerr != cudaSuccess)
                    printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));
                
                offset = idx[0]*32*pitch[1] + r*32;
                if(width[1] > 0){
                    //cudaMemcpy2DAsync(d_ptr[0]+offset, pitch[0], d_ptr[1]+offset, pitch[1], 1*32*sizeof(int), width[1]*32, cudaMemcpyDeviceToDevice, 0);   
                    cudaMemcpy2D(d_ptr[0]+offset, pitch[0], d_ptr[1]+offset, pitch[1], 1*32*sizeof(int), width[1]*32, cudaMemcpyDeviceToDevice);   
                }
                offset = (r+1)*32*pitch[1] + r*32;
                if(width[2] > 0){
                    //cudaMemcpy2DAsync(d_ptr[0]+offset, pitch[0], d_ptr[1]+offset, pitch[1], 1*32*sizeof(int), width[2]*32, cudaMemcpyDeviceToDevice, 0);   
                    cudaMemcpy2D(d_ptr[0]+offset, pitch[0], d_ptr[1]+offset, pitch[1], 1*32*sizeof(int), width[2]*32, cudaMemcpyDeviceToDevice);   
                }
                offset = r*32*pitch[1] + idx[0]*32;
                if(width[1] > 0){
                    //cudaMemcpy2DAsync(d_ptr[0]+offset, pitch[0], d_ptr[1]+offset, pitch[1], width[1]*32*sizeof(int), 1*32, cudaMemcpyDeviceToDevice, 0);   
                    cudaMemcpy2D(d_ptr[0]+offset, pitch[0], d_ptr[1]+offset, pitch[1], width[1]*32*sizeof(int), 1*32, cudaMemcpyDeviceToDevice);   
                }
                offset = r*32*pitch[1] + (r+1)*32;
                if(width[2] > 0){
                    //cudaMemcpy2DAsync(d_ptr[0]+offset, pitch[0], d_ptr[1]+offset, pitch[1], width[2]*32*sizeof(int), 1*32, cudaMemcpyDeviceToDevice, 0);   
                    cudaMemcpy2D(d_ptr[0]+offset, pitch[0], d_ptr[1]+offset, pitch[1], width[2]*32*sizeof(int), 1*32, cudaMemcpyDeviceToDevice);   
                }


                if(width[1] > 0){
                    kernel(r, idx[0], idx[0], width[1], width[1], n, d_ptr[tid], pitch[tid]); // inner-left-up (1,1)
                    //QQ(tid, "(1,1)");
                }
                if((width[1] > 0) && (width[2] > 0)){
                    kernel(r, idx[0], r+1, width[1], width[2], n, d_ptr[tid], pitch[tid]); // inner-left-down (3,1)
                    //QQ(tid, "(3,1)");
                }
                if((width[1] > 0) && (width[2] > 0)){
                    kernel(r, r+1, idx[0], width[2], width[1], n, d_ptr[tid], pitch[tid]); // inner-right-up (1,3)
                    //QQ(tid, "(1,3)");
                }
                if(width[2] > 0){
                    kernel(r, r+1, r+1, width[2], width[2], n, d_ptr[tid], pitch[tid]); // inner-right-down (3,3)
                    //QQ(tid, "(3,3)");
                }

                // TODO
                // wait for memory copy
                // wait for sending
                cudaerr = cudaThreadSynchronize();
                if (cudaerr != cudaSuccess)
                    printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));
                // wait for receiving
                #pragma omp barrier

                if((width[0] > 0) && (width[1] > 0)){
                    kernel(r, idx[0], 0, width[1], width[0], n, d_ptr[tid], pitch[tid]); // outer-up-left-grid (0,1)
                    //QQ(tid, "(0,1)");
                }
                if((width[0] > 0) && (width[2] > 0)){
                    kernel(r, r+1, 0, width[2], width[0],  n, d_ptr[tid], pitch[tid]); // outer-up-right-grid (0,3)
                    //QQ(tid, "(0,3)");
                }
                if((width[1] > 0) && (width[3] > 0)){
                    kernel(r, idx[0], idx[1], width[1], width[3], n, d_ptr[tid], pitch[tid]); // outer-right-up-grid (4,1)
                    //QQ(tid, "(4,1)");
                }
                if((width[2] > 0) && (width[3] > 0)){
                    kernel(r, r+1, idx[1], width[2], width[3], n, d_ptr[tid], pitch[tid]); // outer-right-down-grid (4,3)
                    //QQ(tid, "(4,3)");
                }
                offset = idx[0]*32;
                //if(r!=round-1)
                    cudaMemcpy2D(d_ptr[0]+offset, pitch[0], d_ptr[1]+offset, pitch[1], (width[1]+width[2]+1)*32*sizeof(int), n, cudaMemcpyDeviceToDevice);   
                //else
                    cudaMemcpy2D(((int*)&Dist)+offset, V*sizeof(int), d_ptr[1]+offset, pitch[1], (width[1]+width[2]+1)*32*sizeof(int), n, cudaMemcpyDeviceToHost);   
                #pragma omp barrier
                //test<<<1,1>>>(d_ptr[1], pitch[1], n);
                //if(r==round-1)
                //    test<<<1,1>>>(d_ptr[1], pitch[1], n);
            }
            //if(tid==0)
            //    test2();
        }
        cudaFree(d_ptr[tid]);
    }
        //cudaMemcpy2D(&Dist, V * sizeof(int), d_ptr[tid], pitch[tid], n * sizeof(int), n, cudaMemcpyDeviceToHost);
}

__global__
void cal(int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int *d_ptr, size_t pitch)
{
    // shared memory
    __shared__ int dist[32][32];
    __shared__ int rowBlock[32][32];
    __shared__ int colBlock[32][32];

    int b_x = blockIdx.x * blockDim.x + threadIdx.x;
    int b_y = blockIdx.y * blockDim.y + threadIdx.y;
    int tidx = block_start_x * 32 + b_x;
    int tidy = block_start_y * 32 + b_y;
    //int inRange = (tidx < n) && (tidy < n) && (b_x < block_width*32) && (b_y < block_height*32); 
    int inRange = (tidx < n) && (tidy < n); 
    int k_start = Round*32; // k_start
    int k_end = (Round+1)*32;
    // copy current block
    if (inRange){
        int *row = (int *)((char*)d_ptr + tidy * pitch);
        dist[threadIdx.y][threadIdx.x] = row[tidx]; 
    }

    // copy needed row block
    int idx_x = k_start + threadIdx.x;
    if((idx_x < k_end) && (idx_x < n) && (tidy < n)){
        int *row = (int *)((char*)d_ptr + tidy * pitch);
        rowBlock[threadIdx.y][threadIdx.x] = row[idx_x];
    }
    // copy needed col block
    int idx_y = k_start + threadIdx.y;
    if((idx_y < k_end) && (idx_y < n) && (tidx < n)){
        int *row = (int *)((char*)d_ptr + idx_y * pitch);
        colBlock[threadIdx.y][threadIdx.x] = row[tidx];
    }
    __syncthreads();

    // current k = Round*B ~ Round*B+size-1
    for(int k = 0; (k < 32) && (k_start+k < n); k++){
        if (inRange){
            int ik = rowBlock[threadIdx.y][k];
            int kj = colBlock[k][threadIdx.x];
            if (ik + kj < dist[threadIdx.y][threadIdx.x]) {
                dist[threadIdx.y][threadIdx.x] = ik + kj;
                if((tidx >= k_start) && (tidx < k_end)){
                    rowBlock[threadIdx.y][threadIdx.x] = ik+kj;
                }
                if((tidy >= k_start) && (tidy < k_end)){
                    colBlock[threadIdx.y][threadIdx.x] = ik+kj;
                }
            }
        }
        __syncthreads();
    }

    // copy shared memory to global memory
    if (inRange){
        int *row = (int *)((char*)d_ptr + tidy * pitch);
        row[tidx] = dist[threadIdx.y][threadIdx.x]; 
    }
}



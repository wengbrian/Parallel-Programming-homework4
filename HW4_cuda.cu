#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int INF = 1000000000;
const int V = 20010;
void input(char *inFileName);
void output(char *outFileName);
int *d_ptr;
size_t pitch;
int debug = 0;

void copyTo();
void copyBack();
void block_FW(int B);
int ceil(int a, int b);
__global__ void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int *d_ptr, size_t pitch, int num_block, int debug);

int n, m;	// Number of vertices, edges
static int Dist[V][V];

int main(int argc, char* argv[])
{
	input(argv[1]);
	int B = atoi(argv[3]);
    copyTo();
	block_FW(B);
    copyBack();

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

void copyTo(){
    if(debug){
        printf("\nhost:\n");
        for(int i = 0; i < 5; i++){
            int *row =  Dist[i];
            for(int j = 0; j < 5; j++){
                printf("%d ", row[j]);
            }
            printf("\n");
        }
    }
    cudaMallocPitch(&d_ptr, &pitch, n * sizeof(int), n);
    cudaMemcpy2D(d_ptr, pitch, &Dist, V * sizeof(int), n * sizeof(int), n, cudaMemcpyHostToDevice);
}

void copyBack(){
    cudaMemcpy2D(&Dist, V * sizeof(int), d_ptr, pitch, n * sizeof(int), n, cudaMemcpyDeviceToHost);
    if(debug){
        printf("\nhost:\n");
        for(int i = 0; i < 5; i++){
            int *row =  Dist[i];
            for(int j = 0; j < 5; j++){
                printf("%d ", row[j]);
            }
            printf("\n");
        }
    }
}
__global__
void test(int *d_ptr, size_t pitch, int n){
    printf("\ndevice:\n");
    for(int i = 0; i < n; i++){
        int *row = (int*) ((char*)d_ptr + i * pitch);
        for(int j = 0; j < n; j++){
            printf("%d ", row[j]);
        }
        printf("\n");
    }
}

void block_FW(int B)
{
	int round = ceil(n, B);
    int size = 3*1024*sizeof(int);
    dim3 block(32,32);
    int num_block = ceil(B,32);
	for (int r = 0; r < round; ++r) {
        if(debug)
            printf("%d %d\n", r, round);

		/* Phase 1*/
        int num_block_width = ceil(B,32);
        int num_block_height = ceil(B,32);
        dim3 grid(num_block_width, num_block_height);
		cal<<<grid,block,size>>>(B,	r,	r,	r,	1,	1, n, d_ptr, pitch, num_block, debug);
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess)
            printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));

		/* Phase 2*/
        if(r > 0){
            int num_block_width = ceil(B,32);
            int num_block_height = ceil(B*r,32);
            dim3 grid(num_block_width, num_block_height);
            cal<<<grid,block,size>>>(B, r, r, 0, 1, r, n, d_ptr, pitch, num_block, debug); // up
            cudaError_t cudaerr = cudaDeviceSynchronize();
            if (cudaerr != cudaSuccess)
                printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));
        }
        if(round-r-1 > 0){
            int num_block_width = ceil(B,32);
            int num_block_height = ceil(B*(round-r-1),32);
            dim3 grid(num_block_width, num_block_height);
		    cal<<<grid,block,size>>>(B, r, r, r+1, 1, round-r-1, n, d_ptr, pitch, num_block, debug); // down
            cudaError_t cudaerr = cudaDeviceSynchronize();
            if (cudaerr != cudaSuccess)
                printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));
        }
        if(r > 0){
            int num_block_width = ceil(B*r,32);
            int num_block_height = ceil(B,32);
            dim3 grid(num_block_width, num_block_height);
		    cal<<<grid,block,size>>>(B, r, 0, r, r, 1, n, d_ptr, pitch, num_block, debug); // left
            cudaError_t cudaerr = cudaDeviceSynchronize();
            if (cudaerr != cudaSuccess)
                printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));
        }
        if(round-r-1 > 0){
            int num_block_width = ceil(B*(round-r-1),32);
            int num_block_height = ceil(B,32);
            dim3 grid(num_block_width, num_block_height);
		    cal<<<grid,block,size>>>(B, r, r+1, r, round-r-1, 1, n, d_ptr, pitch, num_block, debug); // right
            cudaError_t cudaerr = cudaDeviceSynchronize();
            if (cudaerr != cudaSuccess)
                printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));
        }

		/* Phase 3*/
        if(r > 0){
            int num_block_width = ceil(B*r,32);
            int num_block_height = ceil(B*r,32);
            dim3 grid(num_block_width, num_block_height);
            cal<<<grid,block,size>>>(B, r, 0, 0, r, r, n, d_ptr, pitch, num_block, debug); // left-up
            cudaError_t cudaerr = cudaDeviceSynchronize();
            if (cudaerr != cudaSuccess)
                printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));
        }
        if((round-r-1 > 0) && (r > 0)){
            int num_block_width = ceil(B*r,32);
            int num_block_height = ceil(B*(round-r-1),32);
            dim3 grid(num_block_width, num_block_height);
		    cal<<<grid,block,size>>>(B, r, 0, r+1, r, round-r-1, n, d_ptr, pitch, num_block, debug); // left-down
            cudaError_t cudaerr = cudaDeviceSynchronize();
            if (cudaerr != cudaSuccess)
                printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));
        }
        if((round-r-1 > 0) && (r > 0)){
            int num_block_width = ceil(B*(round-r-1),32);
            int num_block_height = ceil(B*r,32);
            dim3 grid(num_block_width, num_block_height);
		    cal<<<grid,block,size>>>(B, r, r +1, 0, round-r-1, r, n, d_ptr, pitch, num_block, debug); // right-up
            cudaError_t cudaerr = cudaDeviceSynchronize();
            if (cudaerr != cudaSuccess)
                printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));
        }
        if(round-r-1 > 0){
            int num_block_width = ceil(B*(round-r-1),32);
            int num_block_height = ceil(B*(round-r-1),32);
            dim3 grid(num_block_width, num_block_height);
		    cal<<<grid,block,size>>>(B, r, r+1, r+1, round-r-1, round-r-1, n, d_ptr, pitch, num_block, debug); // right-down
            cudaError_t cudaerr = cudaDeviceSynchronize();
            if (cudaerr != cudaSuccess)
                printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));
        }
	}
    if(debug)
        test<<<1,1>>>(d_ptr, pitch, n);
}

__global__
void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int *d_ptr, size_t pitch, int num_block, int debug)
{
    int size = B < 32 ? B : 32;
    // shared memory
    __shared__ int dist[32][32];
    __shared__ int rowBlock[32][32];
    __shared__ int colBlock[32][32];

    int b_x = blockIdx.x * blockDim.x + threadIdx.x;
    int b_y = blockIdx.y * blockDim.y + threadIdx.y;
    int tidx = block_start_x * B + b_x;
    int tidy = block_start_y * B + b_y;
    // copy memory from global memory to shared memory
    // copy current block
    if ((tidx < n) && (tidy < n) && (b_x < block_width*B) && (b_y < block_height*B)){
        int *row = (int *)((char*)d_ptr + tidy * pitch);
        dist[threadIdx.y][threadIdx.x] = row[tidx]; 
        //if(debug)printf("[%d,%d][ex_tidx=%d, ex_tidy=%d][in_tidx=%d, in_tidy=%d][b_x=%d, b_y=%d]\n",  blockIdx.x, blockIdx.y, tidx, tidy, threadIdx.x, threadIdx.y, b_x, b_y);
    }

    for (int i = 0; i < num_block; i++){
        // copy needed row block
        int idx_x = Round*B + threadIdx.x + i*32; // k_x
        if((idx_x < (Round+1)*B) && (idx_x < n) && (tidy < n)){
        //if(true){
            int *row = (int *)((char*)d_ptr + tidy * pitch);
            rowBlock[threadIdx.y][threadIdx.x] = row[idx_x];
        }
        // copy needed col block
        int idx_y = Round*B + threadIdx.y + i*32;
        if((idx_y < (Round+1)*B) && (idx_y < n) && (tidx < n)){
        //if(true){
            int *row = (int *)((char*)d_ptr + idx_y * pitch);
            colBlock[threadIdx.y][threadIdx.x] = row[tidx];
        }
        __syncthreads();

        // current k = Round*B ~ Round*B+size-1
        for(int k = 0; (k < size) && (Round*B+k < n); k++){
            //if(threadIdx.x+threadIdx.y==0)printf("[k=%d]\n", k);
            if ((tidx < n) && (tidy < n) && (b_x < block_width*B) && (b_y < block_height*B)){
                int ik = rowBlock[threadIdx.y][k];
                int kj = colBlock[k][threadIdx.x];
                //if(debug)if(blockIdx.y==5)printf("[%d/%d][%d,%d][k=%d][ex_tidx=%d, ex_tidy=%d][in_tidx=%d, in_tidy=%d][b_x=%d, b_y=%d]\n", i+1, num_block, blockIdx.x, blockIdx.y, Round*B+k, tidx, tidy, threadIdx.x, threadIdx.y, b_x, b_y);
                if (ik + kj < dist[threadIdx.y][threadIdx.x]) {
                    dist[threadIdx.y][threadIdx.x] = ik + kj;
                    if((tidx >= Round*B) && (tidx < (Round+1)*B)){
                        rowBlock[threadIdx.y][threadIdx.x] = ik+kj;
                    }
                    if((tidy >= Round*B) && (tidy < (Round+1)*B)){
                        colBlock[threadIdx.y][threadIdx.x] = ik+kj;
                    }
                }
            }
            __syncthreads();
        }
    }

    // copy shared memory to global memory
    if ((tidx < n) && (tidy < n) && (b_x < block_width*B) && (b_y < block_height*B)){
        int *row = (int *)((char*)d_ptr + tidy * pitch);
        row[tidx] = dist[threadIdx.y][threadIdx.x]; 
    }
}



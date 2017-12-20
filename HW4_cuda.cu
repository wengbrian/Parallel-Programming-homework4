#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int INF = 1000000000;
const int V = 20010;
void input(char *inFileName);
void output(char *outFileName);
int *d_ptr;
size_t pitch;
int size = 3*32*32*sizeof(int);
dim3 block(32,32);

void block_FW(int B);
int ceil(int a, int b);
__global__ void cal(int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int *d_ptr, size_t pitch);
void kernel(int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int *d_ptr, size_t pitch);

int n, m;	// Number of vertices, edges
static int Dist[V][V];

int main(int argc, char* argv[])
{
	input(argv[1]);
	int B = atoi(argv[3]);
	block_FW(B);

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

void kernel(int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int *d_ptr, size_t pitch){
    int num_block_width = ceil(32*block_width,32);
    int num_block_height = ceil(32*block_height,32);
    dim3 grid(num_block_width, num_block_height);
    cal<<<grid,block,size>>>(Round, block_start_x, block_start_y, block_width,block_height, n, d_ptr, pitch);
}

void block_FW(int B)
{
    // copy memory to host
    cudaMallocPitch(&d_ptr, &pitch, n * sizeof(int), n);
    cudaMemcpy2D(d_ptr, pitch, &Dist, V * sizeof(int), n * sizeof(int), n, cudaMemcpyHostToDevice);

    // num round
	int round = ceil(n, B);
    //cudaError_t cudaerr;
	for (int r = 0; r < round; ++r) {

		/* Phase 1*/
		kernel(r, r, r, 1, 1, n, d_ptr, pitch);
        //cudaerr = cudaDeviceSynchronize();
        //if (cudaerr != cudaSuccess)
        //    printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));

		/* Phase 2*/
        if(r > 0){
            kernel(r, r, 0, 1, r, n, d_ptr, pitch); // up
        }
        if(round-r-1 > 0){
		    kernel(r, r, r+1, 1, round-r-1, n, d_ptr, pitch); // down
        }
        if(r > 0){
		    kernel(r, 0, r, r, 1, n, d_ptr, pitch); // left
        }
        if(round-r-1 > 0){
		    kernel(r, r+1, r, round-r-1, 1, n, d_ptr, pitch); // right
        }
        //cudaerr = cudaDeviceSynchronize();
        //if (cudaerr != cudaSuccess)
        //    printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));

		/* Phase 3*/
        if(r > 0){
            kernel(r, 0, 0, r, r, n, d_ptr, pitch); // left-up
        }
        if((round-r-1 > 0) && (r > 0)){
		    kernel(r, 0, r+1, r, round-r-1, n, d_ptr, pitch); // left-down
        }
        if((round-r-1 > 0) && (r > 0)){
		    kernel(r, r +1, 0, round-r-1, r, n, d_ptr, pitch); // right-up
        }
        if(round-r-1 > 0){
		    kernel(r, r+1, r+1, round-r-1, round-r-1, n, d_ptr, pitch); // right-down
        }
        //cudaerr = cudaDeviceSynchronize();
        //if (cudaerr != cudaSuccess)
        //    printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));
	}
    cudaMemcpy2D(&Dist, V * sizeof(int), d_ptr, pitch, n * sizeof(int), n, cudaMemcpyDeviceToHost);
    cudaFree(d_ptr);
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



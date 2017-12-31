#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int INF = 1000000000;
const int V = 20010;
void input(char *inFileName);
void output(char *outFileName);
int *d_ptr;
size_t pitch;
#define BF 32
int size = 3*BF*BF*sizeof(int);
dim3 block(BF,BF);
double totalTime = 0;
double cummTime = 0;
double IOTime = 0;
int B;
struct timespec diff(struct timespec start, struct timespec end) {
    struct timespec temp;
    if ((end.tv_nsec-start.tv_nsec)<0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec-start.tv_sec;
        temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    return temp;
}
void block_FW(int B);
int ceil(int a, int b);
__global__ void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int *d_ptr, size_t pitch);
void kernel(int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int *d_ptr, size_t pitch);

int n, m;	// Number of vertices, edges
static int Dist[V][V];

int main(int argc, char* argv[])
{
	struct timespec start, end, temp;
    clock_gettime(CLOCK_MONOTONIC, &start);
	input(argv[1]);
	B = atoi(argv[3]);
	block_FW(B);

	output(argv[2]);
	clock_gettime(CLOCK_MONOTONIC, &end);
    temp = diff(start, end);
    double time_used = temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
	totalTime += time_used;
    totalTime -= cummTime;
    totalTime -= IOTime;
    printf("%f %f %f\n", totalTime, cummTime, IOTime);
	return 0; 
}

void input(char *inFileName)
{
	struct timespec start, end, temp;
    clock_gettime(CLOCK_MONOTONIC, &start);
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
	clock_gettime(CLOCK_MONOTONIC, &end);
    temp = diff(start, end);
    double time_used = temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
    IOTime += time_used;
}

void output(char *outFileName)
{
	struct timespec start, end, temp;
    clock_gettime(CLOCK_MONOTONIC, &start);
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
	clock_gettime(CLOCK_MONOTONIC, &end);
    temp = diff(start, end);
    double time_used = temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
    IOTime += time_used;
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
    int num_block_width = ceil(B*block_width,B);
    int num_block_height = ceil(B*block_height,B);
    dim3 grid(num_block_width, num_block_height);
    cal<<<grid,block,size>>>(B, Round, block_start_x, block_start_y, block_width,block_height, n, d_ptr, pitch);
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
void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int *d_ptr, size_t pitch)
{
    // shared memory
    __shared__ int dist[BF][BF];
    __shared__ int rowBlock[BF][BF];
    __shared__ int colBlock[BF][BF];

    int b_x = blockIdx.x * blockDim.x + threadIdx.x;
    int b_y = blockIdx.y * blockDim.y + threadIdx.y;
    int tidx = block_start_x * B + b_x;
    int tidy = block_start_y * B + b_y;
    //int inRange = (tidx < n) && (tidy < n) && (b_x < block_width*B) && (b_y < block_height*B); 
    int inRange = (tidx < n) && (tidy < n); 
    int k_start = Round*B; // k_start
    int k_end = (Round+1)*B;
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
    for(int k = 0; (k < B) && (k_start+k < n); k++){
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



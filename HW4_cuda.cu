#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int INF = 1000000000;
const int V = 20010;
void input(char *inFileName);
void output(char *outFileName);
int *d_ptr;
size_t pitch;

void copyTo();
void copyBack();
void block_FW(int B);
int ceil(int a, int b);
__global__ void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int *d_ptr, size_t pitch);

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
    cudaMallocPitch(&d_ptr, &pitch, n * sizeof(int), n);
    cudaMemcpy2D(d_ptr, pitch, &Dist, n * sizeof(int), n, n, cudaMemcpyHostToDevice);
}

void copyBack(){
    cudaMemcpy2D(&Dist, n * sizeof(int), d_ptr, pitch, n, n, cudaMemcpyDeviceToHost);
}

void block_FW(int B)
{
	int round = ceil(n, B);
    int num_block = ceil(B*B, 1024);
    int size = B*B*sizeof(int);
    num_block = ceil(B, 32);
    dim3 grid(num_block, num_block);
    size = 1024*sizeof(int)*3;
    size = 48*1024;
	for (int r = 0; r < round; ++r) {
        printf("%d %d\n", r, round);
		/* Phase 1*/
		cal<<<grid,1024,size>>>(B,	r,	r,	r,	1,	1, n, d_ptr, pitch);
		//printf("%d ", cal(B,	r,	r,	r,	1,	1));
        // sync
		/* Phase 2*/
        cal<<<grid,1024,size>>>(B, r,     r,     0,             r,             1, n, d_ptr, pitch); // left
		cal<<<grid,1024,size>>>(B, r,     r,  r +1,  round - r -1,             1, n, d_ptr, pitch); // right
		cal<<<grid,1024,size>>>(B, r,     0,     r,             1,             r, n, d_ptr, pitch); // up
		cal<<<grid,1024,size>>>(B, r,  r +1,     r,             1,  round - r -1, n, d_ptr, pitch); // down
        // sync
		/* Phase 3*/
        cal<<<grid,1024,size>>>(B, r,     0,     0,            r,             r, n, d_ptr, pitch); // left-up
		cal<<<grid,1024,size>>>(B, r,     0,  r +1,  round -r -1,             r, n, d_ptr, pitch); // right-up
		cal<<<grid,1024,size>>>(B, r,  r +1,     0,            r,  round - r -1, n, d_ptr, pitch); // left-down
		cal<<<grid,1024,size>>>(B, r,  r +1,  r +1,  round -r -1,  round - r -1, n, d_ptr, pitch); // right-down
        
	}
}

__global__
void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int *d_ptr, size_t pitch)
{
    printf("QQ");
    extern __shared__ int mem[];
    int *dist = mem;
    int *rowBlock = dist + 1024; 
    int *colBlock = rowBlock + 32*B*sizeof(int);
    int num_block = gridDim.x; // num split block, ceil(B/32)

    int idx = threadIdx.y*32+threadIdx.x; // internal index
    int tidx = blockIdx.x*blockDim.x + threadIdx.x; // external index
    int tidy = blockIdx.y*blockDim.y + threadIdx.y; // external index

    // copy memory from global memory to shared memory
    // copy current block
    if ((tidx < n) && (tidy < n)){
        int *row = (int *)((char*)d_ptr + tidy * pitch);
        dist[threadIdx.y*32+threadIdx.x] = row[tidx]; 
    }

    for (int i = 0; i < num_block; i++){

        // copy needed row block
        int idx_x = Round*B + threadIdx.x + i*32;
        if(idx_x < (Round+1)*B){
            int *row = (int *)((char*)d_ptr + tidy * pitch);
            rowBlock[i*1024+threadIdx.y*32+threadIdx.x] = row[idx_x];
        }

        // copy needed col block
        int idx_y = Round*B + threadIdx.y + i*32;
        if(idx_y < (Round+1)*B){
            int *row = (int *)((char*)d_ptr + idx_y * pitch);
            colBlock[i*1024+threadIdx.y*32+threadIdx.x] = row[threadIdx.x];
        }
    }

	int block_end_x = block_start_x + block_height;
	int block_end_y = block_start_y + block_width;
	for (int b_i =  block_start_x; b_i < block_end_x; ++b_i) {
		for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
            int block_internal_start_x 	= b_i * B;
            int block_internal_end_x 	= (b_i +1) * B;
            int block_internal_start_y  = b_j * B;
            int block_internal_end_y 	= (b_j +1) * B;
            if (block_internal_end_x > n)	block_internal_end_x = n;
            if (block_internal_end_y > n)	block_internal_end_y = n;
			// To calculate B*B elements in the block (b_i, b_j)
			// For each block, it need to compute B times
			for (int k = Round * B; (k < (Round +1) * B) && (k < n); ++k) {
				// To calculate original index of elements in the block (b_i, b_j)
				// For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
				for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
					for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
                        int ik = rowBlock[(k-(Round*B))/32*1024+threadIdx.x*32+threadIdx.x];
                        int kj = colBlock[(k-(Round*B))/32*1024+threadIdx.x*32+threadIdx.x];
                            printf("if %d + %d < %d\n", ik, kj, dist[idx]);
						if (ik + kj < dist[idx]) {
							dist[idx] = ik + kj;
                        }
					}
				}
			}
		}
	}
}



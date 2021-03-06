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
void test2();

int n, m;	// Number of vertices, edges
static int Dist[V][V];
int* d=&Dist[0][0];
int debug = -1;

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

int floor(int a, int b)
{
    return a/b;
}

__global__
void test(int *d_ptr, size_t pitch, int n, int a, int b, int round){
    printf("\nround%d, device: (%d %d) %d\n", round, a, b, n);
    for(int i = 0; i < n; i++){
        int *row = (int*) ((char*)d_ptr + i * pitch);
        for(int j = 0; j < n; j++){
            printf("%4d", row[j]);
        }
        printf("\n");
    }
    printf("\n");
}

void test2(){
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    printf("\nhost:\n");
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            printf("%4d", Dist[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void test3(int a, int b){
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    printf("\nhost:(%d,%d)\n",a,b);
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            printf("%4d", Dist[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void kernel(int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int *d_ptr, size_t pitch, cudaStream_t stream){
    //int num_block_width = ceil(B*block_width,B);
    //int num_block_height = ceil(B*block_height,B);
    dim3 grid(block_width, block_height);
    cal<<<grid,block,size, 0>>>(B, Round, block_start_x, block_start_y, block_width,block_height, n, d_ptr, pitch);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("[%s]kernel launch failed with error \"%s\".\n", "kernel", cudaGetErrorString(cudaerr));
}

void myCudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, int tid, int a, int b){
    cudaError_t cudaerr = cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);   
    if (cudaerr != cudaSuccess)
        printf("[%d,%s,%d,%d]kernel launch failed with error \"%s\".\n", tid, "memcpy2d", a, b, cudaGetErrorString(cudaerr));
}

void myCudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream, int tid, int a, int b){
    cudaError_t cudaerr = cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);   
    //cudaError_t cudaerr = cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);   
    if (cudaerr != cudaSuccess)
        printf("[%d,%s,%d,%d]kernel launch failed with error \"%s\".\n", tid, "memcpy2d", a, b, cudaGetErrorString(cudaerr));
}

void myCudaDeviceSynchronize(int tid, char *msg){
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("[%d,%s,%s]kernel launch failed with error \"%s\".\n", tid, "sync", msg, cudaGetErrorString(cudaerr));
}

void myCudaSetDevice(int tid){
    cudaError_t cudaerr = cudaSetDevice(tid);
    if (cudaerr != cudaSuccess)
        printf("[%d,%s]kernel launch failed with error \"%s\".\n", tid, "setDevice", cudaGetErrorString(cudaerr));
}

void myCudaMallocPitch(int **devPtr, size_t *pitch, size_t width, size_t height, int tid){
    cudaError_t cudaerr = cudaMallocPitch(devPtr, pitch, width, height);
    if (cudaerr != cudaSuccess)
        printf("[%d,%s]kernel launch failed with error \"%s\".\n", tid, "mallocPitch", cudaGetErrorString(cudaerr));
}

void myCudaDeviceEnablePeerAccess(int tid){
    cudaError_t cudaerr = cudaDeviceEnablePeerAccess(tid, 0);
    if (cudaerr != cudaSuccess)
        printf("[%d,%s]kernel launch failed with error \"%s\".\n", tid, "enablePeer", cudaGetErrorString(cudaerr));
}

void block_FW(int B)
{
    // num round
	int round = ceil(n, B);
    omp_set_num_threads(2);
    #pragma omp parallel
    {
        // get thread ID
        int tid = omp_get_thread_num();

        // set device
        myCudaSetDevice(tid);

        // enable memory access
        //myCudaDeviceEnablePeerAccess(1-tid);

        // allocate memory and copy memory to device
        myCudaMallocPitch(&(d_ptr[tid]), &(pitch[tid]), n * sizeof(int), n, tid);
        myCudaDeviceSynchronize(tid, "wait malloc");

        cudaStream_t stream[4];
        for (int i = 0; i < 4; ++i)
            cudaStreamCreate(&stream[i]);

        #pragma omp barrier

        myCudaMemcpy2D(d_ptr[tid], pitch[tid], d, V * sizeof(int), n * sizeof(int), n, cudaMemcpyHostToDevice, tid, 0,0);

        int offset;
        int offset2;
        for (int r = 0; r < round; ++r) {
            //printf("round: %d, B: %d\n", r, B);
            int width[5] = {floor(r,2), ceil(r,2), 1, ceil(round-r-1,2), floor(round-r-1,2)};
            int range[5] = {width[0]*B, width[1]*B, 1*B, width[3]*B, width[4]*B};
            int idx[5] = {0, floor(r,2), r, r+1, r+1+width[3]};
            if(r < round-2){
                range[4] = n-idx[4]*B;
            }else if(r==round-2){
                range[3] = n-(round-1)*B;
            }else if(r==round-1){
                range[2] = n-(round-1)*B;
            }
            if(tid == 0){
                // outer
                kernel(r, idx[2], idx[2], width[2], width[2], n, d_ptr[tid], pitch[tid], stream[0]); // (3,3)

                if(width[0] > 0){
                    kernel(r, idx[2], idx[0], width[2], width[0], n, d_ptr[tid], pitch[tid], stream[0]); // (3,1)
                    if(debug==0)test<<<1,1>>>(d_ptr[0], pitch[tid], n, 3,1,r);
                }
                if(width[4] > 0){
                    kernel(r, idx[2], idx[4], width[2], width[4], n, d_ptr[tid], pitch[tid], stream[0]); // (3,5)
                    if(debug==0)test<<<1,1>>>(d_ptr[0], pitch[tid], n, 3,5,r);
                }
                if(width[0] > 0){
                    kernel(r, idx[0], idx[2], width[0], width[2], n, d_ptr[tid], pitch[tid], stream[0]); // (1,3)
                    if(debug==0)test<<<1,1>>>(d_ptr[0], pitch[tid], n, 1,3,r);
                }
                if(width[4] > 0){
                    kernel(r, idx[4], idx[2], width[4], width[2], n, d_ptr[tid], pitch[tid], stream[0]); // (5,3)
                    if(debug==0)test<<<1,1>>>(d_ptr[0], pitch[tid], n, 5,3,r);
                }

                if(range[0] > 0){ // copy (3,1)
                    offset  = idx[2]*B*sizeof(int);
					offset2 = idx[2]*B;
                    myCudaMemcpy2DAsync(d+offset2, V*sizeof(int), (char*)d_ptr[tid]+offset, pitch[tid], range[2]*sizeof(int), range[0], cudaMemcpyDeviceToHost, stream[0], tid, 3, 1);   
                }
                if(range[4] > 0){ // copy(3,5)
                    offset  = idx[4]*B*pitch[tid] + idx[2]*B*sizeof(int);
                    offset2 = idx[4]*B*V        + idx[2]*B;
                    myCudaMemcpy2DAsync(d+offset2, V*sizeof(int), (char*)d_ptr[tid]+offset, pitch[tid], range[2]*sizeof(int), range[4], cudaMemcpyDeviceToHost, stream[0], tid, 3, 5);   
                }
                if(range[0] > 0){ // copy (1,3)
                    offset  = idx[2]*B*pitch[tid];
                    offset2 = idx[2]*B*V;
                    myCudaMemcpy2DAsync(d+offset2, V*sizeof(int), (char*)d_ptr[tid]+offset, pitch[tid], range[0]*sizeof(int), range[2], cudaMemcpyDeviceToHost, stream[0], tid, 1, 3);   
                }
                if(range[4] > 0){ // copy (5,3)
                    offset  = idx[2]*B*pitch[tid] + idx[4]*B*sizeof(int);
                    offset2 = idx[2]*B*V        + idx[4]*B;
                    myCudaMemcpy2DAsync(d+offset2, V*sizeof(int), (char*)d_ptr[tid]+offset, pitch[tid], range[4]*sizeof(int), range[2], cudaMemcpyDeviceToHost, stream[0], tid, 5, 3);   
                }

				if(width[0] > 0){
                    kernel(r, idx[0], idx[0], width[0], width[0], n, d_ptr[tid], pitch[tid], stream[0]); // outer-left-up (1,1)
                    if(debug==0)test<<<1,1>>>(d_ptr[0], pitch[tid], n, 1,1,r);
                }
                if((width[0] > 0) && (width[4] > 0)){
                    kernel(r, idx[0], idx[4], width[0], width[4], n, d_ptr[tid], pitch[tid], stream[0]); // outer-left-down (1,5)
                    if(debug==0)test<<<1,1>>>(d_ptr[0], pitch[tid], n, 1,5,r);
                }
                if((width[0] > 0) && (width[4] > 0)){
                    kernel(r, idx[4], idx[0], width[4], width[0], n, d_ptr[tid], pitch[tid], stream[0]); // outer-right-up (5,1)
                    if(debug==0)test<<<1,1>>>(d_ptr[0], pitch[tid], n, 5,1,r);
                }
                if(width[4] > 0){
                    kernel(r, idx[4], idx[4], width[4], width[4], n, d_ptr[tid], pitch[tid], stream[0]); // right-down (5,5)
                    if(debug==0)test<<<1,1>>>(d_ptr[0], pitch[tid], n, 5,5,r);
                }
                
				/////////////////////////////////////////
                myCudaDeviceSynchronize(tid, "wait phase3");
                #pragma omp barrier
				/////////////////////////////////////////

                if(range[1] > 0){ // copy(3,2)
                    offset =  idx[1]*B*pitch[tid] + idx[2]*B*sizeof(int);
                    offset2 = idx[1]*B*V        + idx[2]*B;
                    myCudaMemcpy2DAsync((char*)d_ptr[0]+offset, pitch[tid], d+offset2, V*sizeof(int), range[2]*sizeof(int), range[1], cudaMemcpyHostToDevice, stream[0], tid, 3,2);   
                    if(debug==0)test<<<1,1>>>(d_ptr[0], pitch[tid], n, 3, 2,r);
                }
                if(range[3] > 0){ // copy(3,4)
                    offset =  idx[3]*B*pitch[tid] + idx[2]*B*sizeof(int);
                    offset2 = idx[3]*B*V        + idx[2]*B;
                    myCudaMemcpy2DAsync((char*)d_ptr[0]+offset, pitch[tid], d+offset2, V*sizeof(int), range[2]*sizeof(int), range[3], cudaMemcpyHostToDevice, stream[0], tid, 3,4);
                    if(debug==0)test<<<1,1>>>(d_ptr[0], pitch[tid], n, 3,4,r);
                }
                if(range[1] > 0){ // copy(2,3)
                    offset =  idx[2]*B*pitch[tid] + idx[1]*B*sizeof(int);
                    offset2 = idx[2]*B*V        + idx[1]*B;
                    myCudaMemcpy2DAsync((char*)d_ptr[0]+offset, pitch[tid], d+offset2, V*sizeof(int), range[1]*sizeof(int), range[2], cudaMemcpyHostToDevice, stream[0], tid, 2,3);   
                    if(debug==0)test<<<1,1>>>(d_ptr[0], pitch[tid], n, 2,3,r);
                }
                if(range[3] > 0){ // copy(4,3)
                    offset =  idx[2]*B*pitch[tid] + idx[3]*B*sizeof(int);
                    offset2 = idx[2]*B*V        + idx[3]*B;
                    myCudaMemcpy2DAsync((char*)d_ptr[0]+offset, pitch[tid], d+offset2, V*sizeof(int), range[3]*sizeof(int), range[2], cudaMemcpyHostToDevice, stream[0], tid, 4,3);   
                    if(debug==0)test<<<1,1>>>(d_ptr[0], pitch[tid], n, 4,3,r);
                }

                if((width[0] > 0) && (width[1] > 0)){
                    kernel(r, idx[0], idx[1], width[0], width[1], n, d_ptr[tid], pitch[tid], stream[0]); // outer-left-up-grid (1,2)
                    if(debug==0)test<<<1,1>>>(d_ptr[0], pitch[tid], n, 1,2,r);
                }
                if((width[0] > 0) && (width[3] > 0)){
                    kernel(r, idx[0], idx[3], width[0], width[3], n, d_ptr[tid], pitch[tid], stream[0]); // outer-left-down-grid (1,4)
                    if(debug==0)test<<<1,1>>>(d_ptr[0], pitch[tid], n, 1,4,r);
                }
                if((width[1] > 0) && (width[4] > 0)){
                    kernel(r, idx[4], idx[1], width[4], width[1], n, d_ptr[tid], pitch[tid], stream[0]); // outer-right-up-grid (5,2)
                    if(debug==0)test<<<1,1>>>(d_ptr[0], pitch[tid], n, 5,2,r);
                }
                if((width[3] > 0) && (width[4] > 0)){
                    kernel(r, idx[4], idx[3], width[4], width[3], n, d_ptr[tid], pitch[tid], stream[0]); // outer-right-down-grid (5,4)
                    if(debug==0)test<<<1,1>>>(d_ptr[0], pitch[tid], n, 5,4,r);
                }
				
				/////////////////////////////////////////
                myCudaDeviceSynchronize(tid, "wait phase3");
				/////////////////////////////////////////
				
                if(range[0] > 0){
                    offset = idx[0];
					myCudaMemcpy2D(d+offset, V*sizeof(int), (char*)d_ptr[0]+offset, pitch[tid], range[0]*sizeof(int), n, cudaMemcpyDeviceToHost, tid, -1, -1);
                }
                if(range[4] > 0){
                    offset =  idx[4]*B*sizeof(int);
					offset2 = idx[4]*B;
					myCudaMemcpy2D(d+offset2, V*sizeof(int), (char*)d_ptr[0]+offset, pitch[tid], range[4]*sizeof(int), n, cudaMemcpyDeviceToHost, tid, -2, -2);
                }
				
				/////////////////////////////////////////
				#pragma omp barrier
				/////////////////////////////////////////
				
				// copy memory from host to device
				offset =  idx[1]*B*sizeof(int);
				offset2 = idx[1]*B;
				myCudaMemcpy2D((char*)d_ptr[0]+offset, pitch[tid], d+offset2, V*sizeof(int), (range[1]+range[2]+range[3])*sizeof(int), n, cudaMemcpyHostToDevice, tid,-3,-3);   
                				
                if(debug==0)test<<<1,1>>>(d_ptr[0], pitch[tid], n, -1,-1, r);
                //test<<<1,1>>>(d_ptr[0], pitch[tid], n, -1,-1, r);
                //test2();

            }else if(tid == 1){
                // inner 
                kernel(r, r, r, 1, 1, n, d_ptr[tid], pitch[tid], stream[0]); // (3,3)
				
                if(width[1] > 0){
                    kernel(r, idx[2], idx[1], width[2], width[1], n, d_ptr[tid], pitch[tid], stream[0]); // (3,2)
                    if(debug==1)test<<<1,1>>>(d_ptr[1], pitch[tid], n, 3,2, r);
                }
                if(width[3] > 0){
                    kernel(r, idx[2], idx[3], width[2], width[3], n, d_ptr[tid], pitch[tid], stream[0]); // (3,4)
                    if(debug==1)test<<<1,1>>>(d_ptr[1], pitch[tid], n, 3,4, r);
                }
                if(width[1] > 0){
                    kernel(r, idx[1], idx[2], width[1], width[2], n, d_ptr[tid], pitch[tid], stream[0]); // (2,3)
                    if(debug==1)test<<<1,1>>>(d_ptr[1], pitch[tid], n, 2,3, r);
                }
                if(width[3] > 0){
                    kernel(r, idx[3], idx[2], width[3], width[2], n, d_ptr[tid], pitch[tid], stream[0]); // (4,3)
                    if(debug==1)test<<<1,1>>>(d_ptr[1], pitch[tid], n, 4,3, r);
                }
				
                if(range[1] > 0){ // copy (3,2)
                    offset  = idx[1]*B*pitch[tid] + idx[2]*B*sizeof(int);
                    offset2 = idx[1]*B*V        + idx[2]*B;
                    myCudaMemcpy2DAsync(d+offset2, V*sizeof(int), (char*)d_ptr[1]+offset, pitch[tid], range[2]*sizeof(int), range[1], cudaMemcpyDeviceToHost, stream[0], tid, 3,2);   
                }
                if(range[3] > 0){ // copy (3,4)
                    offset  = idx[3]*B*pitch[tid] + idx[2]*B*sizeof(int);
                    offset2 = idx[3]*B*V        + idx[2]*B;
                    myCudaMemcpy2DAsync(d+offset2, V*sizeof(int), (char*)d_ptr[1]+offset, pitch[tid], range[2]*sizeof(int), range[3], cudaMemcpyDeviceToHost, stream[0], tid, 3,4);
                }
                if(range[1] > 0){ // copy (2,3)
                    offset =  idx[2]*B*pitch[tid] + idx[1]*B*sizeof(int);
                    offset2 = idx[2]*B*V        + idx[1]*B;
                    myCudaMemcpy2DAsync(d+offset2, V*sizeof(int), (char*)d_ptr[1]+offset, pitch[tid], range[1]*sizeof(int), range[2], cudaMemcpyDeviceToHost, stream[0], tid, 2,3);   
                }
                if(range[3] > 0){ // copy (4,3)
                    offset =  idx[2]*B*pitch[tid] + idx[3]*B*sizeof(int);
                    offset2 = idx[2]*B*V        + idx[3]*B;
                    myCudaMemcpy2DAsync(d+offset2, V*sizeof(int), (char*)d_ptr[1]+offset, pitch[tid], range[3]*sizeof(int), range[2], cudaMemcpyDeviceToHost, stream[0], tid, 4,3);   
                }

				if(width[1] > 0){
                    kernel(r, idx[1], idx[1], width[1], width[1], n, d_ptr[tid], pitch[tid], stream[0]); // inner-left-up (2,2)
                    if(debug==1)test<<<1,1>>>(d_ptr[1], pitch[tid], n, 2,2, r);
                }
                if((width[1] > 0) && (width[3] > 0)){
                    kernel(r, idx[1], idx[3], width[1], width[3], n, d_ptr[tid], pitch[tid], stream[0]); // inner-left-down (2,4)
                    if(debug==1)test<<<1,1>>>(d_ptr[1], pitch[tid], n, 2,4, r);
                }
                if((width[1] > 0) && (width[3] > 0)){
                    kernel(r, idx[3], idx[1], width[3], width[1], n, d_ptr[tid], pitch[tid], stream[0]); // inner-right-up (4,2)
                    if(debug==1)test<<<1,1>>>(d_ptr[1], pitch[tid], n, 4,2, r);
                }
                if(width[3] > 0){
                    kernel(r, idx[3], idx[3], width[3], width[3], n, d_ptr[tid], pitch[tid], stream[0]); // inner-right-down (4,4)
                    if(debug==1)test<<<1,1>>>(d_ptr[1], pitch[tid], n, 4,4, r);
                }
				
				/////////////////////////////////////////
                myCudaDeviceSynchronize(tid, "wait phase3");
				#pragma omp barrier
                /////////////////////////////////////////

                if(range[0] > 0){ // copy (3,1)
                    offset =  idx[2]*B*sizeof(int);
					offset2 = idx[2]*B;
                    myCudaMemcpy2DAsync((char*)d_ptr[1]+offset, pitch[tid], d+offset2, V*sizeof(int), range[2]*sizeof(int), range[0], cudaMemcpyHostToDevice, stream[0], tid, 3, 1);   
                }
                if(range[4] > 0){ // copy(3,5)
                    offset =  idx[4]*B*pitch[tid] + idx[2]*B*sizeof(int);
                    offset2 = idx[4]*B*V        + idx[2]*B;
                    myCudaMemcpy2DAsync((char*)d_ptr[1]+offset, pitch[tid], d+offset2, V*sizeof(int), range[2]*sizeof(int), range[4], cudaMemcpyHostToDevice, stream[0], tid, 3, 5);   
                }
                if(range[0] > 0){ // copy (1,3)
                    offset =  idx[2]*B*pitch[tid];
                    offset2 = idx[2]*B*V;
                    myCudaMemcpy2DAsync((char*)d_ptr[1]+offset, pitch[tid], d+offset2, V*sizeof(int), range[0]*sizeof(int), range[2], cudaMemcpyHostToDevice, stream[0], tid, 1, 3);   
                    //test<<<1,1>>>(d_ptr[1], pitch[tid], n, 1,3,r);
                }
                if(range[4] > 0){ // copy (5,3)
                    offset =  idx[2]*B*pitch[tid] + idx[4]*B*sizeof(int);
                    offset2 = idx[2]*B*V        + idx[4]*B;
                    myCudaMemcpy2DAsync((char*)d_ptr[1]+offset, pitch[tid], d+offset2, V*sizeof(int), range[4]*sizeof(int), range[2], cudaMemcpyHostToDevice, stream[0], tid, 5, 3);   
                }

                if((width[0] > 0) && (width[1] > 0)){
                    kernel(r, idx[1], idx[0], width[1], width[0], n, d_ptr[tid], pitch[tid], stream[0]); // outer-up-left-grid (2,1)
                    if(debug==1)test<<<1,1>>>(d_ptr[1], pitch[tid], n, 2,1, r);
                }
                if((width[0] > 0) && (width[3] > 0)){
                    kernel(r, idx[3], idx[0], width[3], width[0], n, d_ptr[tid], pitch[tid], stream[0]); // outer-up-right-grid (4,1)
                    if(debug==1)test<<<1,1>>>(d_ptr[1], pitch[tid], n, 4,1, r);
                }
                if((width[1] > 0) && (width[4] > 0)){
                    kernel(r, idx[1], idx[4], width[1], width[4], n, d_ptr[tid], pitch[tid], stream[0]); // outer-right-up-grid (2,5)
                    if(debug==1)test<<<1,1>>>(d_ptr[1], pitch[tid], n, 2,5, r);
                }
                if((width[3] > 0) && (width[4] > 0)){
                    kernel(r, idx[3], idx[4], width[3], width[4], n, d_ptr[tid], pitch[tid], stream[0]); // outer-right-down-grid (4,5)
                    if(debug==1)test<<<1,1>>>(d_ptr[1], pitch[tid], n, 4,5, r);
                }

				/////////////////////////////////////////
                myCudaDeviceSynchronize(tid, "wait phase3");
                /////////////////////////////////////////

                offset =  idx[1]*B*sizeof(int);
				offset2 = idx[1]*B;
                myCudaMemcpy2D(d+offset2, V*sizeof(int), (char*)d_ptr[1]+offset, pitch[tid], (range[1]+range[2]+range[3])*sizeof(int), n, cudaMemcpyDeviceToHost, tid,-1,-1);
                
				/////////////////////////////////////////
				#pragma omp barrier
				/////////////////////////////////////////
				
				if(range[0] > 0){
                    offset = idx[0];
					myCudaMemcpy2D((char*)d_ptr[1]+offset, pitch[tid], d+offset, V*sizeof(int), range[0]*sizeof(int), n, cudaMemcpyHostToDevice, tid, -2, -2);
                }
                if(range[4] > 0){
                    offset =  idx[4]*B*sizeof(int);
					offset2 = idx[4]*B;
					myCudaMemcpy2D((char*)d_ptr[1]+offset, pitch[tid], d+offset2, V*sizeof(int), range[4]*sizeof(int), n, cudaMemcpyHostToDevice, tid, -3, -3);
                }
                if(debug==1)test<<<1,1>>>(d_ptr[1], pitch[tid], n, -1,-1, r);
                //test<<<1,1>>>(d_ptr[1], pitch[tid], n, -1,-1, r);
            }
        }
        cudaFree(d_ptr[tid]);
    }
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
    int inRange = (tidx < n) && (tidy < n) && (b_x < block_width*B) && (b_y < block_height*B); 
    //int inRange = (tidx < n) && (tidy < n); 
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



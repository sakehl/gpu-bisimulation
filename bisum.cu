#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "LTS.h"

#define DEBUG
// #define METHOD1

#ifdef DEBUG
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#else
gpuErrchk(ans) ans
#endif
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


//Step 1: reset values and pick a block
__global__ void pick_block(int N, bool* stable, bool* marks, int* current_block, 
                            int* next_numbers) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

// Reset the markings of previous round.
    if(id < N) {
        marks[id] = false;
        next_numbers[id] = -1;
        if(stable[id] == false) {
            // atomicCAS(current_block, -1, id);
            *current_block = id;
        }
   }
}

// Step 2: Mark the states which can reach the current block && set the current block to stable
__global__ void mark(int M, int* sources, int* targets, bool* marks, int* current_block,
                        int* blocks) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < M)
        if(blocks[targets[id]] == *current_block)
            marks[sources[id]] = true;
}

//Step 3: Elect a leader for new blocks that will be splitted
__global__ void leaderElect(int N,
                             bool* stable, bool* marks, int* blocks, int* next_numbers) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // Reset the markings of previous round.
    if(id < N)
        if(marks[blocks[id]] != marks[id]) {
            //Should be an atomic CAS for safety
            next_numbers[blocks[id]] = id;
        }
}

// Step 4: Split the blocks, update the blocks of the split off states
__global__ void split(int N, bool* stable, 
                        bool* marks, int* blocks, int* next_numbers, int* current_block) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < 1 && *current_block != -1)
        stable[*current_block] = true;

    // Reset the markings of previous round.
    if(id < N) {
        stable[*current_block] = true;

        if(marks[blocks[id]] != marks[id]) {
            stable[blocks[id]] = false;
            blocks[id] = next_numbers[blocks[id]];
            stable[blocks[id]] = false;
        }
     }
}


//This alternative combines step 3 and 4, using an atomic.
__global__ void leader_and_split(int N, bool* stable, 
                        bool* marks, int* blocks, int* next_numbers, int* current_block) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < 1 && *current_block != -1)
        stable[*current_block] = true;

    // Reset the markings of previous round.
    if(id < N)
        if(marks[blocks[id]] != marks[id]) {
            atomicCAS(&next_numbers[blocks[id]], -1, id);
            stable[blocks[id]] = false;
            blocks[id] = next_numbers[blocks[id]];
            stable[blocks[id]] = false;
        }
}

void run_bisum(int N, int M, int* sources, int* targets){
    //Device sources and targets
    int *d_sources, *d_targets;

    // All states have a mark and block.
    bool *marks;
    int *blocks;
    bool *d_marks;
    int *d_blocks;

    // All blocks have a next_number (which is the next leader)
    // and indicate if they are stable
    int* next_numbers;
    bool* stable;
    int* d_next_numbers;
    bool* d_stable;

    //The current block, undefined (-1) in the beginning
    int c = -1;
    int *d_c;

    //Initialize the states & blocks

    marks = (bool*)malloc(sizeof(bool) * N);
    blocks = (int*)malloc(sizeof(int) * N);
    next_numbers = (int*)malloc(sizeof(int) * N);
    stable = (bool*)malloc(sizeof(bool) * N);

    for(int i = 0; i < N; i++){
        marks[i] = false;
        blocks[i] = 0;
        next_numbers[i] = -1;
        stable[i] = true;
    }
    stable[0] = false;
    

    // Allocate device memory
    gpuErrchk( cudaMalloc((void**)&d_sources, sizeof(int) * M) );
    gpuErrchk( cudaMalloc((void**)&d_targets, sizeof(int) * M) );

    gpuErrchk( cudaMalloc((void**)&d_marks, sizeof(bool) * N) );
    gpuErrchk( cudaMalloc((void**)&d_blocks, sizeof(int) * N) );
    gpuErrchk( cudaMalloc((void**)&d_next_numbers, sizeof(int) * N) );
    gpuErrchk( cudaMalloc((void**)&d_stable, sizeof(bool) * N) );

    gpuErrchk( cudaMalloc((void**)&d_c, sizeof(int)) );

    // Transfer data from host to device memory
    gpuErrchk( cudaMemcpy(d_sources, sources, sizeof(int) * M, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_targets, targets, sizeof(int) * M, cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMemcpy(d_marks, marks, sizeof(bool) * N, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_blocks, blocks, sizeof(int) * N, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_next_numbers, next_numbers, sizeof(int) * N, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_stable, stable, sizeof(bool) * N, cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMemcpy(d_c, &c, sizeof(int), cudaMemcpyHostToDevice) );


    // Executing kernel 
    // vector_add<<<1,1>>>(d_out, d_a, d_b, N);
    int threads_N = 2;
    int blocks_N = (N + threads_N -1) / threads_N;

    int threads_M = 2;
    int blocks_M = (M + threads_M -1) / threads_M;

    int iter = 0;
    do {
        iter++;
        //Set current block to undefined
        c = -1;
        gpuErrchk( cudaMemcpy(d_c, &c, sizeof(int), cudaMemcpyHostToDevice) );

        // __global__ void pick_block(bool* stable, bool* marks, int* current_block) 
        // Pick the block to split
        pick_block<<<blocks_N, threads_N>>>(N, d_stable, d_marks, d_c, d_next_numbers);

        //Loop over the transitions to mark with the current block.
        mark<<<blocks_M, threads_M>>>(M, d_sources, d_targets, d_marks, d_c, d_blocks);
#ifdef METHOD1
        //Elect the leaders
        leaderElect<<<blocks_N, threads_N>>>(N, d_stable, d_marks, d_blocks, d_next_numbers);

        //Split of the marked blocks, that differ from the block leader
        split<<<blocks_N, threads_N>>>(N, d_stable, d_marks, d_blocks, d_next_numbers, d_c);
#else
        //Alternative that combines the two previous steps
        leader_and_split<<<blocks_N, threads_N>>>(N, d_stable, d_marks, d_blocks, d_next_numbers, d_c);
#endif

        //Get back the current block
        gpuErrchk( cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost) );

#ifdef DEBUG        
        printf("iter: %i, c: %i\n", iter, c);
#endif
    } while( c != -1 && iter < 100 );

    // Transfer data back to host memory
    gpuErrchk( cudaMemcpy(blocks, d_blocks, sizeof(int) * N, cudaMemcpyDeviceToHost) );

#ifdef DEBUG
    for(int i = 0; i < N; i++) {
        printf("id: %i, block: %i\n", i, blocks[i]);
    }
#endif

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    

    // Deallocate device memory
    gpuErrchk( cudaFree(d_sources) );
    gpuErrchk( cudaFree(d_targets) );
    gpuErrchk( cudaFree(d_marks) );
    gpuErrchk( cudaFree(d_blocks) );
    gpuErrchk( cudaFree(d_next_numbers) );
    gpuErrchk( cudaFree(d_stable) );

    // Deallocate host memory
    free(stable); 
    free(next_numbers); 
    free(blocks);
    free(marks); 
}

int main(){
    // LTS data("data/cwi_1_2.aut");
    LTS data(example2);

    run_bisum(data.n, data.m, data.sources, data.targets);

    return 0;
}

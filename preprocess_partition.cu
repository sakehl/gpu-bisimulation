#include "preprocess_partition.h"
// //Step 1: reset the marks

// Step 2: Mark the states that have the current action as outgoing
__global__ void mark_p(int M, int* sources,
            int* labels, bool* marks, int current_action, int* blocks) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < M)
        if(labels[i] == current_action)
            marks[sources[i]] = true;
}

//Step 3: Elect a leader for new blocks that will be splitted
__global__ void leaderElect_p(int N, bool* marks, int* blocks, int* next_numbers) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N) {
        if(marks[blocks[i]] != marks[i]) {
            //Should/Could be an atomic CAS for safety
            next_numbers[blocks[i]] = i;
        }
    }
}

// Step 4: Split the blocks, update the blocks of the split off states
__global__ void split_p(int N, bool* marks, int* blocks, int* next_numbers) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        if(marks[blocks[i]] != marks[i]) {
            blocks[i] = next_numbers[blocks[i]];
        }
    }
}

void make_partition(int N, int M, int L, int* sources, int* targets, int* labels, int* blocks){
    //Make the marks array
    bool* marks;
    int* next_numbers;
    gpuErrchk(cudaMalloc((void **) &marks, N * sizeof(bool)));
    gpuErrchk(cudaMalloc((void **) &next_numbers, N * sizeof(int)));

    //Setup threads
    int threads_M = 32;
    int blocks_M = (M + threads_M -1) / threads_M;
    int threads_N = 32;
    int blocks_N = (N + threads_N -1) / threads_N;

    //Go over each action label
    for(int a = 0; a<L; a++){
        //Reset the marks
        cudaMemset(marks, 0, N * sizeof(bool));

        mark_p<<<blocks_M, threads_M>>>(M, sources, labels, marks, a, blocks);
        leaderElect_p<<<blocks_N, threads_N>>>(N, marks, blocks, next_numbers);
        split_p<<<blocks_N, threads_N>>>(N, marks, blocks, next_numbers);

#ifdef DEBUG2
        int blocks_h[N];
        bool marks_h[N];
        int next_h[N];

        gpuErrchk(cudaMemcpy(blocks_h, blocks, sizeof(int) * N,
            cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(marks_h, marks, sizeof(bool) * N,
            cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(next_h, next_numbers, sizeof(int) * N,
            cudaMemcpyDeviceToHost));

        std::cout << "State Partition after action " << a << std::endl;
        std::cout << "State | Block | Marks | Next" << std::endl;
        for(int i=0; i<N; i++){
            printf("%5d | %5d | %5d | %4d\n", i, blocks_h[i],
                marks_h[i], next_h[i]);
        }
#endif

    }

    gpuErrchk(cudaFree(next_numbers));
    gpuErrchk(cudaFree(marks));
}
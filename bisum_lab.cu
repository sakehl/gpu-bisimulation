#define DEBUG

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "LTS.h"

#include <chrono> 
#include <iostream>
#include <fstream>

//Step 0: Set the correct (un)stable blocks, all the blocks the blocks, that
// have atleast one state are in the beginning unstable.
__global__ void set_stable(int N, bool* stable, int* block){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        stable[block[i]] = false;
    }
}


// Step 1: reset mark and pick a block
__global__ void pick_block(int N, bool* stable, bool* mark, int* current_block) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Reset the markings of previous round.
    if(i < N) {
        mark[i] = false;

        if(!stable[i])
            // atomicCAS(current_block, -1, i);
            *current_block = i;
   }
}
//Step 1a: reset marks

// Step 2: Mark the states which can reach the current block 
// && set the current block to stable
__global__ void mark(int M, int* source, int* target, int* order, int* marks_offset,
    bool* stable, bool* marks, int* current_block, int* block) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < M) {
        if(block[target[i]] == *current_block) {
            // Represents the transition source[i] ->labels[i] target[i]
            marks[marks_offset[source[i]] + order[i]] = true;
       }
    }

    //Set current block to stable
    if (i < 1 && *current_block != -1)
        stable[*current_block] = true;
}

// Step 3: Check for every transition if markings between leader is different
// and elect it as a new leader
__global__ void compare_markings(int M, int* source, int* order, int* marks_offset,
    bool* mark, bool* marks, int* block, int* next_number){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < M){
        if( marks[marks_offset[      source[i] ] + order[i]] != 
            marks[marks_offset[block[source[i]]] + order[i]]){

            mark[source[i]] = true;
            next_number[block[source[i]]] = source[i];
        }
    }
}

// Step 4: Split the block, update the block of the split off states
__global__ void split(int N, bool* stable,
            bool* mark, int* block, int* next_number, int* current_block) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        if(mark[i]) {
            stable[block[i]] = false;
            block[i] = next_number[block[i]];
            stable[block[i]] = false;
            stable[*current_block] = false;
       }
   }
}


int run_bisum_lab(int N, int M, int L, int* source, int* target, int* order, 
    int* block, int* marks_offset, int marks_length){

    //Setting the block sizes (threads per block) and nr of block
    int threads_N = 32;
    int blocks_N = (N + threads_N -1) / threads_N;

    int threads_M = 32;
    int blocks_M = (M + threads_M -1) / threads_M;
    
    //All states have a mark and marks array, also it has the block number
    bool *mark_d;
    bool *marks_d;
    gpuErrchk( cudaMalloc((void**)&mark_d, sizeof(bool) * N) );
    gpuErrchk( cudaMalloc((void**)&marks_d, sizeof(bool) * marks_length) );

    // All block have a next_number (which is the next leader)
    // and indicate if they are stable
    int* next_number_d;
    bool* stable_d;
    gpuErrchk( cudaMalloc((void**)&next_number_d, sizeof(int) * N) );
    gpuErrchk( cudaMalloc((void**)&stable_d, sizeof(bool) * N) );

    gpuErrchk( cudaMemset(stable_d, 1, sizeof(bool) * N) );
    set_stable<<<blocks_N, threads_N>>>(N, stable_d, block);

    //The current block, undefined (-1) in the beginning
    int c = -1;
    int *c_d;
    gpuErrchk( cudaMalloc((void**)&c_d, sizeof(int)) );
    gpuErrchk( cudaMemcpy(c_d, &c, sizeof(int), cudaMemcpyHostToDevice) );

    chrono::time_point<chrono::high_resolution_clock> start, end;

    start = chrono::high_resolution_clock::now();

    int iter = 0;
    // Executing kernel
    do {
        iter++;
        //Set current block to undefined
        c = -1;
        gpuErrchk( cudaMemcpy(c_d, &c, sizeof(int), cudaMemcpyHostToDevice) );

        // Step1: Pick the block to split
        pick_block<<<blocks_N, threads_N>>>(N, stable_d, mark_d, c_d);
        //Step 1a: reset marks
        gpuErrchk( cudaMemset(marks_d, 0, sizeof(bool) * marks_length) );

        //Loop over the transitions to mark with the current block.
        mark<<<blocks_M, threads_M>>>(M, source, target, order, marks_offset, 
            stable_d, marks_d, c_d, block);

        //Compare markings and elect new leaders
        compare_markings<<<blocks_M, threads_M>>>(M, source, order, marks_offset, 
            mark_d, marks_d, block, next_number_d);

        //Split of the marked block
        split<<<blocks_N, threads_N>>>(N, stable_d, mark_d, block, next_number_d, c_d);

        //Get back the current block
        gpuErrchk( cudaMemcpy(&c, c_d, sizeof(int), cudaMemcpyDeviceToHost) );

        if(iter == 1)
        {
            end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
            double time = duration.count() / 1000.0;
            printf("iter: %i, c: %i time: %g\n", iter, c, time);
        }
#ifdef DEBUG2        
        printf("iter: %i, c: %i\n", iter, c);
#endif
    } while( c != -1 && iter < 10*N );
  
    if(c != -1){
        cout << "WARNING: We passed a reasonable number of iterations("<< iter <<"), but we are not stable yet." << endl;
        return -1;
    }


    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Deallocate device memory
    gpuErrchk( cudaFree(marks_d) );
    gpuErrchk( cudaFree(mark_d) );
    gpuErrchk( cudaFree(next_number_d) );
    gpuErrchk( cudaFree(stable_d) );
    gpuErrchk( cudaFree(c_d) );

    return iter;
}

int main(int argc, char *argv[]){
    LTS data;
    bool check = false;
    bool time = false;
    bool out = false;
    string out_fn;
    string in_fn;
    chrono::time_point<chrono::high_resolution_clock> start, end;
    double time_load, time_preprocess, time_alg, time_total;
    int blocks_remaining;
    
    string s_check ("--check");
    string s_time ("--time");
    string s_out ("--out");
    if(argc > 2){
        for(int i =2; i < argc; i++){
            if(s_check.compare(argv[i]) == 0)
                check = true;
            if(s_time.compare(argv[i]) == 0)
                time = true;
            if(s_out.compare(argv[i]) == 0){
                out = true;
                time = true;
                i++;
                if(i < argc)
                    out_fn = argv[i];
                else{
                    printf("Need input file\n");
                    exit(1);
                }
            }
        }
    }
    if(time)
        start = chrono::high_resolution_clock::now();



    if(argc < 1){
        exit(1);
        printf("Need input file\n");
    } else{
        in_fn = argv[1];
        data.Init(in_fn);
    }

    if(time){
        end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        time_load = duration.count() / 1000.0;
        start = chrono::high_resolution_clock::now();
        printf("Input done in %g\n", time_load);
    }

    data.init_device();
    data.preprocess();

    if(time){
        end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        time_preprocess = duration.count() / 1000.0;
        start = chrono::high_resolution_clock::now();
        printf("Preprocess done in %g\n", time_preprocess);
    }

    if(check){
        data.print_states(10);
        data.print_transitions(10);
    }

    int iter = run_bisum_lab(data.N, data.M, data.L, data.source_d,
        data.target_d, data.order_d, data.block_d, data.marks_offset_d,
        data.marks_length);

    if(time){
        end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        time_alg = duration.count() / 1000.0;
        start = chrono::high_resolution_clock::now();
        printf("Alg done in %g\n", time_alg);
    }

    if(time){
        time_total = time_load + time_preprocess + time_alg;
        printf("------------Timings (ms)------------\n");
        printf("Iter | Loading   | Preprocess | Algorithm | Total \n");
        printf("%d, %g, %g, %g, %g\n",
         iter, time_load, time_preprocess, time_alg, time_total);
    }

    if(check || out){
        int block[data.N];

        gpuErrchk( cudaMemcpy(block, data.block_d, sizeof(int) * data.N, cudaMemcpyDeviceToHost) );

        sort(block, block +data.N);
        vector<int> unique_count;
        unique_count.clear();
        unique_copy(block, block + data.N, back_inserter(unique_count));
        blocks_remaining = unique_count.size();

        printf("Remaing blocks: %d\n", blocks_remaining);
    }

    

    if(out){
        ofstream myfile;
        myfile.open (out_fn, ios_base::app);
        myfile << in_fn << "," << iter << "," <<
         time_load << "," << time_preprocess << "," <<
         time_alg << "," << time_total << "," <<
         blocks_remaining <<
         endl;
        myfile.close();
    }


    return 0;
}
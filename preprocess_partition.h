#ifndef PREPROCESS_PARTITION_H
#define PREPROCESS_PARTITION_H
#include "utility.h"
#include <stdio.h>

__global__ void mark_p(int M, int* sources,
            int* labels, bool* marks, int current_action, int* blocks);
__global__ void leaderElect_p(int N, bool* marks, int* blocks, int* next_numbers);
__global__ void split_p(int N, bool* marks, int* blocks, int* next_numbers);

void make_partition(int N, int M, int L, int* sources, int* targets, int* labels, int* blocks);

#endif
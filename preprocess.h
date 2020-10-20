#ifndef PREPROCESS_H
#define PREPROCESS_H
#include "utility.h"

#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

//Used to increase all values of a thrust iterator by one.
struct incr : public thrust::unary_function<int,int>
{
    __host__ __device__
    int operator()(int x) { return x+1; }
};

//Calculates action switch
__global__ void calc_switch(int M, int* sources, int* labels, 
    int* action_switch);

void sort_transitions(int M, int* sources, int* targets, int* labels);

//Given the transistions (sources and labels), fill the order,
//nr_mark and mark_offset on the device and return total marks_length
//N is the number states, M is the number of transitions.
int make_marks_offset(int N, int M, int* sources, int* labels, int* order, 
    int* nr_marks, int* marks_offset);

void test();

#endif
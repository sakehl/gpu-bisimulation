#include "preprocess.h"

//Calculates action switch
__global__ void calc_switch(int M, int* sources, int* labels, 
        int* action_switch){
     int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < M) {
        if(i == 0 || sources[i] != sources[i - 1] || labels[i] == labels[i-1]){
            action_switch[i] = 0;
        } else{
            action_switch[i] = 1;
        }
    }
};

void sort_transitions(int M, int* sources, int* targets, int* labels){
    //Make a tupple itterator from (sources, labels, targets)
    thrust::device_ptr<int> sources_th(sources);
    thrust::device_ptr<int> targets_th(targets);
    thrust::device_ptr<int> labels_th(labels);

    auto first = thrust::make_zip_iterator(
    thrust::make_tuple(sources_th, labels_th, targets_th));
    auto last  = thrust::make_zip_iterator(
    thrust::make_tuple(sources_th + M, labels_th + M, targets_th + M));
    //Ask trust to sort these
    thrust::sort(first, last);
}

//Given the transistions (sources and labels), fill the order,
//nr_marks and marks_offset on the device and return total marks_length
//N is the number states, M is the number of transitions.
int make_marks_offset(int N, int M, int* sources, int* labels, int* order, 
        int* nr_marks, int* marks_offset){
    int threads_M = 32;
    int blocks_M = (M + threads_M -1) / threads_M;
    //Make the action_switch array (size M) on the device
    int *action_switch_d;
    gpuErrchk(cudaMalloc((void **) &action_switch_d, M * sizeof(int)));

    //Call kernel calc_switch_flags to calculate flags and action_switch
    //(total work M)
    calc_switch<<<blocks_M, threads_M>>>(M, sources, labels, 
       action_switch_d);

    //Do a segmented inclusive prefix sum (scan) to calculate order (thrust)
    thrust::device_ptr<int> action_switch_th(action_switch_d);
    thrust::device_ptr<int> order_th(order);
    thrust::device_ptr<int> sources_th(sources);

    thrust::inclusive_scan_by_key(sources_th, sources_th + M, 
        action_switch_th, order_th);

    //Initialize nr_marks to zeros
    //With a reduce and scatter from thrust, we can calculate the nr_marks
    gpuErrchk(cudaMemset(nr_marks, 0, N * sizeof(int)));
    thrust::device_ptr<int> nr_marks_th(nr_marks);
    thrust::device_ptr<int> marks_offset_th(marks_offset);

    thrust::device_ptr<int> keys = thrust::device_malloc(N * sizeof(int));
    thrust::device_ptr<int> values = thrust::device_malloc(N * sizeof(int));

    auto new_end = thrust::reduce_by_key(sources_th,
            sources_th + M, action_switch_th, keys, values);

    thrust::transform(values, new_end.second, values, incr());
    thrust::scatter(values, new_end.second,
           keys, nr_marks_th);

    //Do an exclusive prefix sum on nr_marks to calculate marks_offset (thrust)
    thrust::exclusive_scan(nr_marks_th, nr_marks_th + N, marks_offset_th);

    //Calculate marks_length from marks_offset and nr_marks
    int marks_length = marks_offset_th[N-1] + nr_marks_th[N-1];

    // printf("Marks | Offset | Keys | Values | Source | Action\n");
    // for(int i=0; i<30 && i < N;i++){
    //     int mark = nr_marks_th[i];
    //     int offset = marks_offset_th[i];
    //     int key = keys[i];
    //     int value = values[i];
    //     int source = sources_th[i];
    //     int action = action_switch_th[i];
    //     printf("%5d | %6d | %4d | %6d | %6d | %6d\n",
    //      mark, offset, key, value, source, action);
    
    // }

    gpuErrchk(cudaFree(action_switch_d) );

    return marks_length;
}

void test(){
     int N = 4;
     int M = 8;
     int a = 0; int b = 1; int c =2;
     int sources[M] = {0, 1, 3, 0, 1, 3, 0, 1};
     int labels[M]  = {a, a, c, a, b, c, c, c};
     int targets[M] = {3, 0, 0, 1, 3, 3, 3, 1};

     int *sources_d, *labels_d, *targets_d;

     gpuErrchk(cudaMalloc((void **) &sources_d, M * sizeof(int)));
     gpuErrchk(cudaMalloc((void **) &labels_d, M * sizeof(int)));
     gpuErrchk(cudaMalloc((void **) &targets_d, M * sizeof(int)));
     gpuErrchk(cudaMemcpy(sources_d, sources, sizeof(int) * M,
        cudaMemcpyHostToDevice));
     gpuErrchk(cudaMemcpy(labels_d, labels, sizeof(int) * M,
        cudaMemcpyHostToDevice));
     gpuErrchk(cudaMemcpy(targets_d, targets, sizeof(int) * M,
        cudaMemcpyHostToDevice));
     //Sort transitions
     sort_transitions(M, sources_d, targets_d, labels_d);

     //Calculate order,
     int order[M];
     int nr_marks[N];
     int marks_offset[N];
     int *order_d, *nr_marks_d, *marks_offset_d;
     gpuErrchk(cudaMalloc((void **) &order_d, M * sizeof(int)));
     gpuErrchk(cudaMalloc((void **) &nr_marks_d, N * sizeof(int)));
     gpuErrchk(cudaMalloc((void **) &marks_offset_d, N * sizeof(int)));

     int marks_length = make_marks_offset(N, M, sources_d, labels_d, 
        order_d, nr_marks_d, marks_offset_d);

     gpuErrchk(cudaMemcpy(sources, sources_d, sizeof(int) * M,
        cudaMemcpyDeviceToHost));
     gpuErrchk(cudaMemcpy(labels, labels_d, sizeof(int) * M,
        cudaMemcpyDeviceToHost));
     gpuErrchk(cudaMemcpy(targets, targets_d, sizeof(int) * M,
        cudaMemcpyDeviceToHost));
     gpuErrchk(cudaMemcpy(order, order_d, sizeof(int) * M,
        cudaMemcpyDeviceToHost));
     gpuErrchk(cudaMemcpy(nr_marks, nr_marks_d, sizeof(int) * N,
        cudaMemcpyDeviceToHost));
     gpuErrchk(cudaMemcpy(marks_offset, marks_offset_d, sizeof(int) * N,
        cudaMemcpyDeviceToHost));

     //Print all the results
     int a_char = static_cast<int>('a');
     for(int i = 0; i< M; i++){
            char l = static_cast<char>(labels[i] + a_char);
                 std::cout << "(" << sources[i] 
                     << ", " << l
                     << ", " << targets[i]
                     << ") " 
                     << order[i]
                     << std::endl;;
            }
            std::cout << std::endl;

     for(int i = 0; i< N; i++){
                 std::cout << i
                     << ", " << nr_marks[i]
                     << ", " << marks_offset[i]
                     << std::endl;;
            }
            std::cout << "Total marks length: " << marks_length << std::endl;

     gpuErrchk( cudaFree(order_d) );
     gpuErrchk( cudaFree(nr_marks_d) );
     gpuErrchk( cudaFree(marks_offset_d) );
     gpuErrchk( cudaFree(sources_d) );
     gpuErrchk( cudaFree(labels_d) );
     gpuErrchk( cudaFree(targets_d) );
}
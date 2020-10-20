#include "LTS.h"


LTS::LTS(){

}

LTS::LTS(string file){
    Init(file);
}

LTS::LTS(Example ex){
    Init(ex);
}

void LTS::Init(string file){
    ifstream inFile;
    inFile.open(file);
    if (!inFile) {
        cerr << "Unable to open file '" << file << "'";
        exit(1);   // call system to stop
    }

    cout << "Reading file '" << file << "'" << endl;

    processHeader(&inFile);

    L = 0;

    source = (int*)malloc(sizeof(int) * M);
    target = (int*)malloc(sizeof(int) * M);
    label  = (int*)malloc(sizeof(int) * M);

    for(int i = 0; i < M; i++){
        processLine(&inFile, i);
        // cout << "source: " << source[i] << "target: " << target[i] << endl;
    }

    cout << "Transitions: " << M << endl;
    cout << "States: " << N << endl;

    inFile.close();
    cout << "File closed" << endl;
}

void LTS::Init(Example ex){
    switch(ex) {
        case example1: {
            N = 5;
            M = 4;
            L = 1;

            label_map["-"] = 0;
            reverse_label_map[0] = "-";

            source = (int*)malloc(sizeof(int) * M);
            target = (int*)malloc(sizeof(int) * M);
            label  = (int*)malloc(sizeof(int) * M);
            source[0] = 0; target[0] = 3; label[0] = 0;    
            source[1] = 1; target[1] = 4; label[0] = 0;
            source[2] = 1; target[2] = 2; label[0] = 0;
            source[3] = 2; target[3] = 1; label[0] = 0;
            break;
        }

        case example2: {
            N = 9;
            M = 7;
            L = 3;

            int A = 0;
            int B = 1;
            int C = 2;
            label_map["A"] = 0;
            label_map["B"] = 1;
            label_map["C"] = 2;
            reverse_label_map[0] = "A";
            reverse_label_map[1] = "B";
            reverse_label_map[2] = "C";

            source = (int*)malloc(sizeof(int) * M);
            target = (int*)malloc(sizeof(int) * M);
            label  = (int*)malloc(sizeof(int) * M);

            //Our input LTS
            source[0] = 0;target[0] = 1;label[0] = A;
            source[1] = 0;target[1] = 2;label[1] = A;
            source[2] = 1;target[2] = 3;label[2] = B;
            source[3] = 2;target[3] = 4;label[3] = C;
            source[4] = 5;target[4] = 6;label[4] = A;
            source[5] = 6;target[5] = 7;label[5] = B;
            source[6] = 6;target[6] = 8;label[6] = C;
            break;
        }
        case example3: {
            N = 4;
            M = 8;
            L = 3;

            int A = 0; int B = 1; int C =2;
            label_map["A"] = 0;
            label_map["B"] = 1;
            label_map["C"] = 2;
            reverse_label_map[0] = "A";
            reverse_label_map[1] = "B";
            reverse_label_map[2] = "C";

            source = (int*)malloc(sizeof(int) * M);
            target = (int*)malloc(sizeof(int) * M);
            label  = (int*)malloc(sizeof(int) * M);

            source = new int[M] {0, 1, 3, 0, 1, 3, 0, 1};
            label  = new int[M] {A, A, C, A, B, C, C, C};
            target = new int[M] {3, 0, 0, 1, 3, 3, 3, 1};
        }
    }
}

LTS::~LTS(){
    free(source);
    free(target);
    free(label);
    free_device();
}

void LTS::processHeader(istream* stream){
    string txt;
    //Process the header, which looks like:
    //  des (0, 2387, 1952)
    //First part, and first state which is zero anyway
    getline(*stream, txt, ',');
    //Number of transisitions
    getline(*stream, txt, ',');
    M = stoi(txt);
    //Number of states
    getline(*stream, txt, ')');
    N = stoi(txt);
    //Rest of line
    getline(*stream, txt);
}

void LTS::processLine(istream* stream, int i){
    string txt;
    char ctxt;

    //The lines look like:
    //  (0, "r1(in(d1,in(d1,in(d1,in(d1)))))", 1)
    // Or like
    // (1, i, 17)
    getline(*stream, txt, '(');

    //This part is the source
    getline(*stream, txt, ',');
    source[i] = stoi(txt);
    getline(*stream, txt, ' ');
    stream->get(ctxt);


    //This part is the label
    if(ctxt == '"'){
        getline(*stream, txt, '"');
        string tmp;
        getline(*stream, tmp, ',');
    } else {
        getline(*stream, txt, ',');
        //Back track to get back the first char
        txt.insert(0, 1, ctxt);
    }

    auto search = label_map.find(txt);
    if(search != label_map.end()) {
        label[i] = search->second;
    } else {
        //Not yet present, give it a fresh integer value
        label_map[txt] = L;
        reverse_label_map[L] = txt;
        label[i] = L;
        L++;
    }

    //This part is the target
    getline(*stream, txt, ')');
    target[i] = stoi(txt);
    getline(*stream, txt);
}

void LTS::init_device(){
    gpuErrchk(cudaMalloc((void **) &source_d, M * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &label_d, M * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &target_d, M * sizeof(int)));

    gpuErrchk(cudaMalloc((void **) &order_d, M * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &nr_mark_d, N * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &marks_offset_d, N * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &block_d, N * sizeof(int)));

    gpuErrchk(cudaMemset(block_d, 0, N * sizeof(int)));

    device_initialized = true;
    to_device();
}

void LTS::to_device(){
    if(!device_initialized)
        return;

    gpuErrchk(cudaMemcpy(source_d, source, sizeof(int) * M,
     cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(label_d, label, sizeof(int) * M,
     cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(target_d, target, sizeof(int) * M,
     cudaMemcpyHostToDevice));
}

void LTS::to_host(){
    if(!device_initialized)
        return;

    gpuErrchk(cudaMemcpy(source, source_d, sizeof(int) * M,
     cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(label, label_d, sizeof(int) * M,
     cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(target, target_d, sizeof(int) * M,
     cudaMemcpyDeviceToHost));
}

void LTS::free_device(){
    if(!device_initialized)
        return;

    gpuErrchk(cudaFree(source_d));
    gpuErrchk(cudaFree(label_d));
    gpuErrchk(cudaFree(target_d));

    gpuErrchk(cudaFree(order_d));
    gpuErrchk(cudaFree(nr_mark_d));
    gpuErrchk(cudaFree(marks_offset_d));
    device_initialized = false;
}

void LTS::preprocess(){
    if(!device_initialized)
        return;

    sort_transitions(M, source_d, target_d, label_d);
    make_partition(N, M, L, source_d, target_d, label_d, block_d);
    marks_length = make_marks_offset(N, M, source_d, label_d, order_d,
     nr_mark_d, marks_offset_d);
}

void LTS::print_transitions(int max){
    if(max == -1)
        max = M;

    to_host();
    int order[M];

    gpuErrchk(cudaMemcpy(order, order_d, sizeof(int) * M,
        cudaMemcpyDeviceToHost));

    cout << "Transitions" << endl;
    cout << "Source | Label | Target | Order" << endl;
    for(int i=0; i<M && i<max; i++){
        string l = reverse_label_map[label[i]];
        char ls[l.length()];
        strcpy(ls, l.c_str());
        printf("%6d | %5s | %6d | %5d\n", source[i], ls,
         target[i], order[i]);
    }

}

void LTS::print_states(int max){
    if(!device_initialized)
        return;

    if(max == -1)
        max = N;

    int blocks[N];
    int nr_mark[N];
    int marks_offset[N];

    gpuErrchk(cudaMemcpy(blocks, block_d, sizeof(int) * N,
        cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(nr_mark, nr_mark_d, sizeof(int) * N,
        cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(marks_offset, marks_offset_d, sizeof(int) * N,
        cudaMemcpyDeviceToHost));

    cout << "State Partition" << endl;
    cout << "State | Block | #Marks | Mark offset " << endl;
    for(int i=0; i<N && i<max; i++){
        printf("%5d | %5d | %6d | %11d\n", i, blocks[i], nr_mark[i],
            marks_offset[i]);
    }
    printf("Total marks length: %d\n", marks_length);

}
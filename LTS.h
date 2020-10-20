#ifndef LTS_H
#define LTS_H

#include <bits/stdc++.h>
#include <unordered_map>
#include <fstream>

#include "utility.h"
#include "preprocess_partition.h"
#include "preprocess.h"

using namespace std;


enum Example {example1, example2, example3};

class LTS{
public:
    int N; // Number of states
    int M; // Number of transitions
    int L; // Number of different label
    int* source;
    int* target;
    int* label;
    unordered_map<string, int> label_map;
    unordered_map<int, string> reverse_label_map;

    //These values will come on the device
    int* source_d;
    int* target_d;
    int* label_d;
    int* order_d;

    int* block_d;
    int* nr_mark_d;
    int* marks_offset_d;
    int marks_length;

    LTS();
    LTS(string file);
    LTS(Example ex);

    void Init(string file);
    void Init(Example ex);

    ~LTS();

    void init_device();
    void to_device();
    void to_host();

    void preprocess();

    void free_device();

    void print_transitions(int max = -1);
    void print_states(int max = -1);

private:
	void processHeader(istream* stream);
	void processLine(istream* stream, int i);

    bool device_initialized = false;
};

#endif
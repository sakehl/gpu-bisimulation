# gpu-bisimulation
A GPU Bisimulation algorithm

Run 
```
nvcc bisum_lab.cu preprocess_partition.cu LTS.cu preprocess.cu -o bisum_lab
```
to compile it.

The program can be used like
```
./bisum_lab data/cwi_1_2.aut --out
```
Then the timings will be stored in `results.txt`.

Any other [Aldebaran](https://cadp.inria.fr/man/aldebaran.html) file (.aut) can be loaded as well.
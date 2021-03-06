# Assignment 1: Matrix Multiplication

Assignment No 1 for the multi-core programming course. Modify previous code examples to calculate matrix multiplication. It has to be programmed in three ways:

- In CPU without threads.
- In CPU with threads.
- In CUDA using blocks and threads.

Each program has to do the following:

1. Multiply 2 NxN matrices. N has to be set to 1000, 2000, and 4000.
2. Fill the matrices with natural numbers; use row major order.
3. Validate that the result from the matrix multiplication in CPU is the same as in GPU.

For the CPU version with threads, test performance varying the number of threads depending on your CPU. For the GPU version, test the performance with different block and thread configurations. Vary the blocks in multiples of 32, and the threads in multiples of 128. 

Include a Pdf file with the results of the testing for each case. Measure the time spent for the calculations, and the overall time of each code. Include the characteristics of the computer where the testing was performed; mention the type, speed, number of cores, etc, both for the CPU and the GPU. Add conclusions and thoughts after analyzing the results.

NOTE: Push only the Pdf file and the source code. Modify the .gitignore file as needed.

Rubric:

1. Matrices are properly initialized. *Complete*
2. Matrices are properly multiplied both in CPU and in GPU. *Complete*
3. GPU code is initialized correctly. *Complete*
4. The code is readable and commented as needed. *Complete*
5. The report file has tables with the performance data for the different configurations, as well as for the speedup obtained. *Incomplete*
6. The report file has the computer's characteristics, as well as the conclusions. *Incomplete*

**NOTES**

1. Speedups and conclusions are missing.
2. Did you perform several calculations on the GPU as you did on the CPU?
3. There was an error in your GPU code. Specifically, you did not include a *sum*, which makes the kernel produce mistaken multiplications. Newer GPUs have a protection against this, but it incurs an overhead. 
4. The results were expected in a table, both for readability and to be able to compare easily.
5. We have to check the 4000x4000 code, it should not take more than a few tens of minutes to execute.
6. Why do you think the 4000x4000 code did not work as expected?

**GRADE: 85**
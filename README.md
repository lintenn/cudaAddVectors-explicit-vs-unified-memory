# cudaAddVectors-explicit-vs-unified-memory
## Performance comparison of two different forms of memory management in CUDA
### By Luis Miguel García Marín

I have made two kernel implementations for vector addition (AddVectorsInto), one in which I use unified memory (cudaMallocManaged, cudaMemPrefetchAsync ...) and another in which I explicitly declare the vectors in CPU and GPU (malloc, cudaMalloc) and perform the same transfers manually with cudaMemcpy.

After the implementation of both (both codes can be seen within the repository), I have decided to compare their performance, with the help of the Visual Profiler. So let's take a look: 

### Unified memory version
![Visual Profiler de la versión unified memory](https://user-images.githubusercontent.com/74145538/135771280-66aead40-eeb0-48e8-95fe-f783979d23b0.png)

### Explicit memory version with cudaMemcpy transactions
![Visual Profiler de la versión explicit memory](https://user-images.githubusercontent.com/74145538/135771317-108a3001-50c3-40e3-9c17-22fefcf28691.png)

We see that both memory allocations (Malloc) and a transaction are made from the device (GPU) to the host (CPU) with the resulting vector (c).

We can see that both versions have similar total execution times (395 ms for unified memory and 420 ms for explicit memory), but that of unified memory is a little lower than that of explicit memory, even manually controlling the allocations of memory and transactions on CPU and GPU. The efficiency of Nvidia's unified memory usage (with the help of prefetches) is quite surprising, as it is comparable (and slightly better) to my manual implementation of memory allocation and transactions with cudaMemcpy.

However, we see that the unified memory version uses more main threads (5) than the explicit memory version (3). And not only that, but we also see that the unified memory version performs many more memory operations (64) than the explicit memory version (1), we can corroborate this by consulting the output of the nsys profiler command:

### Unified memory version
![nsys de unified-memory](https://user-images.githubusercontent.com/74145538/135771355-46938946-1418-421f-8a01-c3d960df758a.png)

### Explicit memory version with cudaMemcpy transactions
![nsys de explicit-memory](https://user-images.githubusercontent.com/74145538/135771360-6f330e9d-67ce-468d-8dc4-67ea86357dc1.png)

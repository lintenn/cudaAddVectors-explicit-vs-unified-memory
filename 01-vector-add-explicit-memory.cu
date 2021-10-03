#include <stdio.h>

// Luis Miguel García Marín

__global__
void initWith(float num, float *a, int N)
{

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    a[i] = num;
  }
}

__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

void checkElementsAre(float target, float *vector, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(vector[i] != target)
    {
      printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
      exit(1);
    }
  }
  printf("Success! All values calculated correctly.\n");
}

int main()
{
  int deviceId;
  int numberOfSMs;

  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

  const int N = 2<<24;
  size_t size = N * sizeof(float);

  //float *a;
  //float *b;
  float *c;
  // Punteros que apuntarán a posiciones de memoria con los que trabajará la CPU

  //a = (float *) malloc(size);
  //b = (float *) malloc(size);
  c = (float *) malloc(size);
  // Asignamos espacio en la memoria que trabaja la CPU

  float *da;
  float *db;
  float *dc;
  // Punteros que apuntan a posiciones de la memoria de vídeo con los que trabajará la GPU
  
  cudaMalloc(&da, size);
  cudaMalloc(&db, size);
  cudaMalloc(&dc, size);
  // Asignamos espacio en la memoria de vídeo de la GPU

  size_t threadsPerBlock;
  size_t numberOfBlocks;

  threadsPerBlock = 256;
  numberOfBlocks = 32 * numberOfSMs;

  cudaError_t addVectorsErr;
  cudaError_t asyncErr;

  initWith<<<numberOfBlocks, threadsPerBlock>>>(3, da, N);
  initWith<<<numberOfBlocks, threadsPerBlock>>>(4, db, N);
  initWith<<<numberOfBlocks, threadsPerBlock>>>(0, dc, N);

  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(dc, da, db, N);
  // Les pasamos los punteros de device (da,db,dc) y no de host (a,b,c)

  addVectorsErr = cudaGetLastError();
  if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  //cudaMemcpy(a, da, size, cudaMemcpyDeviceToHost);
  //cudaMemcpy(b, db, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(c, dc, size, cudaMemcpyDeviceToHost);
  // Copiamos los resultados (del último vector, el resultante) de la GPU a la CPU

  checkElementsAre(7, c, N);

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  // Liberamos la memoria de vídeo
  
  //free(a);
  //free(b);
  free(c);
  // Liberamos la memoria principal que trabajaba la CPU
}

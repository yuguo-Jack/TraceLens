// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// See LICENSE for license information.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#define HIP_CHECK(x) do { auto e = (x); if(e != hipSuccess) { std::fprintf(stderr,"%s\n",hipGetErrorString(e)); return 2; } } while(0)
__global__ void qualification_add(const float* a, const float* b, float* c, int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) c[i]=a[i]+b[i];
}
int main() {
  const int n=1<<20;
  std::vector<float> host(n,1.0f);
  float *a,*b,*c;
  HIP_CHECK(hipMalloc(&a,n*sizeof(float))); HIP_CHECK(hipMalloc(&b,n*sizeof(float))); HIP_CHECK(hipMalloc(&c,n*sizeof(float)));
  HIP_CHECK(hipMemcpy(a,host.data(),n*sizeof(float),hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(b,host.data(),n*sizeof(float),hipMemcpyHostToDevice));
  for(int k=0;k<3;k++) qualification_add<<<(n+255)/256,256>>>(a,b,c,n);
  HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(host.data(),c,n*sizeof(float),hipMemcpyDeviceToHost));
  for(float x:host) if(x!=2.0f) return 3;
  std::printf("PASS: %d float32 values, %d algorithmic bytes/dispatch\n",n,n*12);
  HIP_CHECK(hipFree(a)); HIP_CHECK(hipFree(b)); HIP_CHECK(hipFree(c));
}

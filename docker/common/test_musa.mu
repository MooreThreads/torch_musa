#include <cmath>
#include <iostream>
__global__ void axpy(float* x, float* y, float a) {
  y[threadIdx.x] = a * x[threadIdx.x];
}
int main(int argc, char* argv[]) {
  const int kDataLen = 4;
  float a = 2.0f;
  float host_x[kDataLen] = {1.0f, 2.0f, 3.0f, 4.0f};
  float host_y[kDataLen];
  // Copy input data to device.
  float* device_x;
  float* device_y;
  musaMalloc(&device_x, kDataLen * sizeof(float));
  musaMalloc(&device_y, kDataLen * sizeof(float));
  musaMemcpy(
      device_x, host_x, kDataLen * sizeof(float), musaMemcpyHostToDevice);
  // Launch the kernel.
  axpy<<<1, kDataLen>>>(device_x, device_y, a);
  // Copy output data to host.
  musaDeviceSynchronize();
  musaMemcpy(
      host_y, device_y, kDataLen * sizeof(float), musaMemcpyDeviceToHost);
  // Check the results.
  for (int i = 0; i < kDataLen; ++i) {
    if (std::fabs(host_y[i] - a * host_x[i]) > 1e-6) {
      std::cout << "error!";
      break;
    }
  }
  musaFree(device_x);
  musaFree(device_y);
  musaDeviceReset();
  return 0;
}

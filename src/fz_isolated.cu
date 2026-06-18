#include <cuda_runtime.h>
#include <dirent.h>
#include <stdint.h>
#include <sys/stat.h>
#include <thrust/copy.h>
#include <chrono>
#include <cub/cub.cuh>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "../include/kernel/lorenzo_var.cuh"
#include "../include/utils/cuda_err.cuh"

#define UINT32_BIT_LEN 32
#define VERIFICATION
#define DEBUG


typedef struct {
    dim3      inputDimension;
    size_t    inputSize;
    size_t    dataTypeLen;
    size_t    paddingDataTypeLen;
    int       dataChunkSize;
    size_t    quantizationCodeByteLen;

    //float*    deviceInput;
    //uint16_t* deviceCompressedOutput;
    uint32_t* deviceBitFlagArr;
    uint16_t* deviceQuantizationCode;
    uint32_t* deviceOffsetCounter;
    uint32_t* deviceStartPosition;
    uint32_t* deviceCompressedSize;
    uint16_t* deviceDecompressedQuantizationCode;
    //float*    deviceDecompressedOutput;
    bool*     deviceSignNum;
    double    range;

} device_variables_t;



template <typename T>
void verify_data(T* xdata, T* odata, size_t len)
{
    double max_odata = odata[0], min_odata = odata[0];
    double max_xdata = xdata[0], min_xdata = xdata[0];
    double max_abserr = max_abserr = fabs(xdata[0] - odata[0]);

    double sum_0 = 0, sum_x = 0;
    for (size_t i = 0; i < len; i++) sum_0 += odata[i], sum_x += xdata[i];

    double mean_odata = sum_0 / len, mean_xdata = sum_x / len;
    double sum_var_odata = 0, sum_var_xdata = 0, sum_err2 = 0, sum_corr = 0, rel_abserr = 0;

    double max_pwrrel_abserr = 0;
    size_t max_abserr_index  = 0;
    for (size_t i = 0; i < len; i++) {
        max_odata = max_odata < odata[i] ? odata[i] : max_odata;
        min_odata = min_odata > odata[i] ? odata[i] : min_odata;

        max_xdata = max_xdata < odata[i] ? odata[i] : max_xdata;
        min_xdata = min_xdata > xdata[i] ? xdata[i] : min_xdata;

        float abserr = fabs(xdata[i] - odata[i]);
        if (odata[i] != 0) {
            rel_abserr        = abserr / fabs(odata[i]);
            max_pwrrel_abserr = max_pwrrel_abserr < rel_abserr ? rel_abserr : max_pwrrel_abserr;
        }
        max_abserr_index = max_abserr < abserr ? i : max_abserr_index;
        max_abserr       = max_abserr < abserr ? abserr : max_abserr;
        sum_corr += (odata[i] - mean_odata) * (xdata[i] - mean_xdata);
        sum_var_odata += (odata[i] - mean_odata) * (odata[i] - mean_odata);
        sum_var_xdata += (xdata[i] - mean_xdata) * (xdata[i] - mean_xdata);
        sum_err2 += abserr * abserr;
    }
    double std_odata = sqrt(sum_var_odata / len);
    double std_xdata = sqrt(sum_var_xdata / len);
    double ee        = sum_corr / len;

    // s->len = len;

    // s->odata.max = max_odata;
    // s->odata.min = min_odata;
    double inputRange = max_odata - min_odata;
    // s->odata.std = std_odata;

    // s->xdata.max = max_xdata;
    // s->xdata.min = min_xdata;
    // s->xdata.rng = max_xdata - min_xdata;
    // s->xdata.std = std_xdata;

    // s->max_err.idx    = max_abserr_index;
    // s->max_err.abs    = max_abserr;
    // s->max_err.rel    = max_abserr / s->odata.rng;
    // s->max_err.pwrrel = max_pwrrel_abserr;

    // s->reduced.coeff = ee / std_odata / std_xdata;
    double mse   = sum_err2 / len;
    // s->reduced.NRMSE = sqrt(s->reduced.MSE) / s->odata.rng;
    double psnr  = 20 * log10(inputRange) - 10 * log10(mse);
    std::cout << "PSNR: " << psnr << std::endl;
}

long GetFileSize(std::string fidataTypeLename)
{
    struct stat stat_buf;
    int         rc = stat(fidataTypeLename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

template <typename T>
T* read_binary_to_new_array(const std::string& fname, size_t dtype_dataTypeLen)
{
    std::ifstream ifs(fname.c_str(), std::ios::binary | std::ios::in);
    if (not ifs.is_open()) {
        std::cerr << "fail to open " << fname << std::endl;
        exit(1);
    }
    auto _a = new T[dtype_dataTypeLen]();
    ifs.read(reinterpret_cast<char*>(_a), std::streamsize(dtype_dataTypeLen * sizeof(T)));
    ifs.close();
    return _a;
}

template <typename T>
void write_array_to_binary(const std::string& fname, T* const _a, size_t const dtype_dataTypeLen)
{
    std::ofstream ofs(fname.c_str(), std::ios::binary | std::ios::out);
    if (not ofs.is_open()) return;
    ofs.write(reinterpret_cast<const char*>(_a), std::streamsize(dtype_dataTypeLen * sizeof(T)));
    ofs.close();
}

__global__ void compressionFusedKernel(
    const uint32_t* __restrict__ in,
    uint32_t* __restrict__ out,
    uint32_t* deviceOffsetCounter,
    uint32_t* deviceBitFlagArr,
    uint32_t* deviceStartPosition,
    uint32_t* deviceCompressedSize)
{
    // 32 x 32 data chunk size with one padding for each row, overall 4096 bytes per chunk
    __shared__ uint32_t dataChunk[32][33];
    __shared__ uint16_t byteFlagArray[257];
    __shared__ uint32_t bitflagArr[8];
    __shared__ uint32_t startPosition;

    uint32_t byteFlag = 0;
    uint32_t v;

    v = in[threadIdx.x +  threadIdx.y * 32 + blockIdx.x * 1024];
    __syncthreads();

#ifdef DEBUG
    dataChunk[threadIdx.y][threadIdx.x] = v;
    if(threadIdx.y == 0 && threadIdx.x == 0 && blockIdx.x == 1)
    {
        printf("original data:\n");
        for (int tmpIdx = 0; tmpIdx < 32; tmpIdx++)
        {
            printf("%u\t", dataChunk[0][tmpIdx]);
        }
        printf("\n");
    }
#endif

#pragma unroll 32
    for (int i = 0; i < 32; i++)
    {
        dataChunk[threadIdx.y][i] = __ballot_sync(0xFFFFFFFFU, v & (1U << i));
    }
    __syncthreads();

#ifdef DEBUG
    if(threadIdx.y == 0 && threadIdx.x == 0 && blockIdx.x == 1)
    {
        printf("shuffled data:\n");
        for (int tmpIdx = 0; tmpIdx < 32; tmpIdx++)
        {
            printf("%u\t", dataChunk[0][tmpIdx]);
        }
        printf("\n");
    }
#endif

    // generate byteFlagArray
    if (threadIdx.x < 8) 
    {
#pragma unroll 4
        for (int i = 0; i < 4; i++)
        { 
            byteFlag |= dataChunk[threadIdx.x * 4 + i][threadIdx.y]; 
        }
        byteFlagArray[threadIdx.y * 8 + threadIdx.x] = byteFlag > 0;
    }
    __syncthreads();

    // generate bitFlagArray
    uint32_t buffer;
    if (threadIdx.y < 8) {
        buffer                  = byteFlagArray[threadIdx.y * 32 + threadIdx.x];
        bitflagArr[threadIdx.y] = __ballot_sync(0xFFFFFFFFU, buffer);
    }
    __syncthreads();

#ifdef DEBUG
    if(threadIdx.y == 0 && threadIdx.x == 0 && blockIdx.x == 1)
    {
        printf("bit flag array: %u\n", bitflagArr[0]);
    }
#endif

    // write back bitFlagArray to global memory
    if (threadIdx.x < 8 && threadIdx.y == 0) {
        deviceBitFlagArr[blockIdx.x * 8 + threadIdx.x] = bitflagArr[threadIdx.x];
    }


    int blockSize = 256;
    int tid = threadIdx.x + threadIdx.y * 32;

    // prefix summation, up-sweep
    int prefixSumOffset = 1;
#pragma unroll 8
    for (int d = 256 >> 1; d > 0; d = d >> 1)
    {
        if (tid < d)
        {
            int ai = prefixSumOffset * (2 * tid + 1) - 1;
            int bi = prefixSumOffset * (2 * tid + 2) - 1;
            byteFlagArray[bi] += byteFlagArray[ai];
        }
        __syncthreads();
        prefixSumOffset *= 2;
    }

    // clear the last element
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        byteFlagArray[blockSize] = byteFlagArray[blockSize - 1];
        byteFlagArray[blockSize - 1] = 0;
    }
    __syncthreads();

    // prefix summation, down-sweep
#pragma unroll 8
    for (int d = 1; d < 256; d *= 2)
    {
        prefixSumOffset >>= 1;
        if (tid < d)
        {
            int ai = prefixSumOffset * (2 * tid + 1) - 1;
            int bi = prefixSumOffset * (2 * tid + 2) - 1;

            uint32_t t = byteFlagArray[ai];
            byteFlagArray[ai] = byteFlagArray[bi];
            byteFlagArray[bi] += t;
        }
        __syncthreads();
    }

#ifdef DEBUG
    if(threadIdx.y == 0 && threadIdx.x == 0 && blockIdx.x == 1)
    {
        printf("byte flag array:\n");
        for (int tmpIdx = 0; tmpIdx < 32; tmpIdx++)
        {
            printf("%u\t", byteFlagArray[tmpIdx]);
        }
        printf("\n");
    }
#endif

    // use atomicAdd to reserve a space for compressed data chunk
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        startPosition = atomicAdd(deviceOffsetCounter, byteFlagArray[blockSize] * 4);
        deviceStartPosition[blockIdx.x] = startPosition;
        deviceCompressedSize[blockIdx.x] = byteFlagArray[blockSize];
    }
    __syncthreads();

    // write back the compressed data based on the startPosition
    int flagIndex = floorf(tid / 4);
    if(byteFlagArray[flagIndex + 1] != byteFlagArray[flagIndex])
    {
        out[startPosition + byteFlagArray[flagIndex] * 4 + tid % 4] = dataChunk[threadIdx.x][threadIdx.y];
    } 
}

__global__ void decompressionFusedKernel(
    uint32_t* deviceInput,
    uint32_t* deviceOutput,
    uint32_t* deviceBitFlagArr,
    uint32_t* deviceStartPosition)
{
    // allocate shared byte flag array
    __shared__ uint32_t dataChunk[32][33];
    __shared__ uint16_t byteFlagArray[257];
    __shared__ uint32_t startPosition;

    // there are 32 x 32 uint32_t in this data chunk
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int bid = blockIdx.x;

    // transfer bit flag array to byte flag array
    uint32_t bitFlag = 0;
    if(threadIdx.x < 8 && threadIdx.y == 0) 
    {
        bitFlag = deviceBitFlagArr[bid * 8 + threadIdx.x];
#pragma unroll 32
        for(int tmpInd = 0; tmpInd < 32; tmpInd++)
        {
            byteFlagArray[threadIdx.x * 32 + tmpInd] = (bitFlag & (1U<<tmpInd)) > 0;
        }
    }
    __syncthreads();

    int prefixSumOffset = 1;
    int blockSize = 256;

    // prefix summation, up-sweep
#pragma unroll 8
    for (int d = 256 >> 1; d > 0; d = d >> 1)
    {
        if (tid < d)
        {
            int ai = prefixSumOffset * (2 * tid + 1) - 1;
            int bi = prefixSumOffset * (2 * tid + 2) - 1;
            byteFlagArray[bi] += byteFlagArray[ai];
        }
        __syncthreads();
        prefixSumOffset *= 2;
    }

    // clear the last element
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        byteFlagArray[blockSize] = byteFlagArray[blockSize - 1];
        byteFlagArray[blockSize - 1] = 0;
    }
    __syncthreads();

    // prefix summation, down-sweep
#pragma unroll 8
    for (int d = 1; d < 256; d *= 2)
    {
        prefixSumOffset >>= 1;
        if (tid < d)
        {
            int ai = prefixSumOffset * (2 * tid + 1) - 1;
            int bi = prefixSumOffset * (2 * tid + 2) - 1;

            uint32_t t = byteFlagArray[ai];
            byteFlagArray[ai] = byteFlagArray[bi];
            byteFlagArray[bi] += t;
        }
        __syncthreads();
    }

#ifdef DEBUG
    if(threadIdx.y == 0 && threadIdx.x == 0 && blockIdx.x == 1)
    {
        printf("decompressed byte flag array:\n");
        for (int tmpIdx = 0; tmpIdx < 32; tmpIdx++)
        {
            printf("%u\t", byteFlagArray[tmpIdx]);
        }
        printf("\n");
    }
#endif

    // initialize the shared memory to all 0
    dataChunk[threadIdx.y][threadIdx.x] = 0;
    __syncthreads();

    // get the start position
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        startPosition = deviceStartPosition[bid];
    }
    __syncthreads();

    // write back shuffled data to shared mem
    int byteFlagInd = tid / 4;
    if(byteFlagArray[byteFlagInd + 1] != byteFlagArray[byteFlagInd])
    {
        dataChunk[threadIdx.x][threadIdx.y] = deviceInput[startPosition + byteFlagArray[byteFlagInd] * 4 + tid % 4];
    }
    __syncthreads();

    // store the corresponding uint32 to the register buffer
    uint32_t buffer = dataChunk[threadIdx.y][threadIdx.x];
    __syncthreads();

    // bitshuffle (reverse)
#pragma unroll 32
    for (int i = 0; i < 32; i++)
    {
        dataChunk[threadIdx.y][i] = __ballot_sync(0xFFFFFFFFU, buffer & (1U << i));
    }
    __syncthreads();

#ifdef DEBUG
    if(threadIdx.y == 0 && threadIdx.x == 0 && blockIdx.x == 1)
    {
        printf("decompressed data:\n");
        for (int tmpIdx = 0; tmpIdx < 32; tmpIdx++)
        {
            printf("%u\t", dataChunk[0][tmpIdx]);
        }
        printf("\n");
    }
#endif

    // write back to global memory
    deviceOutput[tid + bid * blockDim.x * blockDim.y] = dataChunk[threadIdx.y][threadIdx.x];
}

void fzgpu_init(float* data, int x, int y, int z, device_variables_t* dev)
{
    dev->inputDimension = dim3(x, y, z);
    dev->inputSize = x * y * z * sizeof(float);
    dev->dataTypeLen = dev->inputSize / sizeof(float);

    int  blockSize = 16;
    dev->quantizationCodeByteLen = dev->dataTypeLen * 2;  // quantization code length in unit of bytes
    dev->quantizationCodeByteLen = dev->quantizationCodeByteLen % 4096 == 0 ? dev->quantizationCodeByteLen : dev->quantizationCodeByteLen - dev->quantizationCodeByteLen % 4096 + 4096;
    dev->paddingDataTypeLen = dev->quantizationCodeByteLen / 2;
    dev->dataChunkSize = dev->quantizationCodeByteLen % (blockSize * UINT32_BIT_LEN) == 0 ? dev->quantizationCodeByteLen / (blockSize * UINT32_BIT_LEN) : int(dev->quantizationCodeByteLen / (blockSize * UINT32_BIT_LEN)) + 1;

    printf("valores: dataTypeLen: %zu, quantizationCodeByteLen: %zu, paddingDataTypeLen: %zu, dataChunkSize: %d\n", dev->dataTypeLen, dev->quantizationCodeByteLen, dev->paddingDataTypeLen, dev->dataChunkSize);
    printf("deviceStartPosition: %zu \n",sizeof(uint32_t) * dev->quantizationCodeByteLen / 4096);
    printf("deviceBitFlagArr: %zu \n",sizeof(uint32_t) * dev->dataChunkSize); 
    printf("Field size as float: %zu \n", dev->inputSize); 
    //CHECK_CUDA(cudaMalloc((void**)&dev->deviceQuantizationCode, sizeof(uint16_t) * dev->paddingDataTypeLen));
    //CHECK_CUDA(cudaMalloc((void**)&dev->deviceSignNum, sizeof(bool) * dev->paddingDataTypeLen));
    CHECK_CUDA(cudaMalloc((void**)&dev->deviceBitFlagArr, sizeof(uint32_t) * dev->dataChunkSize));
    //CHECK_CUDA(cudaMalloc((void**)&dev->deviceDecompressedQuantizationCode, sizeof(uint16_t) * dev->paddingDataTypeLen));
    
    //CHECK_CUDA(cudaMalloc((void**)&dev->deviceOffsetCounter, sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc((void**)&dev->deviceStartPosition, sizeof(uint32_t) * floor(dev->quantizationCodeByteLen / 4096)));
    //CHECK_CUDA(cudaMalloc((void**)&dev->deviceCompressedSize, sizeof(uint32_t) * floor(dev->quantizationCodeByteLen / 4096)));
    
    //CHECK_CUDA(cudaMemset(dev->deviceQuantizationCode, 0, sizeof(uint16_t) * dev->paddingDataTypeLen));
    CHECK_CUDA(cudaMemset(dev->deviceBitFlagArr, 0, sizeof(uint32_t) * dev->dataChunkSize));
    //CHECK_CUDA(cudaMemset(dev->deviceDecompressedQuantizationCode, 0, sizeof(uint16_t) * dev->paddingDataTypeLen));

    //CHECK_CUDA(cudaMemset(dev->deviceOffsetCounter, 0, sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(dev->deviceStartPosition, 0, sizeof(uint32_t) * floor(dev->quantizationCodeByteLen / 4096)));
    //CHECK_CUDA(cudaMemset(dev->deviceCompressedSize, 0, sizeof(uint32_t) * floor(dev->quantizationCodeByteLen / 4096)));
    
    return;
}

void fzgpu_free(device_variables_t* dev)
{
    //CHECK_CUDA(cudaFree(dev->deviceInput));
    //CHECK_CUDA(cudaFree(dev->deviceCompressedOutput));
    CHECK_CUDA(cudaFree(dev->deviceBitFlagArr));
    //CHECK_CUDA(cudaFree(dev->deviceQuantizationCode));
    //CHECK_CUDA(cudaFree(dev->deviceOffsetCounter));
    CHECK_CUDA(cudaFree(dev->deviceStartPosition));
    //CHECK_CUDA(cudaFree(dev->deviceCompressedSize));
    //CHECK_CUDA(cudaFree(dev->deviceDecompressedQuantizationCode));
    //CHECK_CUDA(cudaFree(dev->deviceDecompressedOutput));
    //CHECK_CUDA(cudaFree(dev->deviceSignNum));

    return;
}

void fzgpu_compress(float* data_in, uint16_t* data_out, device_variables_t* dev, double eb, cudaStream_t stream)
{

    
    cudaMemset(dev->deviceBitFlagArr, 0, sizeof(uint32_t) * dev->dataChunkSize);
    cudaMemset(dev->deviceStartPosition, 0, sizeof(uint32_t) * floor(dev->quantizationCodeByteLen / 4096));    
        
    CHECK_CUDA(cudaMalloc((void**)&dev->deviceSignNum, sizeof(bool) * dev->paddingDataTypeLen));
    CHECK_CUDA(cudaMalloc((void**)&dev->deviceOffsetCounter, sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc((void**)&dev->deviceQuantizationCode, sizeof(uint16_t) * dev->paddingDataTypeLen));
    CHECK_CUDA(cudaMalloc((void**)&dev->deviceCompressedSize, sizeof(uint32_t) * floor(dev->quantizationCodeByteLen / 4096)));
    
    CHECK_CUDA(cudaMemset(dev->deviceOffsetCounter, 0, sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(dev->deviceQuantizationCode, 0, sizeof(uint16_t) * dev->paddingDataTypeLen));
    CHECK_CUDA(cudaMemset(dev->deviceCompressedSize, 0, sizeof(uint32_t) * floor(dev->quantizationCodeByteLen / 4096)));
    
    float timeElapsed;
    uint32_t  offsetSum;

    std::chrono::time_point<std::chrono::system_clock> compressionStart, compressionEnd;

    dim3 block(32, 32);
    dim3 grid(floor(dev->paddingDataTypeLen / 2048));  // divided by 2 is because the file is transformed from uint32 to uint16


    compressionStart = std::chrono::system_clock::now();

    cusz::experimental::launch_construct_LorenzoI_var<float, uint16_t, float>(data_in, dev->deviceQuantizationCode, dev->deviceSignNum, dev->inputDimension, eb * dev->range, timeElapsed, stream);

    // bitshuffle kernel
    compressionFusedKernel<<<grid, block>>>((uint32_t*)dev->deviceQuantizationCode, (uint32_t*)data_out, dev->deviceOffsetCounter, dev->deviceBitFlagArr, dev->deviceStartPosition, dev->deviceCompressedSize);

    cudaDeviceSynchronize();
    compressionEnd = std::chrono::system_clock::now();

    CHECK_CUDA(cudaMemcpy(&offsetSum, dev->deviceOffsetCounter, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    printf("original size: %zu\n", dev->inputSize);
    printf("compressed size: %ld\n", sizeof(uint32_t) *dev-> dataChunkSize + offsetSum * sizeof(uint32_t) + sizeof(uint32_t) * int(dev->quantizationCodeByteLen / 4096));
    printf("compression ratio: %f\n", float(dev->inputSize) / float(sizeof(uint32_t) * dev->dataChunkSize + offsetSum * sizeof(uint32_t) + sizeof(uint32_t) * floor(dev->quantizationCodeByteLen / 4096)));

    std::chrono::duration<double> compressionTime = compressionEnd - compressionStart;
    
    std::cout << "compression e2e time: " << compressionTime.count() << " s\n";
    std::cout << "compression e2e throughput: " << float(dev->inputSize) / 1024 / 1024 /1024 / compressionTime.count() << " GB/s\n";
    
#ifndef VERIFICATION
    CHECK_CUDA(cudaFree(dev->deviceQuantizationCode));
#endif
    CHECK_CUDA(cudaFree(dev->deviceSignNum));
    CHECK_CUDA(cudaFree(dev->deviceOffsetCounter));
    CHECK_CUDA(cudaFree(dev->deviceCompressedSize));
    return;
}

void fzgpu_decompress(uint16_t* data_in, float* data_out, device_variables_t* dev, double eb, cudaStream_t stream)
{

    CHECK_CUDA(cudaMalloc((void**)&dev->deviceSignNum, sizeof(bool) * dev->paddingDataTypeLen));
    CHECK_CUDA(cudaMalloc((void**)&dev->deviceDecompressedQuantizationCode, sizeof(uint16_t) * dev->paddingDataTypeLen));
    CHECK_CUDA(cudaMemset(dev->deviceDecompressedQuantizationCode, 0, sizeof(uint16_t) * dev->paddingDataTypeLen));

    // cudaMemset(dev->deviceBitFlagArr, 0, sizeof(uint32_t) * dev->dataChunkSize);

    // cudaMemset(dev->deviceStartPosition, 0, sizeof(uint32_t) * floor(dev->quantizationCodeByteLen / 4096));

    float timeElapsed;

    std::chrono::time_point<std::chrono::system_clock> decompressionStart, decompressionEnd;

    dim3 block(32, 32);
    dim3 grid(floor(dev->paddingDataTypeLen / 2048));  // divided by 2 is because the file is transformed from uint32 to uint16


    decompressionStart = std::chrono::system_clock::now();

    // de-bitshuffle kernel
    decompressionFusedKernel<<<grid, block>>>((uint32_t*)data_in, (uint32_t*)dev->deviceDecompressedQuantizationCode, dev->deviceBitFlagArr, dev->deviceStartPosition);

    // de-pre-quantization
    cusz::experimental::launch_reconstruct_LorenzoI_var<float, uint16_t, float>(dev->deviceSignNum, dev->deviceDecompressedQuantizationCode, data_out, dev->inputDimension, eb * dev->range, timeElapsed,  stream);

    cudaDeviceSynchronize();
    decompressionEnd = std::chrono::system_clock::now();

    std::chrono::duration<double> decompressionTime = decompressionEnd - decompressionStart;

    std::cout << "decompression e2e time: " << decompressionTime.count() << " s\n";
    std::cout << "decompression e2e throughput: " << float(dev->inputSize) / 1024 / 1024 /1024 / decompressionTime.count() << " GB/s\n";

#ifndef VERIFICATION
    CHECK_CUDA(cudaFree(dev->deviceDecompressedQuantizationCode));    
#endif
    CHECK_CUDA(cudaFree(dev->deviceSignNum));
    return;
}

void fzgpu_verify(float* field, float* d_field_decompressed, device_variables_t* dev, double eb)
{
    uint16_t* hostQuantizationCode;
    hostQuantizationCode = (uint16_t*)malloc(sizeof(uint16_t) * dev->dataTypeLen);
    CHECK_CUDA(cudaMemcpy(hostQuantizationCode, dev->deviceQuantizationCode, sizeof(uint16_t) * dev->dataTypeLen, cudaMemcpyDeviceToHost));

    // bitshuffle verification
    uint16_t* hostDecompressedQuantizationCode;
    hostDecompressedQuantizationCode = (uint16_t*)malloc(sizeof(uint16_t) * dev->dataTypeLen);
    CHECK_CUDA(cudaMemcpy(hostDecompressedQuantizationCode, dev->deviceDecompressedQuantizationCode, sizeof(uint16_t) * dev->dataTypeLen, cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();

    printf("begin bitshuffle verification\n");
    bool bitshuffleVerify = true;
    for (int tmpIdx = 0; tmpIdx < dev->dataTypeLen; tmpIdx++)
    {
        if(hostQuantizationCode[tmpIdx] != hostDecompressedQuantizationCode[tmpIdx])
        {
            printf("data type len: %zu\n", dev->dataTypeLen);
            printf("verification failed at index: %d\noriginal quantization code: %u\ndecompressed quantization code: %u\n", tmpIdx, hostQuantizationCode[tmpIdx], hostDecompressedQuantizationCode[tmpIdx]);
            bitshuffleVerify = false;
            break;
        }
    }

    free(hostQuantizationCode);
    free(hostDecompressedQuantizationCode);

    // pre-quantization verification
    float* hostDecompressedOutput;
    hostDecompressedOutput = (float*)malloc(sizeof(float) * dev->dataTypeLen);
    CHECK_CUDA(cudaMemcpy(hostDecompressedOutput, d_field_decompressed, sizeof(float) * dev->dataTypeLen, cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();

    bool prequantizationVerify = true;
    if(bitshuffleVerify)
    {
        printf("begin pre-quantization verification\n");
        for (int tmpIdx = 0; tmpIdx < dev->dataTypeLen; tmpIdx++)
        {
            if(std::abs(field[tmpIdx] - hostDecompressedOutput[tmpIdx]) > float(eb * 1.01 * dev->range))
            {
                printf("verification failed at index: %d\noriginal data: %f\ndecompressed data: %f\n", tmpIdx, field[tmpIdx], hostDecompressedOutput[tmpIdx]);
                printf("error is: %f, while error bound is: %f\n", std::abs(field[tmpIdx] - hostDecompressedOutput[tmpIdx]), float(eb * dev->range));
                prequantizationVerify = false;
                break;
            }
        }
    }

    verify_data<float>(hostDecompressedOutput, field, dev->dataTypeLen);

    free(hostDecompressedOutput);
    
    // print verification result
    if(bitshuffleVerify)
    {
        printf("bitshuffle verification succeed!\n");
        if(prequantizationVerify)
        {
            printf("pre-quantization verification succeed!\n");
        }
        else
        {
            printf("pre-quantization verification fail\n");
        }
    }
    else
    {
        printf("bitshuffle verification fail\n");
    }

    return;
}

void runFzgpu(std::string fileName, int x, int y, int z, double eb)
{

    float* hostInput;
    size_t domain_size = x * y * z;
    size_t padded_domain_size = ((domain_size + 3) / 4) * 4;

    hostInput = read_binary_to_new_array<float>(fileName, padded_domain_size);
    
    float* d_field;
    uint16_t* d_field_compressed;
    float* d_field_decompressed;
    
    CHECK_CUDA(cudaMalloc((void**)&d_field, sizeof(float) * padded_domain_size));
    CHECK_CUDA(cudaMalloc((void**)&d_field_compressed, sizeof(uint16_t) * padded_domain_size));
    CHECK_CUDA(cudaMalloc((void**)&d_field_decompressed, sizeof(float) * padded_domain_size));
    
    CHECK_CUDA(cudaMemcpy(d_field, hostInput, sizeof(float) * padded_domain_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_field_decompressed, 0, sizeof(float) * padded_domain_size));
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    device_variables_t* dev = new device_variables_t;
    dev->range = *std::max_element(hostInput , hostInput + padded_domain_size) - *std::min_element(hostInput , hostInput + padded_domain_size);
    
    fzgpu_init(d_field, x, y, z, dev);
    // for (int i = 0; i < 1000; i++) {
        fzgpu_compress(d_field, d_field_compressed, dev, eb, stream);
    // }
    fzgpu_decompress(d_field_compressed, d_field_decompressed, dev, eb, stream);

#ifdef VERIFICATION
    fzgpu_verify(hostInput, d_field_decompressed, dev, eb);
#endif
    
    
    fzgpu_free(dev);

    CHECK_CUDA(cudaFree(d_field));
    CHECK_CUDA(cudaFree(d_field_compressed));
    CHECK_CUDA(cudaFree(d_field_decompressed));

    cudaStreamDestroy(stream);

    delete[] hostInput;

    return;
}

int main(int argc, char* argv[])
{
    using T = float;
    std::string fileName;
    fileName  = std::string(argv[1]);

    int    x  = std::stoi(argv[2]);
    int    y  = std::stoi(argv[3]);
    int    z  = std::stoi(argv[4]);
    double eb = std::stod(argv[5]);

    runFzgpu(fileName, x, y, z, eb);
    return 0;
}


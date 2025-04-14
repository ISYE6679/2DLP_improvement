#include <cuda_runtime.h>
#include "device_launch_parameters.h"
// Comment out unused header to prevent build errors
// #include "cuda_profiler_api.h"
#include <glm/glm.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "FileIO.h"
#include "Auxilary.h"

//Whether Objective function should should minimise distance to point, comment out for minimising linear function
//#define OBJECTIVE_DISTANCE

#define maxSpeed_ 100000000000 // large artificial circular constraint
#define RVO_EPSILON 0.00001f //something close to zero

//#define printInfo
//number of threads in a block. Must be a multiple of 32!
#define BlockDimSize 128

//enum for min max
enum optimisation { MINIMISE, MAXIMISE };

// Structure for coalesced memory access
struct BatchData {
    float4* constraints;     // [constraint_size][batch_size] for coalesced access
    glm::vec2* optimise;    // [batch_size]
    glm::vec2* output;      // [batch_size]
    int size;               // Store size to avoid passing as parameter
    int batches;           // Store batches to avoid passing as parameter
};

// Function to reorganize data for coalesced access
__host__ void reorganizeData(float4* input_constraints, const glm::vec2* input_optimise,
                           BatchData* batch_data, int batches, int size) {
    // Store sizes in the structure
    batch_data->size = size;
    batch_data->batches = batches;

    // Allocate memory with proper alignment
    gpuErrchk(cudaMalloc(&batch_data->constraints, sizeof(float4) * batches * size));
    gpuErrchk(cudaMalloc(&batch_data->optimise, sizeof(glm::vec2) * batches));
    gpuErrchk(cudaMalloc(&batch_data->output, sizeof(glm::vec2) * batches));

    // Use cudaMemcpy2D for better performance with 2D data layout
    const size_t pitch = batches * sizeof(float4);
    for (int i = 0; i < size; i++) {
        gpuErrchk(cudaMemcpy(batch_data->constraints + i * batches,
                            input_constraints + i,
                            sizeof(float4) * batches,
                            cudaMemcpyHostToDevice));
    }

    // Copy optimization parameters using a single memcpy
    glm::vec2* temp_optimise = (glm::vec2*)malloc(sizeof(glm::vec2) * batches);
    if (!temp_optimise) return;
    
    for (int i = 0; i < batches; i++) {
        temp_optimise[i] = input_optimise[0];
    }
    gpuErrchk(cudaMemcpy(batch_data->optimise, temp_optimise,
                         sizeof(glm::vec2) * batches, cudaMemcpyHostToDevice));
    free(temp_optimise);
}

////////////////////////////////////
//Numerical functions

//Determinant of 2 2d vectors
__device__ __forceinline__ float det(const glm::vec2 v1, const glm::vec2 v2)
{
    return (v1.x * v2.y) - (v1.y * v2.x);  // Using direct calculation for better performance
}

//Vector doted with itself
__device__ __forceinline__ float absSq(const glm::vec2 &vector)
{
    return glm::dot(vector, vector);  // Using GLM's optimized dot product
}

//Square of value a
__device__ __forceinline__ float sqr(float a)
{
    return a * a;
}


////////////////////////////////////
//Atomic operation functions

//Custom, optimized float atomic max
__device__ __forceinline__ float atomicMax(float* address, float val)
{
    int* address_as_i = (int*)address;
    int old = *address_as_i;
    int expected;
    do {
        expected = old;
        const float old_val = __int_as_float(old);
        if (val <= old_val) break;  // Early exit if no change needed
        old = atomicCAS(address_as_i, expected, __float_as_int(val));
    } while (expected != old);
    return __int_as_float(old);
}

//Custom, optimized float atomic min
__device__ __forceinline__ float atomicMin(float* address, float val)
{
    int* address_as_i = (int*)address;
    int old = *address_as_i;
    int expected;
    do {
        expected = old;
        const float old_val = __int_as_float(old);
        if (val >= old_val) break;  // Early exit if no change needed
        old = atomicCAS(address_as_i, expected, __float_as_int(val));
    } while (expected != old);
    return __int_as_float(old);
}


////////////////////////////////////
//Block level operations

/** \brief block level sum reduction using warp-level primitives
* \param input_data thread value to be reduced over
* \returns sum reduction written to thread 0, else returns 0 for other threads.
*
* Optimized for Volta+ architectures using warp-level primitives
* not valid for block size greater than 1024 (32*32)
*/
__device__ __forceinline__ int reduce(int input_data) {
    __shared__ int s_ballot_results[BlockDimSize >> 5]; //shared results of the ballots
    const unsigned int lane_id = threadIdx.x & 0x1F;
    const unsigned int warp_id = threadIdx.x >> 5;
    
    // Use warp-level primitives for faster reduction
    int warp_result = __popc(__ballot_sync(__activemask(), input_data));
    
    if (lane_id == 0) {
        s_ballot_results[warp_id] = warp_result;
    }
    __syncthreads();
    
    // Final reduction in first warp
    int block_sum = 0;
    if (threadIdx.x < 32) {
        block_sum = (threadIdx.x < (BlockDimSize >> 5)) ? s_ballot_results[threadIdx.x] : 0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(__activemask(), block_sum, offset);
        }
    }
    
    return (threadIdx.x == 0) ? block_sum : 0;
}

/** \brief compresses the thread varaible input_data using warp shuffles
* \param input_data thread level
* \param compArr shared memory array to store the compressed indices
* \param comp_num the size of the compressed array
*/
__device__ int compress(int input_data, int *compArr) {

    const int tid = threadIdx.x;
    __shared__ int temp[BlockDimSize >> 5]; //stores warp scan results shared
    int int_ret; //value to return

    int temp1 = input_data;
    //scan within warp
    for (int d = 1; d<32; d <<= 1) {
        int temp2 = __shfl_up(temp1, d);
        if (tid % 32 >= d) temp1 += temp2;
    }
    if (tid % 32 == 31) temp[tid >> 5] = temp1;
    __syncthreads();
    //scan of warp sums
    if (threadIdx.x < 32) {
        int temp2 = 0.0f;
        if (tid < blockDim.x / 32)
            temp2 = temp[threadIdx.x];
        for (int d = 1; d<32; d <<= 1) {
            int temp3 = __shfl_up(temp2, d);
            if (tid % 32 >= d) temp2 += temp3;
        }
        if (tid < blockDim.x / 32) temp[tid] = temp2;
    }
    __syncthreads();
    //add to previous warp sums
    if (tid >= 32) temp1 += temp[tid / 32 - 1];
    //compress
    if (input_data == 1) {
        compArr[temp1 - 1] = threadIdx.x;
    }

    //get total number - reduction
    int_ret = reduce(input_data);

    return int_ret;
}





////////////////////////////////////
//Linear program auxillaries

__device__ bool linearProgram1Fractions(const float4 lines_lineNo, const float4 lines_i, const float2 t2, float* tnew, bool *tLeftb)
{
    const glm::vec2 lines_direction_lineNo = glm::vec2(lines_lineNo.x, lines_lineNo.y);
    const glm::vec2 lines_point_lineNo = glm::vec2(lines_lineNo.z, lines_lineNo.w);

    const glm::vec2 lines_direction_i = glm::vec2(lines_i.x, lines_i.y);
    const glm::vec2 lines_point_i = glm::vec2(lines_i.z, lines_i.w);

    const float denominator = det(lines_direction_lineNo, lines_direction_i);
    const float numerator = det(lines_direction_i, lines_point_lineNo - lines_point_i);

    if (fabsf(denominator) <= RVO_EPSILON) {
        // Lines lineNo and i are (almost) parallel.
        if (numerator < 0.0f) {
            return false;
        }
        else {
            //continue, i.e. save a value that is guarateed to not affect the results
            *tnew = -INT_MAX; //an arbitary large value that is larger than tright
            *tLeftb = true;
            return true;
        }
    }

    const float t = numerator / denominator;

    if (denominator >= 0.0f) {
        // Line i bounds line lineNo on the right.
        *tLeftb = false;
    }
    else {
        // Line i bounds line lineNo on the left.
        *tLeftb = true;
    }
    *tnew = t;

    return true;
}

/*
* Solve constrainsts subject to a maximisation/minimisation function
*/
__global__ void lpsolve(BatchData batch_data) {
    //thread index
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int tid = threadIdx.x;

    // Early exit if thread is outside the valid range
    if (index >= batch_data.batches) return;

    //initialize SM
    __shared__ int compArr[BlockDimSize]; 
    __shared__ int s_active_agents;
    __shared__ float4 s_line[BlockDimSize]; 
    __shared__ float2 s_t[BlockDimSize]; 
    __shared__ int s_lineFail[BlockDimSize]; 
    __shared__ glm::vec2 s_newv[BlockDimSize]; 
    __shared__ glm::vec2 s_desv[BlockDimSize]; 
    __shared__ glm::vec2 s_optimiseConst[BlockDimSize];
    enum optimisation optimiseFunc = MAXIMISE;

    // Prefetch optimization data to shared memory
    if (tid < BlockDimSize && index < batch_data.batches) {
        s_optimiseConst[tid] = batch_data.optimise[index];
    }
    __syncthreads();

    //Initialise variables
    if (tid < BlockDimSize) {
        s_desv[tid] = s_optimiseConst[tid]; 
        
        // Avoid division by zero when normalizing
        glm::vec2 norm = s_optimiseConst[tid];
        float length = glm::length(norm);
        if (length > 1e-6f) {
            norm = norm / length;
        } else {
            // Default to upward direction if objective function is zero
            norm = glm::vec2(0.0f, 1.0f);
        }
        
        s_newv[tid] = 1e6f * norm * ((optimiseFunc == MAXIMISE) ? 1.0f : -1.0f); 
        s_lineFail[tid] = -1;
        s_t[tid] = make_float2(-INT_MAX, INT_MAX);
    }
    __syncthreads();

    //number of starting agents
    if (tid == 0) {
        s_active_agents = min(BlockDimSize, batch_data.batches - blockIdx.x * BlockDimSize);
        if (s_active_agents < 0) s_active_agents = 0;
    }
    __syncthreads();

    //loop through all lines in the batch
    for (int i = 0; i < batch_data.size; i++) {
        // Reset t values for this iteration
        if (tid < BlockDimSize) {
            s_t[tid] = make_float2(-INT_MAX, INT_MAX);
        }
        __syncthreads();

        int bthread_data = (index < batch_data.batches && s_lineFail[tid] == -1) ? 1 : 0;

        if (bthread_data == 1) {
            // Load current line info into shared memory with bounds check
            // Using row-major ordering consistent with main() data layout
            const size_t constraint_idx = index * batch_data.size + i;
            if (constraint_idx < batch_data.batches * batch_data.size) {
                s_line[tid] = batch_data.constraints[constraint_idx];
            }

            //check if newVel is satisfied by the constraint line. 1 if not satisfied and requires work, otherwise 0.
            bthread_data = (int)(det(glm::vec2(s_line[tid].x, s_line[tid].y), 
                                  glm::vec2(s_line[tid].z, s_line[tid].w) - s_newv[tid]) > 0.0f);
        }

        //compress through exclusive scan
        int result = compress(bthread_data, compArr);
        //result written to thread0
        if (tid == 0) {
            s_active_agents = result;
        }

        //For compArr to be filled properly
        __syncthreads();

        //calculate the total number of work unit items
        int wu_count = (s_active_agents * i);

        //divide work unit items between threads
        for (int j = 0; j < wu_count; j += blockDim.x) {
            //calculate unique work unit index
            int wu_index = j + tid;

            //do work if there are still wu to complete
            if (wu_index < wu_count) {
                //for each thread work out which agent it is associated with
                int n_tid = compArr[wu_index / i];

                //for each thread work out which line index it should read
                int line_index = wu_index % i;

                //read in the unique agent line combination using the calculated indices
                int newIndex = n_tid + blockIdx.x*blockDim.x;
                // Using row-major ordering consistent with main() data layout
                const size_t line_idx = newIndex * batch_data.size + line_index;
                
                // Bounds check
                if (line_idx < batch_data.batches * batch_data.size) {
                    float4 lines_i = batch_data.constraints[line_idx];

                    //calculate denominator and numerator
                    bool btleft;//whether the t value is left (or right if false)
                    float t;//value of t
                    if (!linearProgram1Fractions(s_line[n_tid], lines_i, s_t[n_tid], &t, &btleft)) {
                        //operation failed
                        s_lineFail[n_tid] = i;
                    }

                    //atomic write tleft and tright to shared memory using an atomic min and max
                    if (btleft) {
                        atomicMax(&s_t[n_tid].x, t);
                    }
                    else {
                        atomicMin(&s_t[n_tid].y, t);
                    }
                }
            }
        }
        //sync to ensure all atomic writes are complete
        __syncthreads();

        //update the new velocity for each active agent
        if (tid < s_active_agents) {
            //New index
            int n_tid = compArr[tid];

            //failure condition if no region of validity
            if (s_t[n_tid].x > s_t[n_tid].y) {
                s_lineFail[n_tid] = i;
            }

            //If not failed up to this point
            if (s_lineFail[n_tid] == -1) {
                // Optimize closest point
                glm::vec2 lineDir = glm::vec2(s_line[n_tid].x, s_line[n_tid].y);
                glm::vec2 linePoint = glm::vec2(s_line[n_tid].z, s_line[n_tid].w);
                
#ifdef OBJECTIVE_DISTANCE
                //for case of minimising distance to point @s_desv
                const float t = glm::dot(lineDir, s_desv[n_tid] - linePoint);

                //best value is to the left of what is allowed
                if (t < s_t[n_tid].x) {
                    s_newv[n_tid] = linePoint + s_t[n_tid].x * lineDir;
                }
                //best value is to the right of what is allowed
                else if (t > s_t[n_tid].y) {
                    s_newv[n_tid] = linePoint + s_t[n_tid].y * lineDir;
                }
                //best value is not on a vertex
                else {
                    s_newv[n_tid] = linePoint + t * lineDir;
                }
#else
                //for case of minimising linear function
                //the objective function
                glm::vec2 fct = s_desv[n_tid];

                glm::vec2 t_left_sln = linePoint + s_t[n_tid].x * glm::vec2(lineDir.x, lineDir.y);
                float t_left_val = glm::dot(fct, t_left_sln);
                
                glm::vec2 t_right_sln = linePoint + s_t[n_tid].y * glm::vec2(lineDir.x, lineDir.y);
                float t_right_val = glm::dot(fct, t_right_sln);
                
                //assign answer from correct t.
                s_newv[n_tid] = ((t_left_val > t_right_val) != (optimiseFunc == (MINIMISE))) ? 
                                t_left_sln : t_right_sln;
#endif // OBJECTIVE_DISTANCE
            }
        }

        //sync to ensure all shared mem writes are complete
        __syncthreads();
    }

    //write to output
    if (index < batch_data.batches && tid < BlockDimSize) {
        batch_data.output[index] = s_newv[tid];
    }
}

int main(int argc, const char* argv[]) {
    int batches = 0; 
    int size = 0; 
    float4* constraintsSingle = NULL; 
    glm::vec2 optimiseSingle; 
    BatchData batch_data = {NULL, NULL, NULL, 0, 0};

    // Parse command line args and read input 
    if (argc != 3) {
        printf("\nIncorrect Number of Arguments!\n");
        printf("Correct Usages/Syntax:\n");
        printf("./program <input file> <number of batches>\n");
        return 1;
    }
    batches = atoi(argv[2]);

    //Input is from file
    printf("Parsing input files... ");
    if (!parseBenchmark(argv[1], &constraintsSingle, &optimiseSingle, &size)) {
        return 1;
    }
    printf("Done\n");
    
    // Validate input parameters
    if (batches <= 0 || size <= 0) {
        printf("Invalid batch size or constraint size\n");
        return 1;
    }

    // Initialize CUDA events 
    cudaEvent_t start, stop;
    float memory_milliseconds = 0;
    float milliseconds = 0;  
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    //start time for memory operations
    gpuErrchk(cudaEventRecord(start));

    // Allocate CUDA memory with proper error handling
    cudaError_t cudaStatus;
    
    // Allocate constraints
    cudaStatus = cudaMalloc(&batch_data.constraints, sizeof(float4) * batches * size);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed for constraints: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    // Allocate optimise
    cudaStatus = cudaMalloc(&batch_data.optimise, sizeof(glm::vec2) * batches);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed for optimise: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(batch_data.constraints);
        return 1;
    }

    // Allocate output
    cudaStatus = cudaMalloc(&batch_data.output, sizeof(glm::vec2) * batches);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed for output: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(batch_data.constraints);
        cudaFree(batch_data.optimise);
        return 1;
    }

    batch_data.size = size;
    batch_data.batches = batches;

    // Create temporary host buffer for constraints - use row-major layout
    float4* temp_constraints = (float4*)malloc(sizeof(float4) * batches * size);
    if (!temp_constraints) {
        printf("Failed to allocate host memory for constraints\n");
        cudaFree(batch_data.constraints);
        cudaFree(batch_data.optimise);
        cudaFree(batch_data.output);
        return 1;
    }

    // Fill constraints in row-major order (batch-major)
    for (int b = 0; b < batches; b++) {
        for (int s = 0; s < size; s++) {
            temp_constraints[b * size + s] = constraintsSingle[s];
        }
    }

    // Copy constraints to device
    gpuErrchk(cudaMemcpy(batch_data.constraints, temp_constraints, 
                         sizeof(float4) * batches * size, cudaMemcpyHostToDevice));
    
    // Create and copy optimization parameters
    glm::vec2* temp_optimise = (glm::vec2*)malloc(sizeof(glm::vec2) * batches);
    if (!temp_optimise) {
        printf("Failed to allocate host memory for optimise\n");
        free(temp_constraints);
        cudaFree(batch_data.constraints);
        cudaFree(batch_data.optimise);
        cudaFree(batch_data.output);
        return 1;
    }
    
    // Fill optimization parameters
    for (int i = 0; i < batches; i++) {
        temp_optimise[i] = optimiseSingle;
    }
    
    // Copy optimization parameters to device
    gpuErrchk(cudaMemcpy(batch_data.optimise, temp_optimise, 
                         sizeof(glm::vec2) * batches, cudaMemcpyHostToDevice));

    // Free temporary host buffers
    free(temp_constraints);
    free(temp_optimise);

    //end time for memory operations
    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&memory_milliseconds, start, stop));

    // Validate block and grid dimensions
    int blockSize = BlockDimSize;
    if (blockSize > 1024) {
        printf("Block size exceeds maximum allowed (1024)\n");
        cudaFree(batch_data.constraints);
        cudaFree(batch_data.optimise);
        cudaFree(batch_data.output);
        return 1;
    }
    
    int maxGridSize;
    cudaDeviceGetAttribute(&maxGridSize, cudaDevAttrMaxGridDimX, 0);
    int gridSize = (batches + blockSize - 1) / blockSize;
    if (gridSize > maxGridSize) {
        printf("Grid size exceeds device maximum\n");
        cudaFree(batch_data.constraints);
        cudaFree(batch_data.optimise);
        cudaFree(batch_data.output);
        return 1;
    }

    // Allocate host memory for results BEFORE kernel launch
    glm::vec2* h_output = (glm::vec2*)malloc(sizeof(glm::vec2) * batches);
    glm::vec2* h_optimise = (glm::vec2*)malloc(sizeof(glm::vec2) * batches);
    if (!h_output || !h_optimise) {
        printf("Failed to allocate host memory for results\n");
        if (h_output) free(h_output);
        if (h_optimise) free(h_optimise);
        cudaFree(batch_data.constraints);
        cudaFree(batch_data.optimise);
        cudaFree(batch_data.output);
        return 1;
    }

    // Launch kernel with timing
    dim3 b(blockSize), g(gridSize);
    
    gpuErrchk(cudaEventRecord(start));
    lpsolve<<<g, b>>>(batch_data);
    
    // Check for kernel launch errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        free(h_output);
        free(h_optimise);
        cudaFree(batch_data.constraints);
        cudaFree(batch_data.optimise);
        cudaFree(batch_data.output);
        return 1;
    }
    
    // Wait for kernel to complete
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy results back from device
    gpuErrchk(cudaMemcpy(h_output, batch_data.output, 
                         sizeof(glm::vec2) * batches, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_optimise, batch_data.optimise, 
                         sizeof(glm::vec2) * batches, cudaMemcpyDeviceToHost));

    // Print results
    for (int i = 0; i < min(1, batches); i++) {
        printf("Batch %i \t Optimal location is x: %f y: %f \t value of %f\n",
               i, h_output[i].x, h_output[i].y,
               glm::dot(h_optimise[i], h_output[i]));
    }

    //------------------------------------------
    //write timing to file
    writeTimingtoFile("timings/timings.txt", size, batches, memory_milliseconds+milliseconds);

    //------------------------------------------
    //cleanup

    //memory - using cudaFree for all CUDA allocated memory
    gpuErrchk(cudaFree(batch_data.constraints));
    gpuErrchk(cudaFree(batch_data.optimise));
    gpuErrchk(cudaFree(batch_data.output));
    free(constraintsSingle);
    free(h_output);
    free(h_optimise);

    // Cleanup CUDA events
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));

    gpuErrchk(cudaDeviceReset());
    return 0;
}

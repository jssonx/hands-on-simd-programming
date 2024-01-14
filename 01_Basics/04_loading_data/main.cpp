#include "immintrin.h" // Include for AVX2, 256-bit operations
#include <chrono>
#include <iostream>

const int ARRAY_SIZE = 8;
const int TEST_ITERATIONS = 100000000;

// Function to initialize array with sequential values
void initializeArray(float* array, int size) {
    for (int i = 0; i < size; ++i) {
        array[i] = static_cast<float>(i);
    }
}

// Function to display SIMD array data
void displaySIMDArray(const char* message, __m256 simdArray) {
    float* data = (float*)&simdArray;
    std::cout << message;
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        std::cout << data[i] << ", ";
    }
    std::cout << std::endl;
}

// Function template to perform and time a specific SIMD operation
template<typename Func>
void performTest(const char* testName, Func testFunction) {
    auto start = std::chrono::high_resolution_clock::now();
    __m256 result;
    for (int i = 0; i < TEST_ITERATIONS; ++i) {
        result = testFunction();
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << testName << " took " << duration.count() << " ms." << std::endl;
}

int main() {
    // Aligned and unaligned data allocation
    float* alignedData = (float*)aligned_alloc(32, ARRAY_SIZE * sizeof(float));
    float* unalignedData = (float*)malloc(ARRAY_SIZE * sizeof(float));

    // Check for allocation failure
    if (!alignedData || !unalignedData) {
        std::cerr << "Memory allocation failed." << std::endl;
        return 1;
    }

    initializeArray(alignedData, ARRAY_SIZE);
    initializeArray(unalignedData, ARRAY_SIZE);

    // Aligned load
    __m256 simdAligned = _mm256_load_ps(alignedData);
    displaySIMDArray("Aligned SIMD Array: ", simdAligned);

    // Unaligned load
    __m256 simdUnaligned = _mm256_loadu_ps(unalignedData);
    displaySIMDArray("Unaligned SIMD Array: ", simdUnaligned);

    // Performance tests using lambda expressions directly
    performTest("setr performance", [&]() -> __m256 {
        return _mm256_setr_ps(alignedData[0], alignedData[1], alignedData[2], alignedData[3],
                              alignedData[4], alignedData[5], alignedData[6], alignedData[7]);
    });
    performTest("aligned load performance", [&]() -> __m256 {
        return _mm256_load_ps(alignedData);
    });
    performTest("unaligned load performance", [&]() -> __m256 {
        return _mm256_loadu_ps(unalignedData);
    });

    // Free allocated memory
    free(alignedData);
    free(unalignedData);

    return 0;
}
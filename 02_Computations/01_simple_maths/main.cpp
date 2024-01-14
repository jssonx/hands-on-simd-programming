#include "immintrin.h" // AVX2, 256 bit operations (8 floats)
#include <chrono>
#include <iostream>

// Function declarations
void displayResult(const char* operation, const float* SIMDdata, int size);

template<typename Func>
long measurePerformance(Func f, int iterations);

int main() {
    // Data preparation
    float data1[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float data2[8] = {101, 102, 103, 104, 105, 106, 107, 108};

    // Load data into SIMD vectors
    __m256 vector1 = _mm256_loadu_ps(data1);
    __m256 vector2 = _mm256_loadu_ps(data2);

    // Addition operation
    std::cout << "----------- Addition ------------" << std::endl;
    __m256 result = _mm256_add_ps(vector1, vector2);
    displayResult("Addition Result", (float*)&result, 8);

    // Measure performance of regular and SIMD addition
    auto regularAdd = [&] {
        float data3[8];
        for (int lane = 0; lane < 8; ++lane) {
            data3[lane] = data1[lane] + data2[lane];
        }
    };
    long durationRegAdd = measurePerformance(regularAdd, 1000000);
    std::cout << "Regular addition took " << durationRegAdd << " ms." << std::endl;

    auto simdAdd = [&] {
        __m256 temp = _mm256_add_ps(vector1, vector2);
    };
    long durationSIMDAdd = measurePerformance(simdAdd, 1000000);
    std::cout << "SIMD addition took " << durationSIMDAdd << " ms." << std::endl;

    // Subtraction operation
    std::cout << "----------- Subtraction ------------" << std::endl;
    result = _mm256_sub_ps(vector1, vector2);
    displayResult("Subtraction Result", (float*)&result, 8);

    // Measure performance of regular and SIMD subtraction
    auto regularSub = [&] {
        float data3[8];
        for (int lane = 0; lane < 8; ++lane) {
            data3[lane] = data1[lane] - data2[lane];
        }
    };
    long durationRegSub = measurePerformance(regularSub, 1000000);
    std::cout << "Regular subtraction took " << durationRegSub << " ms." << std::endl;

    auto simdSub = [&] {
        __m256 temp = _mm256_sub_ps(vector1, vector2);
    };
    long durationSIMDSub = measurePerformance(simdSub, 1000000);
    std::cout << "SIMD subtraction took " << durationSIMDSub << " ms." << std::endl;

    // Multiplication operation
    std::cout << "----------- Multiplication ------------" << std::endl;
    result = _mm256_mul_ps(vector1, vector2);
    displayResult("Multiplication Result", (float*)&result, 8);

    // Measure performance of regular and SIMD multiplication
    auto regularMul = [&] {
        float data3[8];
        for (int lane = 0; lane < 8; ++lane) {
            data3[lane] = data1[lane] * data2[lane];
        }
    };
    long durationRegMul = measurePerformance(regularMul, 1000000);
    std::cout << "Regular multiplication took " << durationRegMul << " ms." << std::endl;

    auto simdMul = [&] {
        __m256 temp = _mm256_mul_ps(vector1, vector2);
    };
    long durationSIMDMul = measurePerformance(simdMul, 1000000);
    std::cout << "SIMD multiplication took " << durationSIMDMul << " ms." << std::endl;

    // Division operation
    std::cout << "----------- Division ------------" << std::endl;
    result = _mm256_div_ps(vector1, vector2);
    displayResult("Division Result", (float*)&result, 8);

    // Measure performance of regular and SIMD division
    auto regularDiv = [&] {
        float data3[8];
        for (int lane = 0; lane < 8; ++lane) {
            data3[lane] = data1[lane] / data2[lane];
        }
    };
    long durationRegDiv = measurePerformance(regularDiv, 1000000);
    std::cout << "Regular division took " << durationRegDiv << " ms." << std::endl;

    auto simdDiv = [&] {
        __m256 temp = _mm256_div_ps(vector1, vector2);
    };
    long durationSIMDDiv = measurePerformance(simdDiv, 1000000);
    std::cout << "SIMD division took " << durationSIMDDiv << " ms." << std::endl;

    // Fused multiply and add operation
    std::cout << "----------- Fused Multiply and Add ------------" << std::endl;
    __m256 vector3 = _mm256_setr_ps(-1, 2, -3, 4, -5, 6, -7, 8);
    result = _mm256_fmadd_ps(vector3, vector1, vector2); // vector3*vector1 + vector2
    displayResult("Fused Multiply-Add Result", (float*)&result, 8);

    // Measure performance of fused operations
    auto fusedOps = [&] {
        __m256 temp = _mm256_fmadd_ps(vector3, vector1, vector2);
    };
    long durationFusedOps = measurePerformance(fusedOps, 1000000);
    std::cout << "Fused operations took " << durationFusedOps << " ms." << std::endl;

    return 0;
}

// Function to display SIMD operation results
void displayResult(const char* operation, const float* SIMDdata, int size) {
    std::cout << operation << ": ";
    for (int lane = 0; lane < size; ++lane) {
        std::cout << SIMDdata[lane] << ", ";
    }
    std::cout << std::endl;
}

// Function to measure performance of a given function
template<typename Func>
long measurePerformance(Func f, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        f();
    }
    auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
}
#include "immintrin.h" // Include for AVX2, 256-bit operations
#include <chrono>
#include <iostream>

void printArray(const float* array, int size, const std::string& description) {
    std::cout << description << ": ";
    for (int i = 0; i < size; ++i) {
        std::cout << array[i] << ", ";
    }
    std::cout << std::endl;
}

void printArray(const double* array, int size, const std::string& description) {
    std::cout << description << ": ";
    for (int i = 0; i < size; ++i) {
        std::cout << array[i] << ", ";
    }
    std::cout << std::endl;
}

int main() {
    // Interfacing with SIMD Data using Pointer Conversion
    __m256 mySIMDArray = _mm256_setzero_ps();
    float* data = reinterpret_cast<float*>(&mySIMDArray);

    printArray(data, 8, "SIMD Array with Pointer Conversion");

    // Interfacing with SIMD Data using Union
    union {
        __m256d mySIMDArray2; // SIMD array for 4 double precision floats
        double data2[4];      // Regular double array
    };

    mySIMDArray2 = _mm256_setr_pd(3.5, -7.0, 2.1, 5.6);
    printArray(data2, 4, "SIMD Array with Union");

    // Modifying data using the regular array interface
    std::cout << "Modifying data..." << std::endl;
    for (int i = 0; i < 2; ++i) {
        data2[i] = i;
    }

    printArray(data2, 4, "Modified SIMD Array with Union");

    return 0;
}

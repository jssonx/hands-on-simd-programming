#include "immintrin.h"
#include <chrono>
#include <iostream>

// Constants
constexpr int NUM_ITERATIONS = 1000000;

template <typename T, size_t N>
void printArray(const T (&arr)[N], const std::string &description) {
    std::cout << description << ": ";
    for (size_t i = 0; i < N; ++i) {
        std::cout << arr[i] << ", ";
    }
    std::cout << std::endl;
}

void copyFromSIMD(float* dest, const __m256& src) {
    _mm256_storeu_ps(dest, src);
}

void copyFromSIMD(double* dest, const __m256d& src) {
    _mm256_storeu_pd(dest, src);
}

void copyFromSIMD(int* dest, const __m256i& src) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dest), src);
}

void copyFromSIMD(short* dest, const __m256i& src) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dest), src);
}

int main()
{
	// --------- setzero -------------
	std::cout << "--------- setzero -------------" << std::endl;

	// Standard method
	float myArray[8] = {0.0f};
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < NUM_ITERATIONS; ++i)
	{
		for (int lane = 0; lane < 8; ++lane)
		{
			myArray[lane] = 0.0f;
		}
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto durationStandard = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "Standard allocation of 0.0f took " << durationStandard.count() << " ms." << std::endl;

	// SIMD method
	__m256 mySIMDArray;
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < NUM_ITERATIONS; ++i)
	{
		mySIMDArray = _mm256_setzero_ps();
	}
	stop = std::chrono::high_resolution_clock::now();
	auto durationSIMD = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "SIMD allocation of 0.0f took " << durationSIMD.count() << " ms." << std::endl;

	printArray(myArray, "myArray");
	float myArraySIMD[8];
	copyFromSIMD(myArraySIMD, mySIMDArray);
	printArray(myArraySIMD, "myArraySIMD");

	// --------- set1 -------------
	std::cout << "--------- set1 -------------" << std::endl;

	// Standard method
	double myArray2[4] = {10.0};
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < NUM_ITERATIONS; ++i)
	{
		for (int lane = 0; lane < 4; ++lane)
		{
			myArray2[lane] = 10.0;
		}
	}
	stop = std::chrono::high_resolution_clock::now();
	durationStandard = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "Standard allocation of 10.0 took " << durationStandard.count() << " ms." << std::endl;

	// SIMD method
	__m256d mySIMDArray2;
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < NUM_ITERATIONS; ++i)
	{
		mySIMDArray2 = _mm256_set1_pd(10.0);
	}
	stop = std::chrono::high_resolution_clock::now();
	durationSIMD = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "SIMD allocation of 10.0 took " << durationSIMD.count() << " ms." << std::endl;

	printArray(myArray2, "myArray2");
	double myArraySIMD2[4];
	copyFromSIMD(myArraySIMD2, mySIMDArray2);
	printArray(myArraySIMD2, "myArraySIMD2");

	// --------- general set -------------
	std::cout << "--------- general set -------------" << std::endl;

	// Standard method
	int myArray3[8] = {1, 2, 3, 4, 5, 6, 7, 8};
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < NUM_ITERATIONS; ++i)
	{
		for (int lane = 0; lane < 8; ++lane)
		{
			myArray3[lane] = lane + 1;
		}
	}
	stop = std::chrono::high_resolution_clock::now();
	durationStandard = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "Standard allocation of integers took " << durationStandard.count() << " ms." << std::endl;

	// SIMD method
	__m256i mySIMDArray3;
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < NUM_ITERATIONS; ++i)
	{
		mySIMDArray3 = _mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1);
	}
	stop = std::chrono::high_resolution_clock::now();
	durationSIMD = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "SIMD allocation of integers took " << durationSIMD.count() << " ms." << std::endl;

	printArray(myArray3, "myArray3");
	int myArraySIMD3[8];
	copyFromSIMD(myArraySIMD3, mySIMDArray3);
	printArray(myArraySIMD3, "myArraySIMD3");

	// --------- setr -------------
	std::cout << "--------- setr -------------" << std::endl;
	// Standard method
	short myArray4[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < NUM_ITERATIONS; ++i)
	{
		for (int lane = 0; lane < 16; ++lane)
		{
			myArray4[lane] = static_cast<short>(lane + 1);
		}
	}
	stop = std::chrono::high_resolution_clock::now();
	durationStandard = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "Standard allocation of shorts took " << durationStandard.count() << " ms." << std::endl;

	// SIMD method
	__m256i mySIMDArray4;
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < NUM_ITERATIONS; ++i)
	{
		mySIMDArray4 = _mm256_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
	}
	stop = std::chrono::high_resolution_clock::now();
	durationSIMD = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "SIMD allocation of shorts took " << durationSIMD.count() << " ms." << std::endl;

	printArray(myArray4, "myArray4");
	short myArraySIMD4[16];
	copyFromSIMD(myArraySIMD4, mySIMDArray4);
	printArray(myArraySIMD4, "myArraySIMD4");

	return 0;
}


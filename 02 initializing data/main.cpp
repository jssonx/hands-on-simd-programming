#include "immintrin.h" //AVX2, 256 bit operations (8 floats)
#include <chrono>
#include <iostream>


/*
 * Compile and run with:
 * g++ -mavx2 -mfma main.cpp -o main
 * ./main
*/


int main()
{

	// __m256 : a chunk of 256 bits, equivalent to an array of 8 floats
	// __m256d : a chunk of 256 bits, equivalent to an array of 4 doubles
	// __m256i : a chunk of 256 bits, equivalent to an array of 8 ints

	// 256 bits = 32 bytes = 8 floats = 4 doubles = 8 ints

	//--------- setzero -------------//
	std::cout << "--------- setzero -------------" << std::endl;
	// Standard array initialization.
	float myArray[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	std::cout << "My Array: ";
	for (int lane = 0; lane < 8; ++lane)
	{
		std::cout << myArray[lane] << ", ";
	}
	std::cout << std::endl;

	// SIMD method: _mm256_setzero_ps()
	/*
	 * SIMD stands for Single Instruction Multiple Data.
	 * _mm256_setzero_ps() is a function that returns a __m256 type.
	 * _mm256_set_zero_ps stands for "set packed single precision floats to zero".
	 */
	__m256 mySIMDArray = _mm256_setzero_ps();
	float *data = (float *)&mySIMDArray; // get the address of the array, then treat it
										 // like the address of a float.
	std::cout << "My SIMD Array: ";
	for (int lane = 0; lane < 8; ++lane)
	{
		std::cout << data[lane] << ", ";
	}
	std::cout << std::endl;

	// Let's test performance.
	//
	// standard
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i)
	{
		for (int lane = 0; lane < 8; ++lane)
		{
			myArray[lane] = 0.0f;
		}
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "standard allocation of 0.0f took " << duration.count() << " ms." << std::endl;
	//
	// SIMD
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i)
	{
		mySIMDArray = _mm256_setzero_ps();
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "simd allocation of 0.0f took " << duration.count() << " ms." << std::endl;

	//--------- set1 -------------//
	std::cout << "--------- set1 ----------------" << std::endl;
	// Standard array initialization.
	double myArray2[4] = {10.0, 10.0, 10.0, 10.0}; // 4 doubles = 32 bytes = 256 bits
	std::cout << "My Array: ";
	for (int lane = 0; lane < 4; ++lane)
	{
		std::cout << myArray2[lane] << ", ";
	}
	std::cout << std::endl;

	// SIMD method: _mm256_set1_pd(double value)
	/*
	 * set packed double precision floats to a uniform value.
	 */
	__m256d mySIMDArray2 = _mm256_set1_pd(10.0);
	double *data2 = (double *)&mySIMDArray2;
	std::cout << "My SIMD Array: ";
	for (int lane = 0; lane < 4; ++lane)
	{
		std::cout << data2[lane] << ", ";
	}
	std::cout << std::endl;

	// Let's test performance.
	//
	// standard
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i)
	{
		for (int lane = 0; lane < 4; ++lane)
		{
			myArray2[lane] = 10.0;
		}
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "standard allocation of 10.0 took " << duration.count() << " ms." << std::endl;
	//
	// SIMD
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i)
	{
		mySIMDArray2 = _mm256_set1_pd(10.0);
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "simd allocation of 10.0 took " << duration.count() << " ms." << std::endl;

	//--------- general set -------------//
	std::cout << "--------- general set ---------" << std::endl;
	// Standard array initialization.
	double myArray3[8] = {1, 2, 3, 4, 5, 6, 7, 8};
	std::cout << "My Array: ";
	for (int lane = 0; lane < 8; ++lane)
	{
		std::cout << myArray3[lane] << ", ";
	}
	std::cout << std::endl;

	// SIMD method: _mm256_set_epi32(int val1, ...)
	/*
	 * set a vector's contents to store the eight provided ints
	 */
	__m256i mySIMDArray3 = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8); // 8 ints = 32 bytes = 256 bits
	int *data3 = (int *)&mySIMDArray3;								 // like the address of an int
	std::cout << "My SIMD Array: ";
	for (int lane = 0; lane < 8; ++lane)
	{
		std::cout << data3[lane] << ", ";
	}
	std::cout << std::endl;

	// Let's test performance.
	//
	// standard
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i)
	{
		for (int lane = 0; lane < 8; ++lane)
		{
			myArray3[lane] = lane; // load in reverse order
		}
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "standard allocation of integers took " << duration.count() << " ms." << std::endl;

	// SIMD
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i)
	{
		mySIMDArray3 = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "simd allocation of integers took " << duration.count() << " ms." << std::endl;

	//--------- setr -------------//
	std::cout << "--------- setr ----------------" << std::endl;
	// Standard array initialization.
	double myArray4[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
	std::cout << "My Array: ";
	for (int lane = 0; lane < 16; ++lane)
	{
		std::cout << myArray4[lane] << ", ";
	}
	std::cout << std::endl;

	// SIMD method: _mm256_setr_epi16(int val1, ...)
	/*
	 * set a vector's contents to store the 16 provided ints,
	 * loading numbers in reverse order
	 */
	__m256i mySIMDArray4 = _mm256_setr_epi16(
		1, 2, 3, 4, 5, 6, 7, 8,
		9, 10, 11, 12, 13, 14, 15, 16);
	short *data4 = (short *)&mySIMDArray4; // like the address of a short.
	std::cout << "My SIMD Array: ";
	for (int lane = 0; lane < 16; ++lane)
	{
		std::cout << data4[lane] << ", ";
	}
	std::cout << std::endl;

	// Let's test performance.
	//
	// standard
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i)
	{
		for (int lane = 0; lane < 16; ++lane)
		{
			myArray4[lane] = lane;
		}
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "standard allocation of integers took " << duration.count() << " ms." << std::endl;

	// SIMD
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i)
	{
		mySIMDArray4 = _mm256_setr_epi16(
			1, 2, 3, 4, 5, 6, 7, 8,
			9, 10, 11, 12, 13, 14, 15, 16);
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "simd allocation of integers took " << duration.count() << " ms." << std::endl;

	return 0;
}
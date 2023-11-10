#include "immintrin.h" // Include for AVX2, 256-bit operations
#include <chrono>
#include <iostream>

/*
 * Use Case: Interfacing with SIMD Data.
 * We have some SIMD data structures, but at some point
 * we need to access the underlying data.
 * With Union, we can access the data as a regular array.
 * 
 * Compile and run with:
 * g++ -mavx2 -mfma main.cpp -o main
 * ./main
*/

int main()
{

	//--------- Interfacing with SIMD Data using Pointer Conversion -------------//
	// Initialize a __m256 type SIMD array with all values set to 0.0f
	__m256 mySIMDArray = _mm256_setzero_ps();

	// Convert the SIMD array to a pointer to access its elements like a regular float array
	float *data = reinterpret_cast<float *>(&mySIMDArray);

	// Iterate and print each element in the SIMD array
	std::cout << "My SIMD Array: ";
	for (int lane = 0; lane < 8; ++lane)
	{
		std::cout << data[lane] << ", ";
	}
	std::cout << std::endl;

	//--------- Interfacing with SIMD Data using Union -------------//
	// Define a union to access SIMD data as a regular array
	union
	{
		__m256d mySIMDArray2; // SIMD array for 4 double precision floats
		double data2[4];	  // Regular double array
	};

	// Initialize the SIMD array with specific values in reverse order
	mySIMDArray2 = _mm256_setr_pd(3.5, -7.0, 2.0, 5.6);

	// Iterate and print each element in the SIMD array
	std::cout << "My SIMD Array: ";
	for (int lane = 0; lane < 4; ++lane)
	{
		std::cout << data2[lane] << ", ";
	}
	std::cout << std::endl;

	// Modify the data using the regular array interface
	std::cout << "Modifying data..." << std::endl;
	for (int lane = 0; lane < 4; ++lane)
	{
		data2[lane] = lane;
	}

	// Print the modified SIMD array
	std::cout << "My SIMD Array: ";
	for (int lane = 0; lane < 4; ++lane)
	{
		std::cout << data2[lane] << ", ";
	}
	std::cout << std::endl;

	// Note: While both pointer conversion and union are valid for interfacing SIMD data,
	// they may not fully exploit the performance benefits of SIMD operations.

	return 0;
}

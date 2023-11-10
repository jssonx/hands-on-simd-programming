#include "immintrin.h" // Include for AVX2, 256-bit operations
#include <chrono>
#include <iostream>

/*
 * Use case: Loading data into SIMD registers.
 * Demonstrates different methods for loading data into SIMD registers.
 * Comparing the performance of setr, load, and loadu functions.
*/

int main() {

    //-------- Aligned Load ---------------//
    // Allocate memory aligned to 32 bytes for AVX compatibility
    float* data = (float*)aligned_alloc(32, 8 * sizeof(float));
	// data = (float*)malloc(8 * sizeof(float));
	// Segmentation fault (core dumped) if using unaligned memory when using _mm256_load_ps

    // Initialize the array with values 0 to 7
    for (int lane = 0; lane < 8; ++lane) {
        data[lane] = lane;
    }

    // Load aligned data into the SIMD register
    __m256 mySIMDArray = _mm256_load_ps(data);
    free(data);  // Free the aligned memory

    // Display the loaded SIMD data
    float* SIMDdata = (float*)&mySIMDArray;
    std::cout << "My SIMD Array (aligned load): ";
    for (int lane = 0; lane < 8; ++lane) {
        std::cout << SIMDdata[lane] << ", ";
    }
    std::cout << std::endl;

    //-------- Unaligned Load ---------------//
    // Allocate unaligned memory
    data = (float*)malloc(8 * sizeof(float));

    // Initialize the array with values 0 to 7
    for (int lane = 0; lane < 8; ++lane) {
        data[lane] = lane;
    }

    // Load unaligned data into the SIMD register
    mySIMDArray = _mm256_loadu_ps(data); 
    free(data);  // Free the unaligned memory

    // Display the loaded SIMD data
    SIMDdata = (float*)&mySIMDArray;
    std::cout << "My SIMD Array (unaligned load): ";
    for (int lane = 0; lane < 8; ++lane) {
        std::cout << SIMDdata[lane] << ", ";
    }
    std::cout << std::endl;

    //-------- Performance Testing ---------------//
    // Allocate aligned memory for performance testing
    data = (float*)aligned_alloc(32, 8 * sizeof(float));

    // Initialize the array with values 0 to 7 for testing
    for (int lane = 0; lane < 8; ++lane) {
        data[lane] = lane;
    }

    // Performance test for setr function
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000000000; ++i) {
        mySIMDArray = _mm256_setr_ps(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Performance of setr took " << duration.count() << " ms." << std::endl;

    // Performance test for load function
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000000000; ++i) {
        mySIMDArray = _mm256_load_ps(data);
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Performance of aligned load took " << duration.count() << " ms." << std::endl;

    // Performance test for loadu function
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000000000; ++i) {
        mySIMDArray = _mm256_loadu_ps(data);
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Performance of unaligned load took " << duration.count() << " ms." << std::endl;

    free(data);  // Free the aligned memory used for performance testing

    return 0;
}


// #include "immintrin.h" //AVX2, 256 bit operations (8 floats)
// #include <chrono>
// #include <iostream>

// /*
//  * Use Case: Different ways to load data into SIMD registers.
// */

// int main() {

// 	//--------- loading data -------------//
// 	/*
// 	* Before we were using set and setr to get data into vectors on the fly,
// 	* however there are better ways
// 	*/

// 	//-------- load ---------------//
// 	// float* data = (float*)malloc(8 * sizeof(float));
// 	float* data = (float*)aligned_alloc(32, 8 * sizeof(float));

// 	for (int lane = 0; lane < 8; ++lane) {
// 		data[lane] = lane;
// 	}
// 	__m256 mySIMDArray = _mm256_load_ps(data);
// 	free(data);
// 	float* SIMDdata = (float*)&mySIMDArray;
// 	std::cout << "My SIMD Array: ";
// 	for (int lane = 0; lane < 8; ++lane) {
// 		std::cout << SIMDdata[lane] << ", ";
// 	}
// 	std::cout << std::endl;

// 	//-------- load unaligned ---------------//
// 	/*
// 	* I fudged the last example a little, the load function
// 	* should actually take data which is properly alligned to memory
// 	* (otherwise results may be unpredictable.)
// 	* 
// 	* loadu, on the other hand, will handle a regular malloc-ed block just fine.
// 	*/
// 	data = (float*)malloc(8 * sizeof(float));
// 	for (int lane = 0; lane < 8; ++lane) {
// 		data[lane] = lane;
// 	}
// 	mySIMDArray = _mm256_loadu_ps(data); // load unaligned data into SIMD register
// 	free(data);
// 	SIMDdata = (float*)&mySIMDArray;
// 	std::cout << "My SIMD Array: ";
// 	for (int lane = 0; lane < 8; ++lane) {
// 		std::cout << SIMDdata[lane] << ", ";
// 	}
// 	std::cout << std::endl;

// 	//Let's test performance
// 	//setr
// 	// data = (float*)malloc(8 * sizeof(float));
// 	data = (float*)aligned_alloc(32, 8 * sizeof(float));

// 	for (int lane = 0; lane < 8; ++lane) {
// 		data[lane] = lane;
// 	}
// 	auto start = std::chrono::high_resolution_clock::now();
// 	for (int i = 0; i < 1000000000; ++i) {
// 		mySIMDArray = _mm256_setr_ps(
// 			data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]
// 		);
// 	}
// 	auto stop = std::chrono::high_resolution_clock::now();
// 	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
// 	std::cout << "load via setr took " << duration.count() << " ms." << std::endl;
// 	//load
// 	start = std::chrono::high_resolution_clock::now();
// 	for (int i = 0; i < 1000000000; ++i) {
// 		mySIMDArray = _mm256_load_ps(data);
// 	}
// 	stop = std::chrono::high_resolution_clock::now();
// 	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
// 	std::cout << "load function took " << duration.count() << " ms." << std::endl;
// 	//loadu
// 	start = std::chrono::high_resolution_clock::now();
// 	for (int i = 0; i < 1000000000; ++i) {
// 		mySIMDArray = _mm256_loadu_ps(data);
// 	}
// 	stop = std::chrono::high_resolution_clock::now();
// 	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
// 	std::cout << "loadu function took " << duration.count() << " ms." << std::endl;
// 	free(data);
// 	return 0;
// }
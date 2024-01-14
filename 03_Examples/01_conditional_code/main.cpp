#include "immintrin.h" //AVX2, 256 bit operations (8 floats)
#include <chrono>
#include <iostream>

/*
 * Key Components:
 * 1. Clamping: Compares SIMD and scalar clamping methods for data2 within 5.0 to 30.0.
 * 2. Positive Extraction: Uses _mm256_cmp_ps for SIMD-based filtering of positive values in data2.
 * 3. Comparative Analysis: Employs SIMD for complex conditional logic, comparing data2 against data1 to
 * identify elements in data2 that are both positive and greater than their counterparts in data1.
 *
 * Focus:
 * - Showcases SIMD's efficiency in conditional operations for large data sets.
 * - Illustrates use of SIMD masks for selective data manipulation.
 */

int main() {

	//--------- Simple maths -------------//
	float* data1 = (float*)malloc(8 * sizeof(float));
	data1[0] = 5;
	data1[1] = 10;
	data1[2] = 15;
	data1[3] = 20;
	data1[4] = 25;
	data1[5] = 30;
	data1[6] = 35;
	data1[7] = 40;
	float* data2 = (float*)malloc(8 * sizeof(float));
	data2[0] = -1;
	data2[1] = 4;
	data2[2] = 9;
	data2[3] = -16;
	data2[4] = 25;
	data2[5] = -36;
	data2[6] = 49;
	data2[7] = -64;
	union { __m256 vector2; float vector_2[8]; };
	__m256 vector1 = _mm256_loadu_ps(data1);
	vector2 = _mm256_loadu_ps(data2);

	float* data3 = (float*)malloc(8 * sizeof(float));
	__m256 result;
	float* SIMDdata;

	//-------- clamping ---------------//
	std::cout << "----------- clamping ---------- ------------ ------ -------------------" << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		for (int lane = 0; lane < 8; ++lane) {
			data3[lane] = std::max(5.0f, std::min(30.0f,data2[lane]));
		}
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "regular clamp took " << duration.count() << " ms." << std::endl;

	// addition, SIMD
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		result = _mm256_max_ps(_mm256_set1_ps(5), _mm256_min_ps(_mm256_set1_ps(30), vector2));
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "SIMD addition took " << duration.count() << " ms." << std::endl;

	//-------- get positive numbers ---------------//
	std::cout << "----------- get positive numbers ----- ------------ -------------------" << std::endl;
	result = _mm256_cmp_ps(vector2, _mm256_setzero_ps(), _CMP_GT_OQ);
	SIMDdata = (float*)&result;
	std::cout << "My comparison Result: ";
	for (int lane = 0; lane < 8; ++lane) {
		std::cout << SIMDdata[lane] << ", ";
	}
	std::cout << std::endl;
	
	/*
	* Positions 1,2,4 and 6 have positive numbers,
	* as a bitmask this would be 2 + 4 + 16 + 64 = 86
	*/
	int mask = _mm256_movemask_ps(result);
	std::cout << "Bitmask: ";
	std::cout << mask;
	std::cout << std::endl;

	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		for (int lane = 0; lane < 8; ++lane) {
			if (data2[lane] > 0) {
				data3[0] = data2[lane];
			}
		}
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "regular conditional test took " << duration.count() << " ms." << std::endl;
	
	// SIMD
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		result = _mm256_cmp_ps(vector2, _mm256_setzero_ps(), _CMP_GT_OQ);
		mask = _mm256_movemask_ps(result);
		for (int lane = 0; lane < 8; ++lane) {
			if (mask & (1 << lane)) {
				data3[0] = vector_2[lane];
			}
		}
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "SIMD conditional test took " << duration.count() << " ms." << std::endl;

	//-------- get positive numbers in vector2 greater than vector1 ---------------//
	std::cout << "----------- get positive numbers in vector2 greater than vector1 ------" << std::endl;
	__m256 posi = _mm256_cmp_ps(vector2, _mm256_setzero_ps(), _CMP_GT_OQ);
	__m256 big = _mm256_cmp_ps(vector2, vector1, _CMP_GT_OQ);
	result = _mm256_and_ps(posi, big);
	SIMDdata = (float*)&result;
	std::cout << "My comparison Result: ";
	for (int lane = 0; lane < 8; ++lane) {
		std::cout << SIMDdata[lane] << ", ";
	}
	std::cout << std::endl;
	
	/*
	* This mask is zero everywhere the tests failed.
	* Let's write a number to index 1 of data3 if it passes,
	* and index 0 if it fails (0 is the garbage can)
	*/
	union {__m256i address; int index[8];};
	address = _mm256_and_si256(_mm256_set1_epi32(1), *(__m256i*) & result);
	std::cout << "Addresses: ";
	for (int lane = 0; lane < 8; ++lane) {
		std::cout << index[lane];
	}
	std::cout << std::endl;

	result = _mm256_blendv_ps(_mm256_setzero_ps(), vector2, result);
	std::cout << "My comparison Result: ";
	for (int lane = 0; lane < 8; ++lane) {
		std::cout << SIMDdata[lane] << ", ";
	}
	std::cout << std::endl;

	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		for (int lane = 0; lane < 8; ++lane) {
			if (data2[lane] > 0 && data2[lane] > data1[lane]) {
				data3[1] = data2[lane];
			}
			else {
				data3[0] = data2[lane];
			}
		}
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "regular conditional test took " << duration.count() << " ms." << std::endl;
	
	// SIMD
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		posi = _mm256_cmp_ps(vector2, _mm256_setzero_ps(), _CMP_GT_OQ);
		big = _mm256_cmp_ps(vector2, vector1, _CMP_GT_OQ);
		result = _mm256_and_ps(posi, big);
		address = _mm256_and_si256(_mm256_set1_epi32(1), *(__m256i*) & result);
		for (int lane = 0; lane < 8; ++lane) {
			data3[index[lane]] = vector_2[lane];
		}
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "SIMD conditional test took " << duration.count() << " ms." << std::endl;
	
	//SIMD take two
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		posi = _mm256_cmp_ps(vector2, _mm256_setzero_ps(), _CMP_GT_OQ);
		big = _mm256_cmp_ps(vector2, vector1, _CMP_GT_OQ);
		result = _mm256_and_ps(posi, big);
		result = _mm256_blendv_ps(_mm256_setzero_ps(), vector2, result);
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "SIMD conditional test took " << duration.count() << " ms." << std::endl;

	free(data1);
	free(data2);
	free(data3);

	return 0;
}
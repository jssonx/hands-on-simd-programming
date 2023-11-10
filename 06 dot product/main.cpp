#include "immintrin.h" //AVX2, 256 bit operations (8 floats)
#include <chrono>
#include <iostream>
#include <glm/glm.hpp> // dependency for glm::vec3
#include <array>

/*
 * This program demonstrates the use of AVX2 SIMD instructions for efficiently computing dot products.
 * The SIMD approach utilizes the 256-bit wide AVX2 registers to process multiple data (8 floats) in parallel,
 * significantly speeding up the computation compared to the naive approach.
 *
 * The process involves:
 * 1. Unpacking the components (x, y, z) of glm::vec3 vectors into separate __m256 variables.
 * 2. Performing SIMD operations using _mm256_fmadd_ps to compute the dot products in parallel.
 * 3. Comparing the performance of SIMD implementation against the naive approach.
 *
 * SIMD Operations for Dot Product:
 * - The dot product formula is: Dot(A, B) = A_x * B_x + A_y * B_y + A_z * B_z.
 * - The SIMD implementation uses _mm256_fmadd_ps, which performs fused multiply-add operations.
 * - The calculation SIMDresult = _mm256_fmadd_ps(x1, x2, _mm256_fmadd_ps(y1, y2, _mm256_mul_ps(z1, z2)))
 *   efficiently computes the dot product for all vector pairs simultaneously.
 *   It performs the operation: x1*x2 + (y1*y2 + (z1*z2)).
 *
 * Compile and run with:
 * g++ -mavx2 -mfma main.cpp -o main
 * ./main
 *
 */

int main() {

	//--------- Dot Products! -------------//
	std::array<glm::vec3, 8> vectors1 = { {
			glm::vec3(1.0f, 0.2f, 0.3f),
			glm::vec3(0.1f, -0.4f, -0.6f),
			glm::vec3(1.0f, 1.0f, 0.9f),
			glm::vec3(-0.2f, 1.0f, -1.2f),
			glm::vec3(1.0f, -0.8f, 1.0f),
			glm::vec3(-0.3f, 1.6f, 1.0f),
			glm::vec3(1.0f, 1.0f, 1.0f),
			glm::vec3(-0.4f, -3.2f, 1.0f),
	} };
	std::array<glm::vec3, 8> vectors2 = { {
			glm::vec3(-1.0f, -0.6f, 0.7f),
			glm::vec3(-0.4f, 1.2f, -1.4f),
			glm::vec3(-1.0f, 1.0f, 2.1f),
			glm::vec3(0.8f, 1.0f, -2.8f),
			glm::vec3(-1.0f, -1.6f, 1.0f),
			glm::vec3(1.2f, 1.6f, 1.0f),
			glm::vec3(-1.0f, 1.0f, 1.0f),
			glm::vec3(1.6f, -2.4f, 1.0f),
	} };
	float result;

	//-------- naive approach ---------------//
	std::cout << "-------- naive approach ---------------" << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		for (int lane = 0; lane < 8; ++lane) {
			result = glm::dot(vectors1[lane], vectors2[lane]);
		}
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "glm dot product took " << duration.count() << " ms." << std::endl;

	//-------- simd approach ---------------//
	std::cout << "-------- simd approach ---------------" << std::endl;
	//declare simd memory containers
	union { __m256 x1; float x_1[8]; };
	union { __m256 x2; float x_2[8]; };
	union { __m256 y1; float y_1[8]; };
	union { __m256 y2; float y_2[8]; };
	union { __m256 z1; float z_1[8]; };
	union { __m256 z2; float z_2[8]; };
	__m256 SIMDresult;

	//unpack data from vectors, pack into simd lanes
	for (int lane = 0; lane < 8; ++lane) {
		x_1[lane] = vectors1[lane].x;
		x_2[lane] = vectors2[lane].x;
		y_1[lane] = vectors1[lane].y;
		y_2[lane] = vectors2[lane].y;
		z_1[lane] = vectors1[lane].z;
		z_2[lane] = vectors2[lane].z;
	}

	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		//compute dot product
		// x1*x2 + y1*y2 + z1*z2
		// = (x1 * x2 + [y1 * y2 + <z1 * z2>]), 2 fmas, 1 mul
		SIMDresult = _mm256_fmadd_ps(x1, x2, _mm256_fmadd_ps(y1, y2, _mm256_mul_ps(z1, z2)));
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "SIMD dot product took " << duration.count() << " ms." << std::endl;

	return 0;
}
#include "immintrin.h" //AVX2, 256 bit operations (8 floats)
#include <chrono>
#include <iostream>
#include <limits>
#include <math.h>


int main() {

	//--------- Project: Quadratic Equations -------------//
	float* a = (float*)malloc(8 * sizeof(float)); // Coefficients of x^2
	a[0] = 5;
	a[1] = 12;
	a[2] = 6;
	a[3] = 7;
	a[4] = 1;
	a[5] = 1;
	a[6] = 1;
	a[7] = 1;
	float* b = (float*)malloc(8 * sizeof(float)); // Coefficients of x
	b[0] = 3;
	b[1] = 1;
	b[2] = 4;
	b[3] = -2;
	b[4] = 2;
	b[5] = 1;
	b[6] = 1;
	b[7] = 1;
	float* c = (float*)malloc(8 * sizeof(float)); // Constant terms
	c[0] = -1;
	c[1] = -5;
	c[2] = -6;
	c[3] = -6;
	c[4] = 5;
	c[5] = 30;
	c[6] = 35;
	c[7] = -40;
	__m256 aCoeffs, bCoeffs, cCoeffs; // Coefficients of x^2, x, and constant terms

	union { __m256 simdResult; float numericResult[8]; };

	//-------- standard approach ---------------//
	std::cout << "----------- standard approach " << std::endl;

	for (int lane = 0; lane < 8; ++lane) {
		numericResult[lane] = 9999;
	}
	
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		for (int lane = 0; lane < 8; ++lane) {
			float discriminant = b[lane] * b[lane] - 4.0f * a[lane] * c[lane];
			if (discriminant > 0) {
				numericResult[lane] = (-b[lane] - sqrtf(discriminant)) / (2.0 * a[lane]);
			}
		}
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "solution took " << duration.count() << " ms." << std::endl;

	std::cout << "Solutions: ";
	for (int lane = 0; lane < 8; ++lane) {
		std::cout << numericResult[lane] << ", ";
	}
	std::cout << std::endl;

	//-------- simd ---------------//
	std::cout << "----------- simd approach " << std::endl;
	
	simdResult = _mm256_set1_ps(9999);
	aCoeffs = _mm256_loadu_ps(a);
	bCoeffs = _mm256_loadu_ps(b);
	cCoeffs = _mm256_loadu_ps(c);
	
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000000; ++i) {
		__m256 disc = _mm256_fmsub_ps(bCoeffs, bCoeffs,
			_mm256_mul_ps(_mm256_set1_ps(4), _mm256_mul_ps(aCoeffs, cCoeffs))
		);
		__m256 mask = _mm256_cmp_ps(disc, _mm256_setzero_ps(), _CMP_GT_OQ);
		simdResult = _mm256_div_ps(
			_mm256_sub_ps(
				_mm256_sub_ps(_mm256_setzero_ps(), bCoeffs),
				_mm256_sqrt_ps(disc)
			),
			_mm256_mul_ps(_mm256_set1_ps(2), aCoeffs)
		);
		simdResult = _mm256_blendv_ps(_mm256_set1_ps(9999), simdResult, mask);
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "SIMD solution took " << duration.count() << " ms." << std::endl;

	std::cout << "Solutions: ";
	for (int lane = 0; lane < 8; ++lane) {
		std::cout << numericResult[lane] << ", ";
	}
	std::cout << std::endl;

	free(a);
	free(b);
	free(c);

	return 0;
}
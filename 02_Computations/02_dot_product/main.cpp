#include "immintrin.h" // AVX2, 256 bit operations
#include <chrono>
#include <iostream>
#include <array>

// 3D vector structure
struct Vec3 {
    float x, y, z;
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
};

// Function declarations to perform dot product calculations
void naiveDotProduct(const std::array<Vec3, 8>& vectors1, const std::array<Vec3, 8>& vectors2);
void simdDotProduct(const std::array<Vec3, 8>& vectors1, const std::array<Vec3, 8>& vectors2);

int main() {
    // Initialize vectors
	std::array<Vec3, 8> vectors1 = {
        Vec3(1.0f, 0.5f, -0.2f),
        Vec3(-0.1f, 0.8f, 0.6f),
        Vec3(0.3f, -0.5f, 0.9f),
        Vec3(0.4f, 0.4f, -0.3f),
        Vec3(-0.2f, -0.9f, 0.1f),
        Vec3(0.6f, 0.2f, -0.8f),
        Vec3(-0.5f, -0.4f, 0.7f),
        Vec3(0.1f, -0.1f, 0.5f)
    };

	std::array<Vec3, 8> vectors2 = {
        Vec3(-0.6f, 0.3f, 0.8f),
        Vec3(0.9f, -0.5f, 0.2f),
        Vec3(-0.2f, 0.7f, -0.4f),
        Vec3(0.8f, -0.3f, 0.1f),
        Vec3(-0.4f, 0.6f, 0.9f),
        Vec3(0.2f, 0.0f, -0.7f),
        Vec3(-0.1f, 0.4f, 0.3f),
        Vec3(0.5f, -0.8f, -0.2f)
    };

	// Perform dot product calculations
    naiveDotProduct(vectors1, vectors2);
    simdDotProduct(vectors1, vectors2);

    return 0;
}

// Standard dot product calculation for 3D vectors
float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Naive approach to calculate dot products
void naiveDotProduct(const std::array<Vec3, 8>& vectors1, const std::array<Vec3, 8>& vectors2) {
    std::cout << "-------- Naive Approach ---------------" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    float result = 0.0f;
    for (int i = 0; i < 1000000; ++i) { // Repeat to simulate workload
        for (int lane = 0; lane < 8; ++lane) {
            result += dot(vectors1[lane], vectors2[lane]);
        }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Naive dot product took " << duration.count() << " ms." << std::endl;
}

// SIMD approach to calculate dot products
void simdDotProduct(const std::array<Vec3, 8>& vectors1, const std::array<Vec3, 8>& vectors2) {
    std::cout << "-------- SIMD Approach ---------------" << std::endl;

	// Unions to access vector data as float arrays
    union { __m256 x1; float x_1[8]; };
    union { __m256 x2; float x_2[8]; };
    union { __m256 y1; float y_1[8]; };
    union { __m256 y2; float y_2[8]; };
    union { __m256 z1; float z_1[8]; };
    union { __m256 z2; float z_2[8]; };

	// Load vector components into arrays for SIMD processing
    for (int lane = 0; lane < 8; ++lane) {
        x_1[lane] = vectors1[lane].x;
        y_1[lane] = vectors1[lane].y;
        z_1[lane] = vectors1[lane].z;
        x_2[lane] = vectors2[lane].x;
        y_2[lane] = vectors2[lane].y;
        z_2[lane] = vectors2[lane].z;
    }

	// Load data from arrays into SIMD registers
    x1 = _mm256_loadu_ps(x_1);
    y1 = _mm256_loadu_ps(y_1);
    z1 = _mm256_loadu_ps(z_1);
    x2 = _mm256_loadu_ps(x_2);
    y2 = _mm256_loadu_ps(y_2);
    z2 = _mm256_loadu_ps(z_2);

    auto start = std::chrono::high_resolution_clock::now();

    __m256 SIMDresult;
    for (int i = 0; i < 1000000; ++i) { // Repeat to simulate workload
        SIMDresult = _mm256_fmadd_ps(x1, x2, _mm256_fmadd_ps(y1, y2, _mm256_mul_ps(z1, z2)));
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "SIMD dot product took " << duration.count() << " ms." << std::endl;

    float dotProducts[8];
    _mm256_storeu_ps(dotProducts, SIMDresult);
    std::cout << "Dot products: ";
    for (float dotProduct : dotProducts) {
        std::cout << dotProduct << " ";
    }
    std::cout << std::endl;
}

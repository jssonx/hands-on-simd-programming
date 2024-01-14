// SIMD operations can be included in C/C++ programs via specific header files.
// Below is a hierarchy of headers provided by Intel, grouped by the instruction sets they implement.

#include "immintrin.h" // The all-encompassing header for Intel SIMD: AVX, AVX2, FMA, AVX-512, etc.

// Legacy SIMD headers for different instruction sets:
#include <mmintrin.h>    // MMX
#include <xmmintrin.h>   // SSE
#include <emmintrin.h>   // SSE2
#include <pmmintrin.h>   // SSE3
#include <tmmintrin.h>   // SSSE3
#include <smmintrin.h>   // SSE4.1
#include <nmmintrin.h>   // SSE4.2
#include <ammintrin.h>   // SSE4A
#include <wmmintrin.h>   // AES (Advanced Encryption Standard) new instructions

/*
 * Generally, "immintrin.h" is sufficient for most SIMD operations as it includes all the above.
 * Here is a simple example that uses AVX2 instructions to add two arrays of floats:
 */

#include <stdio.h>

int main() {
    // Example of using AVX2 instructions to add float arrays
    __m256 a = _mm256_set_ps(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
    __m256 b = _mm256_set_ps(8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f);
    __m256 c = _mm256_add_ps(a, b);

    float result[8];
    _mm256_storeu_ps(result, c);

    printf("Result of the addition: ");
    for (int i = 0; i < 8; i++) {
        printf("%f ", result[i]);
    }
    printf("\n");

    return 0;
}

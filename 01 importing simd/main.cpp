/*
* SIMD operations can be included via the appropriate headers
*/

#include "immintrin.h" //AVX2, 256 bit operations (8 floats)

/*
 * immintirn.h is the single header file that includes all the SIMD operations.
 *
 * SIMD headers:
 * #include <mmintrin.h> //MMX
 * #include <xmmintrin.h> //SSE
 * #include <emmintrin.h> //SSE2
 * #include <pmmintrin.h> //SSE3
 * #include <tmmintrin.h> //SSSE3
 * #include <smmintrin.h> //SSE4.1
 * #include <nmmintrin.h> //SSE4.2
 * #include <ammintrin.h> //SSE4A
 * #include <wmmintrin.h> //AES
 * #include <immintrin.h> //AVX, AVX2, FMA, AVX512
 */
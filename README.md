# Hands-on SIMD Programming with C++

## Overview

This repository provides a practical, step-by-step guide to SIMD (Single Instruction, Multiple Data) programming in C++, tailored for beginners. Through a progressive, example-driven approach, it delves into the fundamental techniques of SIMD programming, emphasizing minimal code to cover a broad range of methods. The guide ensures a smooth learning curve for SIMD programming noobs. Key topics include:

 - **SIMD Instruction Sets Overview**: Insight into various SIMD instruction set features.
 - **SIMD Headers**: Introduction to `immintrinsic.h`.
 - **Data Initialization**: Working with types like `__m256`, `__m256d`, `__m256i` and functions such as `_mm256_setzero_ps()`, `_mm256_set1_pd()`, `_mm256_set_epi32()`, and `_mm256_setr_epi16()`.
 - **Accessing SIMD Data**: Techniques including Pointer Conversion and Union.
 - **Loading SIMD Data**: Utilization of `_mm256_load_ps()` and `_mm256_loadu_ps()`.
 - **Mathematical Computations**: Employing functions like `_mm256_add_ps()`, `_mm256_sub_ps()`, `_mm256_hadd_ps()`, `_mm256_addsub_ps()`, `_mm256_mul_ps()`, `_mm256_mullo_epi16()`, `_mm256_mulhi_epi16()`, `_mm256_div_ps()`, `_mm256_fmadd_ps()`.
 - **Practical Examples**: Implementation in scenarios such as vector dot products, conditional code, and solving quadratic equations.

## Getting Started
Certainly, keeping the "Getting Started" section concise while making it a bit more informative can be done with some subtle enhancements. Here's a revised version with just two bullet points:

## Getting Started
1. **Clone the Repository**: Begin by cloning the repository to your local machine. This will provide you with all the necessary files and examples to get started.
   ```bash
   git clone [repo-url]
   ```

2. **Work Through the Chapters**: Once the repository is cloned, navigate through its contents and follow the instructions provided in each chapter. These chapters are designed to sequentially build your understanding and skills in SIMD programming.

## Reference Material
 - [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html): The official guide for Intel SIMD intrinsics, detailing function usage, syntax, and optimization.
 - [SIMD Instructions from CMU 18-613](https://www.cs.cmu.edu/afs/cs/academic/class/15213-s19/www/lectures613/04-simd.pdf): A comprehensive overview of SIMD instruction utilization from CMU.
 - [Simplified SIMD Intrinsics Wrapper](https://gist.github.com/jssonx/85c1561321009e4b41eebfd68b1941ce): A header file providing a clearer abstraction for SIMD intrinsics to enhance code readability and ease of use.
# Application of Linear Algebra

This repository contains implementations of fundamental linear algebra concepts through a series of four homework assignments. Each assignment demonstrates practical applications of linear algebra in areas such as text generation, error-correcting codes, 3D graphics, and image compression.

---

## 1: Markov Chains and Text Generation

Implemented a Markov chain model to generate text based on N-gram analysis.

**Key Features:**
- Text preprocessing and N-gram generation (2-grams to 4-grams)
- Transition matrix computation
- Sequence generation using the Markov model
- Stationary distribution calculation using the power method
- Analysis of how the training text affects sequence variety

---

## 2: Hamming (7,4) Error-Correcting Code

Developed an implementation of the Hamming (7,4) code to demonstrate single-bit error detection and correction.

**Key Features:**
- Matrix-based encoding and decoding
- Parity check matrix implementation
- Single-bit error correction logic
- File recovery from corrupted data
- Mathematical proof of correctness included

---

## 3: 3D Graphics and Lighting

Applied linear algebra techniques to implement 3D rendering and lighting effects, including dynamic animations.

**Key Features:**
- 3D object visualization and rendering
- Phong shading and specular lighting calculations
- Animation with moving light sources
- OBJ file parsing and vertex normal computation
- Custom 3D model rendering (e.g., dodecahedron)

---

## 4: Image Compression with Singular Value Decomposition (SVD)

Focused on compressing grayscale images using Singular Value Decomposition (SVD) and analyzing the results.

**Key Features:**
- `compress_image_by_svd()` performs low-rank approximation using truncated SVD
- Clipping and conversion to uint8 format for valid image output
- Visualizations of original and compressed images at various ranks (e.g., 10, 50, 100)
- Plots of singular values and compression metrics

---

## Technical Details

- **Language:** Python
- **Libraries:** 
  - NumPy for matrix operations and linear algebra
  - Matplotlib for visualization
- **Testing:** Includes unit tests for major functions
- **Documentation:** Mathematical derivations and proofs provided for key algorithms

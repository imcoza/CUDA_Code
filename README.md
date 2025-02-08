# Overview

This repository contains CUDA code generated as part of a learning journey using Programming Massively Parallel Processors (PMPP) by David B. Kirk & Wen-mei W. Hwu and other resources. 
The primary objective is to develop a deep understanding of CUDA programming, parallel computing, and GPU optimizations through hands-on implementation.

# Contents

This repository includes:
<ol>
  <li>Basic CUDA Programs: Fundamental CUDA programs covering thread hierarchy, memory models, and execution.  
  </li>
  <li>Memory Management: Code demonstrating shared memory, global memory, constant memory, and texture memory usage.
  </li>
  <li>Parallel Computing Patterns: Implementations of reduction, scan, stencil computations, and tiled algorithms.
  </li>
  <li>Optimization Techniques: Performance tuning strategies, warp efficiency analysis, and occupancy calculations.
  </li>
  <li>Real-world Applications: Small projects and benchmark tests utilizing CUDA for computational acceleration.
  </li>
  <li>Performance Comparisons: Execution time analysis and optimizations using profiling tools like nvprof and Nsight Compute. </li>     
</ol>

# How to Use

Clone the repository:
git clone https://github.com/imcoza/CUDA_Code.git

Navigate to the project directory:
cd CUDA_Code

Compile and run CUDA programs:
nvcc filename.cu -o output
./output

Use NVIDIA profiling tools for performance analysis:
nvprof ./output

or with Nsight Compute:
nsys profile ./output


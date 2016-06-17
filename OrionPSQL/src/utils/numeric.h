/*
Copyright 2015 Arun Kumar

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <assert.h>
#include <math.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <time.h>

//for regularization
inline void ball_project(double*  x, const int size, const double B, const double B2) {
  double norm_square = 0.0;
  int i;
  for(i = size - 1; i >= 0; i --) {
    norm_square += x[i]*x[i];
  }
  if(norm_square > B2) {
    double norm = sqrt(norm_square);
    int j;
    for(j = size - 1; j >= 0; j --) {
      x[j] *= B / norm;
    }
  }
}

inline double dot(const double* x, const double* y, const int size) {
  double ret = 0.0;
  int i;
  for(i = size - 1; i >= 0; i--) {
    ret += x[i]*y[i];
  }
  return ret;
}

inline double dot_dss(const double* x, const int* k, const double* v, const int sparseSize) {
  double ret = 0.0;
  int i;
  for(i = sparseSize - 1; i >= 0; i--) {
    ret += x[k[i]]*v[i];
  }
  return ret;
}

inline void add_and_scale(double* x, const int size, const double* y, const double c) {
  int i;
  for(i = size - 1; i >= 0; i--) {
    x[i] += y[i]*c;
  }
}

inline void add_c_dss(double* x, const int* k, const int sparseSize, const double c) {
  int i;
  for(i = sparseSize - 1; i >= 0; i--) {
    x[k[i]] += c;
  }
}

inline void add_and_scale_dss(double* x, const int* k, const double* v, const int sparseSize, const double c) {
  int i;
  for(i = sparseSize - 1; i >= 0; i--) {
    x[k[i]] += v[i]*c;
  }
}

inline void scale_i(double* x, const int size, const double c) {
  int i;
  for(i = size - 1; i >= 0; i --) {
    x[i] *= c;
  }
}

inline double norm(const double *x, const int size) {
  double norm = 0;
  int i;
  for(i = size - 1; i >= 0; i --) {
    norm += x[i] * x[i];
  }
  return norm;
}

inline double sigma(const double v) {
  if (v > 30) { return 1.0 / (1.0 + exp(-v)); }
  else { return exp(v) / (1.0 + exp(v)); }
}

inline void l1_shrink_mask(double* x, const double u, const int* k, const int sparseSize) {
  int i;
  for(i = sparseSize-1; i >= 0; i--) {
    if (x[k[i]] > u) { x[k[i]] -= u; }
    else if (x[k[i]] < -u) { x[k[i]] += u; }
    else { x[k[i]] = 0; }
  }
}

inline void l2_shrink_mask_d(double* x, const double u, const int size) {
  int i;
  for(i = size-1; i >= 0; i--) {
	if (x[i] == 0.0) { continue; }
    x[i] /= 1 + u;
  }
}

inline void l1_shrink_mask_d(double* x, const double u, const int size) {
  int i;
  double xi = 0.0;
  for(i = size-1; i >= 0; i--) {
    xi = x[i];
    if (xi > u)		  { x[i] -= u; }
    else if (xi > -u) { x[i] += u; }
    else			  { x[i] = 0.0; }
    //if (x[i] > u) { x[i] -= u; }
    //else if (x[i] < -u) { x[i] += u; }
    //else { x[i] = 0; }
  }
}

inline double log_sum(const double a, const double b) {
	return a + log(1.0 + exp(b - a));
}

//atomic operation compareandswap
inline unsigned char compare_and_swap (volatile int* ptr, int oldvar, int newvar) {
	unsigned char result;
	__asm__ __volatile__ (
			"lock; cmpxchg %3, %1\n"
			"sete %b0\n"
			: "=r"(result),
			  "+m"(*ptr),
			  "+a"(oldvar)
			: "r"(newvar)
			: "memory", "cc"
			);
	return result;
}

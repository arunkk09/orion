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

#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <string>
#include <sstream>
#include <fstream>

//Computes dot (inner) product of vectors x and y 
double dot(int len, double *x, double *y) {
	double ret = 0.0;
	for(int i = len -1; i >= 0; i--) {
		ret += x[i]*y[i];
	}
	return ret;
}

//Gaussian random number generator
void getgaussrv(double &v1, double &v2, double var) {
	double a, b, w;
	do {
		a = 2.0 * drand48() - 1.0;
		b = 2.0 * drand48() - 1.0;
		w = a * a + b * b;
	} while (w >= 1.0);
	w = sqrt((-2.0 * log(w)) / w);
	v1 = a * w * var;
	v2 = b * w * var;
}

//Writes out the files for S and R in space-separated-values format
//S is the entity table (the outer table for the join)
//Schema of S is (SID Target ForeignKey XS)
//R is the attribute table (the inner table for the join)
//Schema of R is (RID XR)
//nS is number of tuples in S, nR is number of tuples in R
//dS is number of features in S (including target), dR is number of features in R
//Variance is a parameter for the Gaussian random number generator (1.0 is recommended)
//Sfile is the prefix of the name for S files
//Rfile is the name of the R file
//The last argument enables generation of only R file or only S files
int main(int argc, char* argv[]) {
	if(argc < 8) {
		std::cout << "gen_data <Ratio of nS:nR> <nR> <dS> <dR> <Variance> <Sfile> <Rfile> [1 for R alone | 2 for S alone]" << std::endl;
		return -1;
	}
	int r = atoi(argv[1]);
	int nR = atoi(argv[2]);
	int dS = atoi(argv[3]); //includes target
	int dR = atoi(argv[4]);
	double var = atof(argv[5]);
	std::cerr << "r " << r << " nR " << nR << " dS " << dS << " dR " << dR << " var " << var << std::endl;

	int dim = dS + dR - 1;
	double w[dim];
	double XS[dS - 1];
	double XR[dR];
	double normsq = 0.0;
	srand48(0xDEADBEEF);

	int onlyflag = 0;
	if(argc > 8) {
		onlyflag = atoi(argv[8]);
	}

	// generate the model
	for(int i = dim - 1; i >= 0; i--) {w[i] = drand48() - 0.5; normsq += w[i]*w[i];}
	double norm = sqrt(normsq);
	for(int i = dim - 1; i >= 0; i--) {w[i] /= norm;}
	//for(int i = dim - 1; i >= 0; i--) { std::cerr << " " << w[i]; if(i > 0) { std::cerr << " "; } }
	// std::cerr << std::endl;

	std::ofstream Sout;
	if(onlyflag != 1) {
		Sout.open(argv[6], std::ios::out);
	}
	std::ofstream Rout;
	if(onlyflag != 2)
		Rout.open(argv[7], std::ios::out);

	// Sample for S and R
	for(int k=1; k <= nR; k++)  {
		// generate a random vector for XR
		for(int i = dR - 1; i >= 0; i-=2) {
			if(i >= 1) {
				getgaussrv(XR[i], XR[i-1], var);
				//XR[i] += s;
				//XR[i-1] +=s;
			} else {
				double y;
				getgaussrv(XR[i], y, var);
				//XR[i] += s;
			}      
		}
		// transform it by w
		for(int i = 0; i < dR; i++) { XR[i] *= w[dS - 1 + i]; }

		for(int ri=1; ri <= r; ri++) {
			// assign this point to a cluster
			double s =  (rand() % 2 == 0) ? -1 : 1;    
			// generate a random vector for XS
			for(int i = dS - 2; i >= 0; i-=2) {
				if(i >= 1) {
					getgaussrv(XS[i], XS[i-1], var);
					XS[i] += s;
					XS[i-1] +=s;
				} else {
					double y;
					getgaussrv(XS[i], y, var);
					XS[i] += s;
				}      
			}
			// transform it by w
			for(int i = 0; i < dS - 1; i++) { XS[i] *= w[i]; }

			if(onlyflag != 1) {
				//output to S (SID Target ForeignKey XS)
				Sout << r * (k - 1) + ri << " " << s << " " << k << " ";
				for(int i = 0; i < dS - 1; i++) { 
					Sout << XS[i]; 
					if(i < dS - 2) { 
						Sout << " "; 
					} 
				}
				Sout << std::endl;
			}
		}

		if(onlyflag != 2) {
			//output to R (RID XR)
			Rout << k << " ";
			for(int i = 0; i < dR; i++) { 
				Rout << XR[i]; 
				if(i < dR - 1) { 
					Rout << " "; 
				} 
			}
			Rout << std::endl;
		}

		// status
		if(k % 1000000 == 0) {std::cerr << "."; }    
	}
	Rout.close();
	Sout.close();
}

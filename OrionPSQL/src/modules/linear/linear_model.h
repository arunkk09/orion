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

#ifndef LINEAR_MODEL_H
#define LINEAR_MODEL_H

#define META_LEN (7)

//a structure for model parameters and meta data
struct LinearModel {
    int mid;
    int token;
    // meta data
    int nDims;
    // weight vector (followed by gradient vector)
    double *w;
};

//a structure for a factorized linear model
struct FactLinearModel {
    int mid;
    int token;
    // meta data
    int nb, dS, dR, gradloc, hashloc; //number of buckets, dS, dR, start of hash in w
    // weight vector (followed by gradient vector and hash table for Fact)
    double *w;
};

//assign initial values to the model
inline void LinearModel_init(struct LinearModel *ptrModel, int mid, int nDims) {
    ptrModel->mid = mid;
    ptrModel->token = 0;
    ptrModel->nDims = nDims;
    ptrModel->w = (double *)(&(ptrModel->w) + 1);
}

//assign initial values to the factorized model
inline void FactLinearModel_init(struct FactLinearModel *ptrModel, int mid, int nb, int dS, int dR) {
    ptrModel->mid = mid;
    ptrModel->token = 0;
    ptrModel->nb = nb;
    ptrModel->dS = dS;
    ptrModel->dR = dR;
    ptrModel->gradloc = dS + dR - 1;
    ptrModel->hashloc = 2*(dS + dR - 1);
    ptrModel->w = (double *)(&(ptrModel->w) + 1);
}

#endif

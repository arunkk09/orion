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

#ifndef LOGIT_H
#define LOGIT_H

inline void dense_logit_grad(struct LinearModel *ptrModel, const double *v, const double y) {
    double wx = dot(ptrModel->w, v, ptrModel->nDims);
    double sig = sigma(-y * wx);
    double c = -y * sig; // scale factor
    add_and_scale(ptrModel->w + ptrModel->nDims, ptrModel->nDims, v, c); //g comes after w
}

inline double dense_logit_grad_loss(struct LinearModel *ptrModel, const double *v, const double y) {
    double wx = dot(ptrModel->w, v, ptrModel->nDims);
    double sig = sigma(-y * wx);
    double c = -y * sig; // scale factor
    add_and_scale(ptrModel->w + ptrModel->nDims, ptrModel->nDims, v, c); //g comes after w
    return log(1 + exp(-y * wx));
}

inline double dense_logit_loss(struct LinearModel *ptrModel, const double *v, const double y) {
    double wx = dot(ptrModel->w, v, ptrModel->nDims);
    return log(1 + exp(-y * wx));
}

inline double dense_logit_pred(struct LinearModel *ptrModel, const double *v) {
    double wx = dot(ptrModel->w, v, ptrModel->nDims);
    return 1. / (1. + exp(-1 * wx));
}

#endif

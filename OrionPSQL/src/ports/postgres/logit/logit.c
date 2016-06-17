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

#include "../c_udf_helper.h"
#include "utils/numeric.h"
#include "modules/linear/linear_model.h"
#include "modules/logit/logit.h"
#include "../myhash.h"

PG_FUNCTION_INFO_V1(init);
PG_FUNCTION_INFO_V1(grad);
PG_FUNCTION_INFO_V1(final);
PG_FUNCTION_INFO_V1(loss);
PG_FUNCTION_INFO_V1(pred);
PG_FUNCTION_INFO_V1(getmodel);
PG_FUNCTION_INFO_V1(putmodel);

PG_FUNCTION_INFO_V1(fact_init);
PG_FUNCTION_INFO_V1(fact_func1);
PG_FUNCTION_INFO_V1(fact_func2);
PG_FUNCTION_INFO_V1(fact_func3);
PG_FUNCTION_INFO_V1(fact_final);
PG_FUNCTION_INFO_V1(fact_getmodel);
PG_FUNCTION_INFO_V1(fact_putmodel);

//similar to init, except the memory space also stores the hash of statistics
//fact_linear_model:  mid integer, nb integer, dS integer, dR integer
Datum fact_init(PG_FUNCTION_ARGS) {
    int32 mid = PG_GETARG_INT32(0);
    int32 nb = PG_GETARG_INT32(1);
    int32 dS = PG_GETARG_INT32(2);
    int32 dR = PG_GETARG_INT32(3);
    int ndims = dS + dR - 1;
    struct FactLinearModel* ptrModel;
    int size = sizeof(struct FactLinearModel) + sizeof(double) * (ndims * 2 + nb * 3 + ndims);
    int shmid = shmget(ftok("/", mid), size, SHM_R | SHM_W | IPC_CREAT);
    if (shmid == -1) { elog(ERROR, "In init, shmget failed!\n"); }
    // elog(WARNING, "init: after shmget\n");
    ptrModel = (struct FactLinearModel*) shmat(shmid, NULL, 0);
    FactLinearModel_init(ptrModel, mid, nb, dS, dR);
    //elog(WARNING, "fact_init: nb %d dS %d dR %d allocated shmem size %d\n", nb, dS, dR, size);
    PG_RETURN_NULL();
}

//compute the hash of statistics based on the pass over R
//args are (mid, rid, xR)
Datum fact_func1(PG_FUNCTION_ARGS) {
    int32 mid = PG_GETARG_INT32(0);
    struct FactLinearModel modelBuffer;
    struct FactLinearModel* ptrModel = &modelBuffer;
    static struct FactLinearModel* ptrSharedModel = NULL;
    if (ptrSharedModel == NULL || ptrSharedModel->mid != mid) {
        ptrSharedModel = (struct FactLinearModel*)get_model_by_mid(mid);
        // elog(WARNING, "grad: NO");
    }
    *ptrModel = (*ptrSharedModel);
    ptrModel->w = (double *)(&(ptrSharedModel->w) + 1);
    unsigned int rid = (unsigned int) PG_GETARG_INT32(1);
    struct varlena* v2 = (struct varlena*) PG_GETARG_RAW_VARLENA_P(2);
    float8* v;
    int len = my_parse_array_no_copy(v2, sizeof(float8), (char **)&v);
    //if((rid % 10000) == 0)
    //    elog(WARNING, "fact_func1: mid %d rid %d v len %d\n", mid, rid, len);
    assert(len == ptrModel->dR);
    //compute partial dot products and insert into hash table's v1
    double w2xi2 = dot(ptrModel->w + (ptrModel->dS - 1), v, ptrModel->dR);
    add_entry(rid, w2xi2, 0.0, ptrModel->w + ptrModel->hashloc, ptrModel->nb);
    //if((rid % 10000) == 0)
    //    elog(WARNING, "fact_func1: added entry for rid %d with w2xi2 %lf\n", rid, w2xi2);
    PG_RETURN_NULL();
}

//update the hash of statistics based on the pass over S and obtain partial gradient
//with grad+loss integration, we also output sum of loss of each example
//args are (mid, fk, xS, y)
Datum fact_func2(PG_FUNCTION_ARGS) {
    int32 mid = PG_GETARG_INT32(0);
    struct FactLinearModel modelBuffer;
    struct FactLinearModel* ptrModel = &modelBuffer;
    static struct FactLinearModel* ptrSharedModel = NULL;
    if (ptrSharedModel == NULL || ptrSharedModel->mid != mid) {
        ptrSharedModel = (struct FactLinearModel*)get_model_by_mid(mid);
        // elog(WARNING, "grad: NO");
    }
    *ptrModel = (*ptrSharedModel);
    ptrModel->w = (double *)(&(ptrSharedModel->w) + 1);
    unsigned int fk = (unsigned int) PG_GETARG_INT32(1);
    struct varlena* v2 = (struct varlena*) PG_GETARG_RAW_VARLENA_P(2);
    float8* v;
    int len = my_parse_array_no_copy(v2, sizeof(float8), (char **)&v);
    float8 y = PG_GETARG_FLOAT8(3);
    //if((fk % 100) == 0)
    //    elog(WARNING, "fact_func2: mid %d fk %d y %lf len v %d\n", mid, fk, y, len);
    assert(len == (ptrModel->dS - 1));
    //compute partial dot products, probe into hash table and get full dot pdt
    unsigned int fndbkt;
    bool found = find_bkt(fk, ptrModel->w + ptrModel->hashloc, ptrModel->nb, &fndbkt);
    if(!found) { elog(ERROR, "In fact_func2, find_bkt failed on fk %d!\n", fk); }
    unsigned int fsid = (unsigned int) ptrModel->w[ptrModel->hashloc + 3*fndbkt];
    if(fk != fsid) { elog(ERROR, "In fact_func2, found bucket has mismatched rid %d instead of fk %d!\n", fsid, fk); }
    double fulldot = dot(ptrModel->w, v, ptrModel->dS - 1) + ptrModel->w[ptrModel->hashloc + 3*fndbkt + 1];
    double gwx = -y * sigma(-y * fulldot); //scale factor g(w.x)
    //update v2 with sum of g(w.x); automatic group by
    ptrModel->w[ptrModel->hashloc + 3*fndbkt + 2] += gwx;
    //add to first portion of gradient by scaling xS by gwx
    add_and_scale(ptrModel->w + ptrModel->gradloc, ptrModel->dS - 1, v, gwx); //g comes after w
    //if((fk % 100) == 0)
    //    elog(WARNING, "fact_func2: done with r tuple with fk %d with gwx %lf\n", fk, gwx);
    //with grad+loss, we also compute the loss of the given w vector
    double err = log(1 + exp(-y * fulldot));
    PG_RETURN_FLOAT8(err);
    //PG_RETURN_NULL();
}

//complete gradient computations by passing over R
//args are (mid, rid, xR)
Datum fact_func3(PG_FUNCTION_ARGS) {
    int32 mid = PG_GETARG_INT32(0);
    struct FactLinearModel modelBuffer;
    struct FactLinearModel* ptrModel = &modelBuffer;
    static struct FactLinearModel* ptrSharedModel = NULL;
    if (ptrSharedModel == NULL || ptrSharedModel->mid != mid) {
        ptrSharedModel = (struct FactLinearModel*)get_model_by_mid(mid);
        // elog(WARNING, "grad: NO");
    }
    *ptrModel = (*ptrSharedModel);
    ptrModel->w = (double *)(&(ptrSharedModel->w) + 1);
    unsigned int rid = (unsigned int) PG_GETARG_INT32(1);
    struct varlena* v2 = (struct varlena*) PG_GETARG_RAW_VARLENA_P(2);
    float8* v;
    int len = my_parse_array_no_copy(v2, sizeof(float8), (char **)&v);
    //if((rid % 10000) == 0)
    //    elog(WARNING, "fact_func3: mid %d rid %d v len %d\n", mid, rid, len);
    assert(len == ptrModel->dR);
    //probe into hash table and get full dot pdt
    unsigned int fndbkt;
    bool found = find_bkt(rid, ptrModel->w + ptrModel->hashloc, ptrModel->nb, &fndbkt);
    if(!found) { elog(ERROR, "In fact_func3, find_bkt failed on rid %d!\n", rid); }
    unsigned int fsid = (unsigned int) ptrModel->w[ptrModel->hashloc + 3*fndbkt];
    if(rid != fsid) { elog(ERROR, "In fact_func3, found bucket has mismatched rid %d instead of rid %d!\n", fsid, rid); }
    //add to second portion of gradient by scaling xR by v2 from hash
    add_and_scale(ptrModel->w + (ptrModel->gradloc + ptrModel->dS - 1), ptrModel->dR, v, ptrModel->w[ptrModel->hashloc + 3*fndbkt + 2]);
    //if((rid % 10000) == 0)
    //    elog(WARNING, "fact_func3: done tuple with rid %d with scale %lf\n", rid, ptrModel->w[ptrModel->hashloc + 3*fndbkt + 2]);
    PG_RETURN_NULL();
}

//remove the model from shmem
Datum fact_final(PG_FUNCTION_ARGS) {
    ArrayType *warray;
    int32 mid = PG_GETARG_INT32(0);
    int shmid = shmget(ftok("/", mid), 0, SHM_R | SHM_W);
    if (shmid == -1) {	elog(ERROR, "In final, shmget failed!\n"); }
    struct shmid_ds shm_buf;
    if (shmctl(shmid, IPC_RMID, &shm_buf) == -1) {
    	elog(ERROR, "shmctl failed in final()");
    }
    PG_RETURN_NULL();
}

//return the model and gradient vector
Datum fact_getmodel(PG_FUNCTION_ARGS) {
    ArrayType *warray;
    double *w;
    int wLen; 
    int32 mid = PG_GETARG_INT32(0);
    struct FactLinearModel* ptrSharedModel = (struct FactLinearModel*) get_model_by_mid(mid);
    wLen = ptrSharedModel->hashloc; //w and g only (not the hash table)
    warray = my_construct_array(wLen, sizeof(float8), FLOAT8OID);
    wLen = my_parse_array_no_copy((struct varlena *)warray, sizeof(float8), (char **)&w);
    //elog(WARNING, "In fact_getmodel, wLen %d copied", wLen);
    memcpy(w, &(ptrSharedModel->w) + 1, wLen * sizeof(float8));
    PG_RETURN_ARRAYTYPE_P(warray);
}

//set the w vector to the given vector; clean up the space of g and hash table
Datum fact_putmodel(PG_FUNCTION_ARGS) {
    int32 mid = PG_GETARG_INT32(0);
    struct FactLinearModel modelBuffer;
    struct FactLinearModel* ptrModel = &modelBuffer;
    static struct FactLinearModel* ptrSharedModel = NULL;
    if (ptrSharedModel == NULL || ptrSharedModel->mid != mid) {
        ptrSharedModel = (struct FactLinearModel*)get_model_by_mid(mid);
        // elog(WARNING, "grad: NO");
    }
    *ptrModel = (*ptrSharedModel);
    ptrModel->w = (double *)(&(ptrSharedModel->w) + 1);
    struct varlena* v2 = (struct varlena*) PG_GETARG_RAW_VARLENA_P(1);
    float8* v;
    int len = my_parse_array_no_copy(v2, sizeof(float8), (char **)&v);
    assert(len == ptrModel->gradloc);
    memcpy(ptrModel->w, v, sizeof(float8) * len);
    memset(ptrModel->w + ptrModel->gradloc, '\0', sizeof(float8) * (ptrModel->gradloc + 3 * ptrModel->nb));
    PG_RETURN_NULL();
}

//linear_model:  mid integer, ndims integer
Datum init(PG_FUNCTION_ARGS) {
    int32 mid = PG_GETARG_INT32(0);
    int32 ndims = PG_GETARG_INT32(1);
    struct LinearModel* ptrModel;
    int size = sizeof(struct LinearModel) + sizeof(double) * (ndims * 3);
    int shmid = shmget(ftok("/", mid), size, SHM_R | SHM_W | IPC_CREAT);
    if (shmid == -1) { elog(ERROR, "In init, shmget failed!\n"); }
    // elog(WARNING, "init: after shmget\n");
    ptrModel = (struct LinearModel*) shmat(shmid, NULL, 0);
    LinearModel_init(ptrModel, mid, ndims);
    PG_RETURN_NULL();
}

//now it computes grad using w and returns loss too
Datum grad(PG_FUNCTION_ARGS) {
    int32 mid = PG_GETARG_INT32(0);
    struct LinearModel modelBuffer;
    struct LinearModel* ptrModel = &modelBuffer;
    static struct LinearModel* ptrSharedModel = NULL;
    if (ptrSharedModel == NULL || ptrSharedModel->mid != mid) {
        ptrSharedModel = (struct LinearModel*)get_model_by_mid(mid);
        // elog(WARNING, "grad: NO");
    }
    *ptrModel = (*ptrSharedModel);
    ptrModel->w = (double *)(&(ptrSharedModel->w) + 1);
    struct varlena* v2 = (struct varlena*) PG_GETARG_RAW_VARLENA_P(1);
    float8* v;
    int len = my_parse_array_no_copy(v2, sizeof(float8), (char **)&v);
    // assert(len == N_DIMS);
    float8 y = PG_GETARG_FLOAT8(2);
    double err = dense_logit_grad_loss(ptrModel, v, y);
    PG_RETURN_FLOAT8(err);
}

Datum getmodel(PG_FUNCTION_ARGS) {
    ArrayType *warray;
    double *w;
    int wLen; 
    int32 mid = PG_GETARG_INT32(0);
    struct LinearModel* ptrSharedModel = (struct LinearModel*) get_model_by_mid(mid);
    wLen = 2 * ptrSharedModel->nDims; //w and g
    warray = my_construct_array(wLen, sizeof(float8), FLOAT8OID);
    wLen = my_parse_array_no_copy((struct varlena *)warray, sizeof(float8), (char **)&w);
    memcpy(w, &(ptrSharedModel->w) + 1, wLen * sizeof(float8));
    PG_RETURN_ARRAYTYPE_P(warray);
}

//set the w vector the given vector; clean up the space of g
Datum putmodel(PG_FUNCTION_ARGS) {
    int32 mid = PG_GETARG_INT32(0);
    struct LinearModel modelBuffer;
    struct LinearModel* ptrModel = &modelBuffer;
    static struct LinearModel* ptrSharedModel = NULL;
    if (ptrSharedModel == NULL || ptrSharedModel->mid != mid) {
        ptrSharedModel = (struct LinearModel*)get_model_by_mid(mid);
        // elog(WARNING, "grad: NO");
    }
    *ptrModel = (*ptrSharedModel);
    ptrModel->w = (double *)(&(ptrSharedModel->w) + 1);
    struct varlena* v2 = (struct varlena*) PG_GETARG_RAW_VARLENA_P(1);
    float8* v;
    int len = my_parse_array_no_copy(v2, sizeof(float8), (char **)&v);
    assert(len == ptrModel->nDims);
    memcpy(ptrModel->w, v, sizeof(float8) * len);
    memset(ptrModel->w + ptrModel->nDims, '\0', sizeof(float8) * ptrModel->nDims);
    PG_RETURN_NULL();
}

//remove the model from shmem
Datum final(PG_FUNCTION_ARGS) {
    ArrayType *warray;
    int32 mid = PG_GETARG_INT32(0);
    int shmid = shmget(ftok("/", mid), 0, SHM_R | SHM_W);
    if (shmid == -1) {	elog(ERROR, "In final, shmget failed!\n"); }
    struct shmid_ds shm_buf;
    if (shmctl(shmid, IPC_RMID, &shm_buf) == -1) {elog(ERROR, "shmctl failed in final()");}
    PG_RETURN_NULL();
}

Datum loss(PG_FUNCTION_ARGS) {
    int32 mid = PG_GETARG_INT32(0);
    struct LinearModel modelBuffer;
    struct LinearModel* ptrModel = &modelBuffer;
    static struct LinearModel* ptrSharedModel = NULL;
    if (ptrSharedModel == NULL || ptrSharedModel->mid != mid) {
        ptrSharedModel = (struct LinearModel*) get_model_by_mid(mid);
        // elog(WARNING, "grad: NO");
    }
    *ptrModel = (*ptrSharedModel);
    ptrModel->w = (double *) (&(ptrSharedModel->w) + 1);
    struct varlena* v2 = (struct varlena*) PG_GETARG_RAW_VARLENA_P(1);
    float8* v;
    int len = my_parse_array_no_copy(v2, sizeof(float8), (char **)&v);
    // assert(len == N_DIMS);
    float8 y = PG_GETARG_FLOAT8(2);
    double err = dense_logit_loss(ptrModel, v, y);
    PG_RETURN_FLOAT8(err);
}

Datum pred(PG_FUNCTION_ARGS) {
    int32 mid = PG_GETARG_INT32(0);
    struct LinearModel modelBuffer;
    struct LinearModel* ptrModel = &modelBuffer;
    static struct LinearModel* ptrSharedModel = NULL;
    if (ptrSharedModel == NULL || ptrSharedModel->mid != mid) {
        ptrSharedModel = (struct LinearModel*) get_model_by_mid(mid);
        // elog(WARNING, "grad: NO");
    }
    *ptrModel = (*ptrSharedModel);
    ptrModel->w = (double *) (&(ptrSharedModel->w) + 1);
    struct varlena* v2 = (struct varlena*) PG_GETARG_RAW_VARLENA_P(1);
    float8* v;
    int len = my_parse_array_no_copy(v2, sizeof(float8), (char **)&v);
    double pred = -1;
    pred = dense_logit_pred(ptrModel, v);
    PG_RETURN_FLOAT8(pred);
}

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

#include "utils/numeric.h"
#include "../c_udf_helper.h"

//the proof of postgresql version 1 C UDF
PG_FUNCTION_INFO_V1(alloc_float8_array);
PG_FUNCTION_INFO_V1(alloc_float8_array_random);

//alloc and return a huge float8 array
Datum alloc_float8_array(PG_FUNCTION_ARGS) {
    int ndims = PG_GETARG_INT32(0);
    PG_RETURN_ARRAYTYPE_P(my_construct_array(ndims, sizeof(float8), FLOAT8OID));
}

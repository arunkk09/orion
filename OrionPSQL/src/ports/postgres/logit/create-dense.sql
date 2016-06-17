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

--------------------------------------------------------------------------
-- factorized learning functions
--------------------------------------------------------------------------

--fact_linear_model:  mid integer, nb integer, ds integer, dr integer, w double precision[] 
DROP FUNCTION IF EXISTS dense_logit_fact_shmem_push(mid integer, nb integer, ds integer, dr integer) CASCADE;
CREATE FUNCTION dense_logit_fact_shmem_push(mid integer, nb integer, ds integer, dr integer)
RETURNS VOID
AS 'dense-logit-shmem', 'fact_init'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS dense_logit_fact_func1(integer, integer, double precision[]) CASCADE;
CREATE FUNCTION dense_logit_fact_func1(integer, integer, double precision[])
RETURNS VOID
AS 'dense-logit-shmem', 'fact_func1'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS dense_logit_fact_func2(integer, integer, double precision[], double precision) CASCADE;
CREATE FUNCTION dense_logit_fact_func2(integer, integer, double precision[], double precision)
RETURNS double precision
AS 'dense-logit-shmem', 'fact_func2'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS dense_logit_fact_func3(integer, integer, double precision[]) CASCADE;
CREATE FUNCTION dense_logit_fact_func3(integer, integer, double precision[])
RETURNS VOID
AS 'dense-logit-shmem', 'fact_func3'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS dense_logit_fact_shmem_pop(integer) CASCADE;
CREATE FUNCTION dense_logit_fact_shmem_pop(integer)
RETURNS VOID
AS 'dense-logit-shmem', 'fact_final'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS dense_logit_fact_putmodel(integer, double precision[]) CASCADE;
CREATE FUNCTION dense_logit_fact_putmodel(integer, double precision[])
RETURNS VOID
AS 'dense-logit-shmem', 'fact_putmodel'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS dense_logit_fact_getmodel(integer) CASCADE;
CREATE FUNCTION dense_logit_fact_getmodel(integer)
RETURNS double precision []
AS 'dense-logit-shmem', 'fact_getmodel'
LANGUAGE C STRICT;

--------------------------------------------------------------------------
-- for shared memory version
--------------------------------------------------------------------------

--linear_model:  mid integer, ndims integer, w double precision[] 
DROP FUNCTION IF EXISTS dense_logit_shmem_push(mid integer, ndims integer) CASCADE;
CREATE FUNCTION dense_logit_shmem_push(mid integer, ndims integer)
RETURNS VOID
AS 'dense-logit-shmem', 'init'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS dense_logit_grad(integer, double precision[], double precision) CASCADE;
CREATE FUNCTION dense_logit_grad(integer, double precision[], double precision)
RETURNS double precision
AS 'dense-logit-shmem', 'grad'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS dense_logit_shmem_pop(integer) CASCADE;
CREATE FUNCTION dense_logit_shmem_pop(integer)
RETURNS VOID
AS 'dense-logit-shmem', 'final'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS dense_logit_putmodel(integer, double precision[]) CASCADE;
CREATE FUNCTION dense_logit_putmodel(integer, double precision[])
RETURNS VOID
AS 'dense-logit-shmem', 'putmodel'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS dense_logit_getmodel(integer) CASCADE;
CREATE FUNCTION dense_logit_getmodel(integer)
RETURNS double precision []
AS 'dense-logit-shmem', 'getmodel'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS dense_logit_loss(integer, double precision[], double precision) CASCADE;
CREATE FUNCTION dense_logit_loss(integer, double precision[], double precision)
RETURNS double precision
AS 'dense-logit-shmem', 'loss'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS dense_logit_pred(integer, double precision[]) CASCADE;
CREATE FUNCTION dense_logit_pred(integer, double precision[])
RETURNS double precision
AS 'dense-logit-shmem', 'pred'
LANGUAGE C STRICT;

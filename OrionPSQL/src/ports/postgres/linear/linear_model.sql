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

DROP TABLE IF EXISTS linear_model CASCADE;

CREATE TABLE linear_model (
	mid 			integer,
	ndims			integer,
	ntuples			integer,
	mu			double precision,
	stepsize		double precision,
	decay			double precision,
	w			double precision [])
--DISTRIBUTED BY (mid);
;

DROP TABLE IF EXISTS fact_linear_model CASCADE;

CREATE TABLE fact_linear_model (
	mid 			integer,
	nr			integer,
	nb			integer,
	ds			integer,
	dr			integer,
	mu			double precision,
	stepsize		double precision,
	decay			double precision,
	w			double precision [])
;

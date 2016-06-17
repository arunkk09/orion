# orion
Learning GLMs over Joins

This repo contains the source code for the Orion system that implements GLMs (specifically, logistic regression) over joins, especially factorized learning, over the PostgreSQL RDBMS and Apache Hive. Refer to the SIGMOD 2015 paper titled "Learning Generalized Linear Models Over Normalized Data" for technical details.

There are three folders:

1. OrionPSQL:
Source code in C for the PostgreSQL user-defined functions that implement BGD over joins for logistic regression. Python front-end to invoke Orion on a given dataset and model specification.

2. OrionHive:
Source code in Java for the Map-Reduce functions that implement BGD over joins for logistic regression.

3. OrionDataGen:
Synthetic data generators for both PostgreSQL and Hive to create a two-table dataset.

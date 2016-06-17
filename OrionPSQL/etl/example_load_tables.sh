echo "Loading S"
cat s | perl psql_rep_S.pl | psql -c "COPY s FROM STDIN DELIMITER ' '";

echo "Loading R"
cat r | perl psql_rep_R.pl | psql -c "COPY r FROM STDIN DELIMITER ' '";

echo "Loading S"
cat s | perl psql_rep_S.pl | psql -c "COPY news FROM STDIN DELIMITER ' '";

echo "Loading R"
cat r | perl psql_rep_R.pl | psql -c "COPY newr FROM STDIN DELIMITER ' '";

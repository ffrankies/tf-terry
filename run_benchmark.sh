for batch_size in 5 10 15 20 25 30 35 40
do
    for truncate in 5 10 15 20 25 30 35 40
    do
        time python3 run_rnn.py options -m benchmark -d small-sents-c -hs 200 -es 200 -e 1000 -b $batch_size -t $truncate;
    done
done
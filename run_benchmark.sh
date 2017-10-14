for batch_size in 50 100 200 300 400 500 1000
do
    for truncate in 1 2 4 6 8 10 15 20 25
    do
        echo "Benchmarking training for batch size: $batch_size and truncate: $truncate"
        time python3 run_rnn.py options -m benchmark -d small-sents-c.pkl -hs 200 -es 200 -e 100 -b $batch_size -t $truncate 2> tensorflow_stuff.out;
    done
done

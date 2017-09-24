for file in ./dataset_configs/*
do
  python3 dataset_maker.py dataset -c "$file";
done

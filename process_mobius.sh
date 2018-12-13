sudo rm -r /efs/sid/mobius_data/mimic/clean
python process_mobius.py
sudo mv output/ /efs/sid/mobius_data/mimic/clean
python train.py
aws s3 cp hyper.json s3://slurm-cluster/sidsachdeva/

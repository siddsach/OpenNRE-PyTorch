sudo rm -r /efs/sid/mobius_data/mimic/output
python process_mobius.py
sudo mv output/ /efs/sid/mobius_data/mimic/
python train.py
aws s3 cp results.json s3://slurm-cluster/sidsachdeva/

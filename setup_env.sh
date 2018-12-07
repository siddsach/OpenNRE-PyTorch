git clone https://github.com/siddsach/OpenNRE-PyTorch
cd OpenNRE-PyTorch

sudo yum install centos-release-scl -y
sudo yum install rh-python36 -y
scl enable rh-python36 bash

python3 -m virtualenv .env
source .env/bin/activate
sudo pip3 install -r requirements.txt

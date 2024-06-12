# Run from MLP-CW4 directory
module load anaconda
module load phys/compilers/gcc/11.2.0

conda create -n ircot python=3.8.0 -y && conda activate ircot
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Download the processed data
./download/processed_data.sh

# Download the raw data
./download/raw_data.sh

# Install elastic search
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz.sha512
shasum -a 512 -c elasticsearch-7.10.2-linux-x86_64.tar.gz.sha512
tar -xzf elasticsearch-7.10.2-linux-x86_64.tar.gz

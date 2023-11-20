pip install opendelta
sed -i '6d' /usr/local/lib/python3.8/dist-packages/opendelta/basemodel.py
pip install jieba
python -m nltk.downloader punkt
python -m nltk.downloader stopwords
python -m nltk.downloader wordnet

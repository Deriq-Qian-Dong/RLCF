import glob
from beir.datasets.data_loader import GenericDataLoader

for dir_path in glob.glob('beir/*/'):
    print(dir_path)
    corpus, queries, qrels = GenericDataLoader(data_folder=dir_path).load(split="test")
    collection_saved = []
    pstr2pid = {}
    for key in list(corpus.keys()):
        psg = corpus[key]['title'].replace("\t"," ").replace("\n"," ").replace("\r"," ").replace('"','')+' '+corpus[key]['text'].replace("\t"," ").replace("\n"," ").replace("\r"," ").replace('"','')+'\n'
        if psg.strip():
            pstr2pid[key]=len(pstr2pid)
            tmp = str(pstr2pid[key])+'\t'+psg
            collection_saved.append(tmp)
    with open(dir_path+"collection.tsv","w") as f:
        f.writelines(collection_saved)
    qstr2qid={}
    qry_saved = []
    for key in list(queries.keys()):
        qstr2qid[key]=len(qstr2qid)
        tmp = str(qstr2qid[key])+'\t'+queries[key]+'\n'
        qry_saved.append(tmp)
    with open(dir_path+"test.query.tsv","w") as f:
        f.writelines(qry_saved)
    qrels_saved = []
    for qstr in qrels:
        qid = qstr2qid[qstr]
        for pstr in qrels[qstr]:
            try:
                pid = pstr2pid[pstr]
                label = qrels[qstr][pstr]
                qrels_saved.append(str(qid)+'\t'+str(pid)+'\t'+str(label)+'\n')
            except:
                print(pstr)
    with open(dir_path+'test.qrels.tsv','w') as f:
        f.writelines(qrels_saved)

from FlagEmbedding import BGEM3FlagModel

# 填写模型路径
model = BGEM3FlagModel('models/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181',  use_fp16=True) 

# 待计算的句子
sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

sentence_pairs = [[i,j] for i in sentences_1 for j in sentences_2]

# 计算混合相似度
# w[0]*dense_score + w[1]*sparse_score + w[2]*colbert_score
print(model.compute_score(sentence_pairs, 
                          max_passage_length=128, 
                          weights_for_different_modes=[0.4, 0.2, 0.4])) 
from FlagEmbedding import BGEM3FlagModel

# 填写模型路径
model = BGEM3FlagModel('models/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181',  
                       use_fp16=True)

# 待计算的句子
sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

# 计算 Embedding
embeddings_1 = model.encode(sentences_1, 
                            batch_size=12, 
                            max_length=8192, 
                            )['dense_vecs']
embeddings_2 = model.encode(sentences_2)['dense_vecs']

# 计算相似度
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
# 结果应该是：
# [[0.6265, 0.3477], [0.3499, 0.678 ]]
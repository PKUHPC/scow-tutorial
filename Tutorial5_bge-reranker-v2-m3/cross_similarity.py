from FlagEmbedding import FlagReranker

# 填写模型路径
reranker = FlagReranker('models/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e', use_fp16=True) 

# 计算相似度
score = reranker.compute_score(['query', 'passage'], normalize=True)
print(score) # 0.003497010252573502

scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']], normalize=True)
print(scores) # [0.00027803096387751553, 0.9948403768236574]
import numpy as np

# 计算两个向量的余弦相似度 相似度 = <a, b> / (||a|| * ||b||)
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)
embedding_query = np.array([0.2, 0.8, 0.5, 0.3, 0.9])
embedding_doc1 = np.array([0.3, 0.7, 0.6, 0.4, 0.8])  # 相关文档
embedding_doc2 = np.array([0.9, 0.1, 0.2, 0.8, 0.3])  # 不相关文档

sim1 = cosine_similarity(embedding_query,embedding_doc1)
sim2 = cosine_similarity(embedding_query,embedding_doc2)
print("=== 向量相似度计算 ===")
print(f"查询向量: {embedding_query}")
print(f"文档1相似度: {sim1:.3f}")
print(f"文档2相似度: {sim2:.3f}")
print(f"\n最相关文档: {'文档1' if sim1 > sim2 else '文档2'}")

# NumPy 数组操作
embeddings = np.array([embedding_query, embedding_doc1, embedding_doc2])
print(f"\n嵌入矩阵形状: {embeddings.shape}")  # (3, 5)
print(f"平均值: {embeddings.mean(axis=0)}")
print(f"标准差: {embeddings.std(axis=0)}")

# 批量计算相似度
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
similarities = sklearn_cosine([embedding_query], [embedding_doc1, embedding_doc2])
print(f"\n批量相似度: {similarities}")
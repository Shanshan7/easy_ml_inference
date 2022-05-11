import hnswlib
import numpy as np


neighbor_count = 9
embedding_train = np.fromfile("./embedding_train.bin", dtype=np.float32)
embedding_t = np.fromfile("./embedding_test.bin", dtype=np.float32)
embedding_coreset = embedding_train.reshape(194, 1440)
# embedding_coreset = np.ones([194, 1440]) * 1.5
# embedding_coreset =  np.tile(np.array(range(0,1440)) / 200, (194,1))
embedding_test =  embedding_t.reshape(324, 1440) # np.ones([324, 1440]) * 1.2 np.tile(np.array(range(0,1440)) / 100, (324,1))# 
# embedding_test =  np.ones([324, 1440]) * 1.2

# for i in range(0, 19):
#     for j in range(0, 40):
#         embedding_coreset[i,j] = 1.5 - i/19 + j/40

# for i in range(0, 32):
#     for j in range(0, 40):
#         embedding_test[i,j] = 1.2 + i/32 - j/40

nbrs = hnswlib.Index(space='l2', dim=len(embedding_coreset[0]))
nbrs.init_index(max_elements=len(embedding_coreset), ef_construction=neighbor_count * 5, M=16) # K*10 == K*5
int_labels = nbrs.add_items(embedding_coreset, np.arange(len(embedding_coreset)))
nbrs.set_ef(neighbor_count * 5)
_, score_patches = nbrs.knn_query(embedding_test, neighbor_count)
# for row in range(0, embedding_test.shape[0]):
#     for k in range(0, neighbor_count):
#         print("l: {}, k: {}, dist: {}".format(row, k, score_patches[row, k]))
N_b = score_patches[np.argmax(score_patches[:, 0])]
w = (1 - (np.max(np.exp(N_b)) / np.sum(np.exp(N_b))))
score = w * max(score_patches[:, 0])  # Image-level score
print("score: ", score)
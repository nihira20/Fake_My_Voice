import torch
import torch.nn as nn
# ge2e loss
class GE2ELoss(nn.Module):
# w is the weights and b is the bias, both can be updated later
    def __init__(self, init_w=10.0, init_b=-5.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, embeddings, labels):
        #embeddings are dim (batch, embedding_size) ie (batch,256)
        unique_labels = torch.unique(labels)
        #calculate the centroid ie avg of all their embeddings
        centroids = []
        for ul in unique_labels:
            centroids.append(embeddings[labels==ul].mean(0))
        centroids = torch.stack(centroids)
        
        #cosine similarity
        #create a matrix where rows is the batch size ie 16 ie 16clips, and the columns are the number of unique speakers inthat specific batch 
        sim_matrix = torch.matmul(embeddings, centroids.T)
        sim_matrix = self.w * sim_matrix + self.b
        
        target = torch.zeros_like(labels)
        for i, lbl in enumerate(labels):
            target[i] = (unique_labels == lbl).nonzero(as_tuple=True)[0]
        
        loss = nn.functional.cross_entropy(sim_matrix, target.to(embeddings.device))
        return loss
# compare each embedding to all centroids,train embeddings so they are closesttt o their own centroid
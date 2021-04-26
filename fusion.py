import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.notebook import tqdm


class TransE(nn.Module):
    """
    A model for TransE
    """

    def __init__(
        self,
        kg_obj,
        margin=1,
        embed_dim=100,
        lr=0.01,
        weight_decay=1e-4,
        n_epochs=1,
    ):
        super(TransE, self).__init__()

        self.kg_obj = kg_obj
        self.embed_dim = embed_dim  # embedding dimension
        self.num_ent = len(kg_obj.entities)  # number of entities
        self.num_rel = len(kg_obj.relations)  # number of relations
        self.margin = margin  # margin
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs

        # Initialize model weights
        self.ent_embeds = nn.Embedding(self.num_ent, self.embed_dim)
        self.rel_embeds = nn.Embedding(self.num_rel, self.embed_dim)

        # Initialize weights using xavier
        # TODO: Use method from paper like I did above
        nn.init.xavier_uniform_(self.ent_embeds.weight.data)
        nn.init.xavier_uniform_(self.rel_embeds.weight.data)

    def forward(self, data):
        """
        input: training triplet
        output: score to be inserted to the loss
        """
        # print(data)
        # Get embeddings from input tuple
        triple = self.idx2embeds(data)
        # Get corrupted triple
        corrupted_triple = self.get_corrupted_triple(triple)
        # get scores/energies/distances for both triples
        pos = self.get_distance(triple)
        neg = self.get_distance(corrupted_triple)
        # return score including margin
        return pos - neg + self.margin

    def get_distance(self, triple):
        """
        input: triple (or numpy array of triples)
        output: distance score
        """
        h = triple["h"]
        r = triple["r"]
        t = triple["t"]
        score = (h + r) - t
        return torch.norm(score, dim=-1)

    def idx2embeds(self, data):
        """
        input: tuple of entity/rel idx - ex: (1,0,2)
        output: dictionary of embeddings
        ex:
        triple['h'] = [1.01, 4.67, 0.12, 0.5] for k = 4
        """
        triple = {}
        h = self.ent_embeds(data[-1, 0])
        t = self.ent_embeds(data[-1, 2])
        r = self.ent_embeds(data[-1, 1])
        # Normalizing constraint (not sure if this is best way to apply this -
        # check out "max norm" prop in embeddings)
        triple["h"] = F.normalize(h, dim=-1)
        triple["t"] = F.normalize(t, dim=-1)
        triple[
            "r"
        ] = r  # according to paper - they don't normalize the relations
        return triple

    def get_corrupted_triple(self, triple):
        """
        input: triple dict
        output: corrupted triple dict
        """
        # Need to randomly create the corrupted triplet by
        # replacing either head or tail with random entity
        corrupted_ent = np.random.choice(["h", "t"], 1, p=[0.5, 0.5]).item()
        random_ent_idx = torch.LongTensor(
            np.random.randint(low=0, high=self.num_ent, size=1)
        )
        corrupted_triple = triple.copy()
        corrupted_triple[corrupted_ent] = F.normalize(
            self.ent_embeds(random_ent_idx), dim=-1
        )
        # print(corrupted_ent)
        # print(triple)
        # print(corrupted_triple)
        return corrupted_triple

    def get_tail(self, head, relation, k):
        """
        input: head + relation words, k nearest neighbors
        output: nearest tail word
        """
        with torch.no_grad():
            # convert to indices
            h = torch.LongTensor([self.kg_obj.ent2idx[head]])
            r = torch.LongTensor([self.kg_obj.rel2idx[relation]])
            # convert to embeddings
            h_embed = self.ent_embeds(h)
            r_embed = self.rel_embeds(r)
            # Add embeddings
            t_embed = h_embed + r_embed
            # return nearest neighbor
            # print(t_embed)
            dist = torch.zeros((self.num_ent, 1))
            for i in range(self.num_ent):
                dist[i, :] = torch.norm(
                    self.ent_embeds(torch.LongTensor([i])) - t_embed,
                    dim=1,
                    p=None,
                )
            # print(dist)
            knn = dist.topk(k, dim=0, largest=False)
            top_tails = []
            for i in knn.indices:
                top_tails.append(self.kg_obj.idx2ent[i.item()])
            print("kNN dist: {}, words: {}".format(knn.values, top_tails))

    @staticmethod
    def loss_fn(score):
        zero_tensor = torch.LongTensor([0])
        return torch.maximum(
            score, zero_tensor
        ).mean()  # average losses over batch

    def fit(self):

        # Create array of indices for the knowledge graph DataFrame.
        idx_df = self.kg_obj.knowledge_graph_df.copy(deep=True)
        idx_df["entity_id"] = idx_df["entity_id"].apply(
            lambda x: self.kg_obj.ent2idx[x]
        )
        idx_df["value"] = idx_df["value"].apply(
            lambda x: self.kg_obj.ent2idx[x]
        )
        idx_df["relation"] = idx_df["relation"].apply(
            lambda x: self.kg_obj.rel2idx[x]
        )
        idx_array = idx_df.values

        # set optimizer
        optimizer = optim.SGD(self.parameters(), lr=0.01, weight_decay=1e-4)

        for _ in range(self.n_epochs):
            for triple in tqdm(idx_array):
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.zero_grad()

                # Step 2. Get our inputs ready for the network, that is,
                # turn them into Tensors of word indices.
                input_triple = torch.LongTensor(triple).unsqueeze(0)
                # adding unsqueeze if batch size = 1

                # Step 3. Run our forward pass.
                scores = self.forward(input_triple)
                loss = self.loss_fn(scores)

                # Step 4. Compute the loss, gradients,
                # and update the parameters by calling optimizer.step()
                loss.backward()
                optimizer.step()

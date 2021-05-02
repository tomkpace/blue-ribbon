import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.stats
from sklearn.model_selection import KFold, StratifiedShuffleSplit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.notebook import tqdm
from knowledge_graph_generator import KnowledgeGraphGenerator


class TransE(nn.Module):
    """
    A model for TransE
    """

    def __init__(
        self,
        kg_obj,
        margin=1,
        embed_dim=100,
        n_random_sample=1000,
        lr=0.01,
        weight_decay=1e-4,
        n_epochs=10,
    ):
        super(TransE, self).__init__()

        self.kg_obj = kg_obj
        self.embed_dim = embed_dim  # embedding dimension
        self.num_ent = len(kg_obj.entities)  # number of entities
        self.num_rel = len(kg_obj.relations)  # number of relations
        self.num_val = len(kg_obj.values)  # number of values
        self.margin = margin  # margin
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False
        self.n_random_sample = n_random_sample
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs

        # Initialize model weights
        self.ent_embeds = nn.Embedding(self.num_ent, self.embed_dim)
        nn.init.xavier_uniform_(self.ent_embeds.weight.data)

        self.rel_embeds = nn.Embedding(self.num_rel, self.embed_dim)
        nn.init.xavier_uniform_(self.rel_embeds.weight.data)

        self.val_embeds = nn.Embedding(self.num_val, self.embed_dim)
        nn.init.xavier_uniform_(self.val_embeds.weight.data)

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
        r = self.rel_embeds(data[-1, 1])
        t = self.val_embeds(data[-1, 2])

        # Normalizing constraint (not sure if this is best way to apply this -
        # check out "max norm" prop in embeddings)
        triple["h"] = F.normalize(h, dim=-1)
        triple["t"] = F.normalize(t, dim=-1)
        triple["r"] = r  # according to paper, r is unnormalized.
        return triple

    def get_corrupted_triple(self, triple):
        """
        input: triple dict
        output: corrupted triple dict
        """
        # Need to randomly create the corrupted triplet by
        # replacing either head or tail with random entity
        corrupted_ent = np.random.choice(["h", "t"], 1, p=[0.5, 0.5]).item()
        corrupted_triple = triple.copy()
        if corrupted_ent == "h":
            random_ent_idx = torch.LongTensor(
                np.random.randint(low=0, high=self.num_ent, size=1)
            )
            corrupted_triple[corrupted_ent] = F.normalize(
                self.ent_embeds(random_ent_idx), dim=-1
            )
        elif corrupted_ent == "t":
            random_ent_idx = torch.LongTensor(
                np.random.randint(low=0, high=self.num_val, size=1)
            )
            corrupted_triple[corrupted_ent] = F.normalize(
                self.val_embeds(random_ent_idx), dim=-1
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
            dist = torch.zeros((self.num_val, 1))
            for i in range(self.num_val):
                dist[i, :] = torch.norm(
                    self.val_embeds(torch.LongTensor([i])) - t_embed,
                    dim=1,
                    p=None,
                )
            # print(dist)
            knn = dist.topk(k, dim=0, largest=False)
            top_tails = []
            for i in knn.indices:
                top_tails.append(self.kg_obj.idx2val[i.item()])
            print("kNN dist: {}, words: {}".format(knn.values, top_tails))

    @staticmethod
    def loss_fn(score):
        zero_tensor = torch.LongTensor([0])
        return torch.maximum(
            score, zero_tensor
        ).mean()  # average losses over batch

    def gen_training_array(self, training_df):
        # Create array of indices for the knowledge graph DataFrame.
        idx_df = training_df.copy(deep=True)
        idx_df["entity_id"] = idx_df["entity_id"].apply(
            lambda x: self.kg_obj.ent2idx[x]
        )
        idx_df["value"] = idx_df["value"].apply(
            lambda x: self.kg_obj.val2idx[x]
        )
        idx_df["relation"] = idx_df["relation"].apply(
            lambda x: self.kg_obj.rel2idx[x]
        )
        idx_array = idx_df.values
        return idx_array

    def gen_random_dist(self, input_array, return_dist=False):
        """"""
        random_dist = []
        for _ in range(self.n_random_sample):
            idx = np.random.randint(0, len(input_array))
            triple = self.idx2embeds(
                torch.LongTensor(input_array[idx]).unsqueeze(0)
            )
            corrupted_triple = self.get_corrupted_triple(triple)
            distance_tensor = self.get_distance(corrupted_triple)
            distance = distance_tensor.detach().numpy()[0]
            random_dist.append(distance)
        self.random_mu, self.random_sigma = norm.fit(random_dist)
        if return_dist:
            return random_dist

    def fit(self, training_array, perform_random_dist=True):
        # set optimizer
        optimizer = optim.SGD(self.parameters(), lr=0.01, weight_decay=1e-4)

        for _ in range(self.n_epochs):
            for triple in tqdm(training_array):
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
        if perform_random_dist:
            self.gen_random_dist(training_array)

    def evaluate(self, test_array, method="rank"):
        if method == "loss":
            loss = []
            for triple in test_array:
                self.zero_grad()
                input_triple = torch.LongTensor(triple).unsqueeze(0)
                scores = self.forward(input_triple)
                loss.append(self.loss_fn(scores).detach().numpy().item())
            return np.mean(loss)
        elif method == "rank":
            rank = []
            for test_triple in test_array:
                triple = test_triple.copy()
                entity_idx = triple[0]
                loss = []
                for test_entity_idx in range(self.num_ent):
                    self.zero_grad()
                    triple[0] = test_entity_idx
                    input_triple = torch.LongTensor(triple).unsqueeze(0)
                    scores = self.forward(input_triple)
                    loss.append(self.loss_fn(scores).detach().numpy().item())
                ranks = scipy.stats.rankdata(loss)
                rank.append(ranks[entity_idx])
            return np.mean(rank)
        else:
            raise NotImplementedError

    def get_global_distance(self, input_array):
        """"""
        dist = []
        for idx in range(len(input_array)):
            triple = self.idx2embeds(
                torch.LongTensor(input_array[idx]).unsqueeze(0)
            )
            distance_tensor = self.get_distance(triple)
            distance = distance_tensor.detach().numpy().item()
            dist.append(distance)
        return dist

    def gen_probability(self, distance_array):
        """"""
        probability = 1 - norm.cdf(
            distance_array, loc=self.random_mu, scale=self.random_sigma
        )
        return probability

    def array2kg_df(self, idx_array):
        """"""
        kg_df = pd.DataFrame(
            idx_array, columns=["entity_id", "relation", "value"]
        )
        kg_df["entity_id"] = kg_df["entity_id"].apply(
            lambda x: self.kg_obj.idx2ent[x]
        )
        kg_df["value"] = kg_df["value"].apply(lambda x: self.kg_obj.idx2val[x])
        kg_df["relation"] = kg_df["relation"].apply(
            lambda x: self.kg_obj.idx2rel[x]
        )
        return kg_df

    def gen_full_array(self, idx_array):
        """
        Generate a full array of all possible combinations of
        the indices for entity_id, relation and value.
        """
        full_array = np.array(
            np.meshgrid(
                np.unique(idx_array[:, 0]),
                np.unique(idx_array[:, 1]),
                np.unique(idx_array[:, 2]),
            )
        ).T.reshape(-1, 3)
        return full_array


class TransEFuser:
    """"""

    def __init__(self, n_splits=20, max_epochs=20, patience=5, **kwargs):
        self.n_splits = n_splits
        self.max_epochs = max_epochs
        self.patience = patience
        self.kwargs = kwargs

    def fuse(self, input_kg_df):
        kg_df = pd.DataFrame(
            columns=[
                "entity_id",
                "relation",
                "value",
                "distance",
                "probability",
            ]
        )
        for relation in input_kg_df["relation"].unique():
            df = input_kg_df.loc[
                input_kg_df["relation"] == relation
            ].reset_index(drop=True)
            kg = KnowledgeGraphGenerator(known_data_list=[df])
            trans = TransE(kg_obj=kg, n_epochs=1, **self.kwargs)
            training_array = trans.gen_training_array(df)
            entities = training_array[:, 0]
            strat = StratifiedShuffleSplit(n_splits=self.n_splits)
            kfold = KFold(self.n_splits)
            try:
                for i, j in strat.split(training_array, entities):
                    pass
                splitter = strat.split(training_array, entities)
            except Exception:
                print("Resorted to Kfold splitter.")
                splitter = kfold.split(training_array)
            j = 0
            for train_idx, test_idx in splitter:
                loss = np.inf
                stagnant_iterations = 0

                for _ in range(self.max_epochs):
                    if stagnant_iterations >= self.patience:
                        continue
                    trans.fit(training_array[train_idx])
                    new_loss = trans.evaluate(training_array[test_idx])
                    if new_loss >= loss:
                        stagnant_iterations += 1
                    else:
                        stagnant_iterations = 0
                    loss = new_loss
                    print(
                        f"relation {relation}, cv {j}, epoch {_}, loss {new_loss}"
                    )

                full_array = trans.gen_full_array(training_array[test_idx])
                new_df = trans.array2kg_df(full_array)
                new_df["distance"] = trans.get_global_distance(full_array)
                new_df["probability"] = trans.gen_probability(
                    new_df["distance"]
                )
                kg_df = pd.concat([kg_df, new_df]).reset_index(drop=True)
                j += 1

        return kg_df

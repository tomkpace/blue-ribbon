import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.stats
from sklearn.model_selection import KFold, StratifiedShuffleSplit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm.notebook import tqdm
from knowledge_graph_generator import KnowledgeGraphGenerator


# This is the line in the code
# training_array = trans.gen_training_array(df)

# The pytorch dataset classs
class MovieKG(Dataset):
    """Movie knowledge graph data."""

    def __init__(self, movie_data, kg_obj, transform=None):
        """
        Args:
            movie_data (pandas df): relational triples from movie database.
            knowledge_graph_obj (object): KG object created from
            'KnowledgeGraphExtractor'.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.kg_obj = kg_obj
        self.transform = transform
        idx_df = movie_data.copy(deep=True)
        # Create array of indices for the knowledge graph DataFrame.
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
        # save idx_array as data
        self.data = idx_array

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # returns the triple as indices, cuts off year and arc count
        X = self.data[idx][:3]

        if self.transform:
            X = self.transform(X)
        return X


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
        # n_random_sample is the number of samples to generate the
        # random distance distribution.
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

        h = self.ent_embeds(data[:, 0])
        r = self.rel_embeds(data[:, 1])
        t = self.val_embeds(data[:, 2])

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
        idx_df = training_df.copy()
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
        """
        Method that will generate a random distance distribution
        by randomly sampling corrupted triples with n_random_sample
        samples.  The random distribution is fit to a normal
        distribution and then populates self.random_mu and self.random_sigma
        in order to compare the distance of a triple in the knowledge
        graph to the distribution of randomly generated triples.
        """
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

    def fit(self, training_loader):
        # set optimizer
        optimizer = optim.SGD(self.parameters(), lr=0.01, weight_decay=1e-4)
        self.train()
        with torch.enable_grad():
            for _ in range(self.n_epochs):
                for batch in tqdm(training_loader):
                    # Step 1. Remember that Pytorch accumulates gradients.
                    # We need to clear them out before each instance
                    self.zero_grad()

                    # Step 2. Get our inputs ready for the network, that is,
                    # turn them into Tensors of word indices.
                    input_triple = torch.LongTensor(batch)

                    # Step 3. Run our forward pass.
                    scores = self.forward(input_triple)
                    loss = self.loss_fn(scores)

                    # Step 4. Compute the loss, gradients,
                    # and update the parameters by calling optimizer.step()
                    loss.backward()
                    optimizer.step()

    def evaluate(self, test_loader, method="loss"):
        """
        Method to characterize the 'accuracy' of self.parameters()
        at fitting the data in test_array using one of two methods:
        loss or rank.  The loss method will return the average loss and
        the rank method will return the average rank of the true
        entity in the triple.
        """
        if method == "loss":
            loss = []
            self.eval()  # Put model in eval mode.
            with torch.no_grad():
                for batch in tqdm(test_loader):
                    self.zero_grad()
                    input_triple = torch.LongTensor(batch)
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
        """
        Method that will generate a list of distances corresponding to
        each triple in input_array.
        """
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
        """
        Method that will generate an array of probabilities based on
        an input list of distances.  The probability that is generated
        is the cumulative probability of obtaining at a distance at
        most that large with a randomly generated triple.
        """
        probability = 1 - norm.cdf(
            distance_array, loc=self.random_mu, scale=self.random_sigma
        )
        return probability

    def array2kg_df(self, idx_array):
        """
        Method that will convert an array of indices (idx_array) into
        a DataFrame of entities, relations and values.
        """
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
    """
    Wrapper class that uses TransE to fuse a input knowledge
    graph DataFrame and generate a probabilistic knowledge
    graph DataFrame.
    """

    def __init__(
        self, kg_obj, n_splits=5, max_epochs=10, patience=5, **kwargs
    ):
        self.kg_obj = kg_obj
        self.n_splits = n_splits
        self.max_epochs = max_epochs
        self.patience = patience
        self.kwargs = kwargs

    def gen_data_loader(
        self, input_data, batch_size=128, shuffle=True, drop_last=True
    ):
        """
        Method to generate a PyTorch DataLoader based on an input MovieKG
        PyTorch Dataset object that is instantiated from input_data.
        """
        loader = DataLoader(
            MovieKG(input_data, self.kg_obj),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        return loader

    def fuse(self):
        """
        Method that fits and predicts the distances and probabilities for
        the triples in input_kg_df using a k = n_splits cross-validation
        approach.  The method will train the parameters in TransE for
        max_epochs or until no improvement has been oberved in the
        performance of TransE.evaluate() for 'patience' epochs.
        """
        kg_df = pd.DataFrame(
            columns=[
                "entity_id",
                "relation",
                "value",
                "distance",
                "probability",
            ]
        )
        trans = TransE(kg_obj=self.kg_obj, **self.kwargs)
        input_kg_df = self.kg_obj.knowledge_graph_df
        for relation in input_kg_df["relation"].unique():
            df = input_kg_df.loc[
                input_kg_df["relation"] == relation
            ].reset_index(drop=True)
            # kg = KnowledgeGraphGenerator(known_data_list=[df])
            # trans = TransE(kg_obj=self.kg_obj, **self.kwargs)
            # training_array = trans.gen_training_array(df)
            # entities = training_array[:, 0]
            # strat = StratifiedShuffleSplit(n_splits=self.n_splits)
            kfold = KFold(self.n_splits)
            """
            try:
                for i, j in strat.split(training_array, entities):
                    pass
                splitter = strat.split(training_array, entities)
            except Exception:
                print("Resorted to Kfold splitter.")
                splitter = kfold.split(training_array)
            j = 0
            """
            for train_idx, test_idx in kfold.split(df):
                loss = np.inf
                stagnant_iterations = 0

                # Instantiate the training DataLoader
                training_loader = self.gen_data_loader(df.iloc[train_idx])

                # Instantiate the test DataLoader
                test_loader = self.gen_data_loader(df.iloc[test_idx])

                # Gen random dist distribution for probability calculations.
                training_array = trans.gen_training_array(df.iloc[train_idx])
                trans.gen_random_dist(training_array)

                for _ in range(self.max_epochs):
                    if stagnant_iterations >= self.patience:
                        continue
                    trans.fit(training_loader)
                    new_loss = trans.evaluate(test_loader)
                    if new_loss >= loss:
                        stagnant_iterations += 1
                    else:
                        stagnant_iterations = 0
                    loss = new_loss
                    print(f"relation {relation}, epoch {_}, loss {new_loss}")

                idx_array = trans.gen_training_array(df.iloc[test_idx])
                full_array = trans.gen_full_array(idx_array)
                new_df = trans.array2kg_df(full_array)
                new_df["distance"] = trans.get_global_distance(full_array)
                new_df["probability"] = trans.gen_probability(
                    new_df["distance"]
                )
                kg_df = pd.concat([kg_df, new_df]).reset_index(drop=True)

        return kg_df

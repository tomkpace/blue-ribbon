import pandas as pd
from relation_extractor import RelationExtractor


class KnowledgeGraphGenerator:
    def __init__(self, input_data_list=None, known_data_list=None):
        self.input_data_list = input_data_list
        self.known_data_list = known_data_list
        self.knowledge_graph_df = self.generate_graph()
        self.entities = self.generate_entities()
        self.idx2ent = self.generate_idx2ent()
        self.ent2idx = self.generate_ent2idx()
        self.relations = self.generate_relations()
        self.idx2rel = self.generate_idx2rel()
        self.rel2idx = self.generate_rel2idx()

    def generate_graph(self):
        """
        Function that will generate a knowledge graph based
        on the input_data_list, which must have the form
        [DataFrame, DataFrame...], where each DataFrame must
        have columns labeled entity_id and text.  The relations
        will be extracted for each row using the RelationExtractor
        extract method using the text column.  Known knowledge graph
        triples can be provided in a list of DataFrames by using
        the optional known_data_list argument.  The known DataFrames
        should be provided with known relations such that each
        DataFrame has columns entity_id, relation and value.
        """
        columns = ["entity_id", "relation", "value"]
        kg_df = pd.DataFrame(columns=columns)
        # Extract relations from input_data_list and compile DataFrame.
        if self.input_data_list is not None:
            for df in self.input_data_list:
                for i in range(len(df)):
                    row = df.iloc[i]
                    extractor = RelationExtractor(
                        row["entity_id"], row["text"]
                    )
                    new_df = pd.DataFrame(extractor.relations, columns=columns)
                    kg_df = pd.concat([kg_df, new_df])
        # Combine the extracted relations DataFrame with known_data_list.
        if self.known_data_list is not None:
            for new_df in self.known_data_list:
                kg_df = pd.concat([kg_df, new_df])
        return kg_df

    def generate_entities(self):
        """
        Function that will return the union of the set of entities (entity_id)
        and values from the knowledge graph DataFrame.
        """
        entities = set(self.knowledge_graph_df["entity_id"].unique())
        values = set(self.knowledge_graph_df["value"].unique())
        entities = entities.union(values)
        return entities

    def generate_idx2ent(self):
        """
        Function that generates a dictionary that maps between the
        entity index and the entity.
        """
        idx2ent = pd.DataFrame(self.entities, columns=["entity"]).to_dict()[
            "entity"
        ]
        return idx2ent

    def generate_ent2idx(self):
        """
        Function that generates a dictionary that maps between the
        entity and the entity index.
        """
        ent2idx = {v: k for k, v in self.idx2ent.items()}
        return ent2idx

    def generate_relations(self):
        """
        Function that will return a set of the relations in the
        knowledge graph DataFrame.
        """
        relations = set(self.knowledge_graph_df["relation"].unique())
        return relations

    def generate_idx2rel(self):
        """
        Function that generates a dictionary that maps between the
        relation index and the relation.
        """
        idx2rel = pd.DataFrame(self.relations, columns=["entity"]).to_dict()[
            "entity"
        ]
        return idx2rel

    def generate_rel2idx(self):
        """
        Function that generates a dictionary that maps between the
        relation and the relation index.
        """
        rel2idx = {v: k for k, v in self.idx2rel.items()}
        return rel2idx

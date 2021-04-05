import pandas as pd
from relation_extractor import RelationExtractor


class KnowledgeGraphGenerator:
    def __init__(self, input_data_list=None, known_data_list=None):
        self.input_data_list = input_data_list
        self.known_data_list = known_data_list
        self.knowledge_graph_df = self.generate()

    def generate(self):
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

from ner import ner, process_ner_output


class RelationExtractor:
    def __init__(self, entity_id, text):
        self.entity_id = entity_id
        self.text = text
        self.relations = self.extract()

    def extract(self):
        """
        Function that extracts a set of triples from the
        Spacy text object in the form [(triple), (triple), ...]
        """
        triples = []
        triples += self.extract_ner_relation(self.entity_id, self.text)
        return triples

    @staticmethod
    def extract_ner_relation(entity_id, text):
        ner_output = ner(text)
        processed_ner_output = process_ner_output(ner_output)
        relations = []
        for r in processed_ner_output:
            relation = (entity_id, f"has{r[1]}Theme", r[0])
            relations.append(relation)
        return relations

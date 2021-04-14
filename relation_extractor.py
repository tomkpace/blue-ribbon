from nlp import nlp, ner, process_ner_output


def negation_detector(token):
    head = token.head
    try:
        while head.pos_ != "VERB":
            if head == head.head:
                return False
            head = head.head
        for token in head.children:
            if token.dep_ == "neg":
                return True
        return False
    except Exception:
        return False


def verb_extractor(token):
    head = token.head
    try:
        while head.pos_ != "VERB":
            if head == head.head:
                return None
            head = head.head
        return head.text
    except Exception:
        return None


class RelationExtractor:
    def __init__(self, entity_id, text):
        self.entity_id = entity_id
        self.text = text
        self.relations = self.extract()

    def extract(self):
        """
        Function that extracts a set of relations from the
        Spacy text object in the form [(entity_id, relation, value), ...]
        """
        relations = []
        relations += self.extract_ner_relations(self.entity_id, self.text)
        relations += self.extract_noun_relations(self.entity_id, self.text)
        relations += self.extract_ad_relations(self.entity_id, self.text)
        return self.relation_filter(relations)

    @staticmethod
    def extract_ner_relations(entity_id, text):
        ner_output = ner(text)
        processed_ner_output = process_ner_output(ner_output)
        relations = []
        for r in processed_ner_output:
            relation = (entity_id, "features", r[0])
            relations.append(relation)
        return relations

    @staticmethod
    def extract_noun_relations(entity_id, text):
        doc = nlp(text)
        relations = []
        for n in doc.noun_chunks:
            noun_chunk = []
            if not negation_detector(n.root):
                for t in n:
                    if not t.is_stop:
                        noun_chunk.append(t.text.lower())
                relations.append((entity_id, "has", " ".join(noun_chunk)))
        return relations

    @staticmethod
    def extract_ad_relations(entity_id, text):
        doc = nlp(text)
        relations = []
        for t in doc:
            if not negation_detector(t):
                if t.pos_ in ("ADJ"):
                    relations.append((entity_id, "is", t.text.lower()))
                if t.pos_ in ("ADV"):
                    verb = verb_extractor(t)
                    if verb is not None:
                        relations.append(
                            (entity_id, verb.lower(), t.text.lower())
                        )
        return relations

    @staticmethod
    def relation_filter(relations):
        filtered_relations = []
        for relation in relations:
            if relation[1] == "" or relation[2] == "":
                continue
            else:
                filtered_relations.append(relation)
        return filtered_relations

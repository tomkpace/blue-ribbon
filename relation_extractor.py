from nlp import nlp, ner, process_ner_output
from constants import ner_relations, removal_values


def find_composite_noun_chunk(noun_chunk):
    composite_noun_chunk_idx = set()
    composite_noun_chunk = []
    for n in noun_chunk.doc.noun_chunks:
        if n == noun_chunk:
            for t in n:
                composite_noun_chunk_idx.add(t.i)
        elif n.root.head.head == noun_chunk.root:
            for t in n:
                composite_noun_chunk_idx.add(t.i)
            composite_noun_chunk_idx.add(n.root.head.i)
    for t in noun_chunk.doc:
        if t.i in composite_noun_chunk_idx:
            composite_noun_chunk.append(t)
    return composite_noun_chunk


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


def noun_chunk_ent_detector(noun_chunk):
    for t in noun_chunk:
        if t.ent_type_ != "":
            return True
    return False


def stop_phrase_detector(text):
    for t in nlp(text):
        if t.is_stop:
            continue
        return False
    return True


def removal_word_detector(text, removal_values=removal_values):
    for t in nlp(text):
        if t.is_stop or t.text.lower() in removal_values:
            continue
        return False
    return True


def value_smoother(text):
    text = text.replace(" -", "-")
    text = text.replace("- ", "-")
    text = text.replace("' s", "'s")
    text = text.replace(" 's", "'s")
    if text.count('"') == 1:
        text = text.replace('"', "")
    while text[0] == " ":
        text = text[1:]
    return text


class RelationExtractor:
    def __init__(self, entity_id, text):
        self.entity_id = entity_id
        self.text = text
        self.doc = nlp(text)
        self.relations = self.extract()

    def extract(self):
        """
        Function that extracts a set of relations from the
        Spacy text object in the form [(entity_id, relation, value), ...]
        """
        relations = []
        relations += self.extract_ner_relations(self.entity_id, self.text)
        relations += self.extract_noun_relations(self.entity_id, self.doc)
        relations += self.extract_ad_relations(self.entity_id, self.doc)
        return self.relation_filter(relations)

    @staticmethod
    def extract_ner_relations(entity_id, text):
        ner_output = ner(text)
        processed_ner_output = process_ner_output(ner_output)
        relations = []
        for r in processed_ner_output:
            relation = (entity_id, ner_relations[r[1]], r[0])
            relations.append(relation)
        return relations

    @staticmethod
    def extract_noun_relations(entity_id, doc):
        relations = []
        for n in doc.noun_chunks:
            noun_chunk = []
            if not noun_chunk_ent_detector(n) and not negation_detector(
                n.root
            ):
                for t in find_composite_noun_chunk(n):
                    noun_chunk.append(t.text.lower())
                relations.append(
                    (entity_id, "features the theme", " ".join(noun_chunk))
                )
        return relations

    @staticmethod
    def extract_ad_relations(entity_id, doc):
        relations = []
        for t in doc:
            if not negation_detector(t):
                if t.pos_ in ("ADJ"):
                    relations.append((entity_id, "is", t.text.lower()))
        return relations

    @staticmethod
    def relation_filter(relations):
        filtered_relations = []
        for relation in relations:
            if relation[1] == "" or relation[2] == "":
                continue
            elif stop_phrase_detector(relation[2]):
                continue
            elif removal_word_detector(relation[2]):
                continue
            else:
                relation = (
                    relation[0],
                    relation[1],
                    value_smoother(relation[2]),
                )
                filtered_relations.append(relation)
        return filtered_relations

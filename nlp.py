import spacy
from transformers import pipeline

nlp = spacy.load("en_core_web_sm")
ner = pipeline("ner")


def process_ner_output(ner_output):
    """
    Function that processes the Hugging Face NER output into
    a list of tuples, [(entity: entity_type), ...]
    """

    def _combine_tokens(token_list):
        """
        Function that combines a list of tokens that may contain
        subwords (and ##) into a single phrase.
        """
        ent = ""
        for t in token_list:
            if "#" in t:
                ent = ent[: len(ent) - 1]
                ent += f"{t.replace('#', '')} "
            else:
                ent += f"{t} "
        if ent[len(ent) - 1] == " ":
            ent = ent[: len(ent) - 1]
        if ent[0] == " ":
            ent = ent[1:]
        return ent

    i0 = 0
    entity = []
    entity_type = set()
    entity_list = []
    entity_type_list = []
    for i in ner_output:
        if i["index"] > i0 + 1:
            if len(entity) > 0:
                entity_list.append(entity)
                entity_type_list.append(entity_type)
            entity = []
            entity_type = set()
        i0 = i["index"]
        entity.append(i["word"])
        entity_type.add(i["entity"])
    if len(entity) > 0:
        entity_list.append(entity)
        entity_type_list.append(entity_type)
    results = []
    for i, raw_ent in enumerate(entity_list):
        ent = _combine_tokens(raw_ent)
        raw_ent_type = next(iter(entity_type_list[i]))
        result = (ent, raw_ent_type.replace("I-", ""))
        results.append(result)
    return results

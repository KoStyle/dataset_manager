from sentence_transformers import SentenceTransformer

from att_generators.void_attr_gen import VoidAttGen
from constants import TYPE_LST


class BertAttGen(VoidAttGen):
    model = SentenceTransformer('bert-base-nli-mean-tokens')

    @staticmethod
    def gen_bert_vector():
        def get_bert_vector(text):
            sentences = [text]
            sentence_embeddings = BertAttGen.model.encode(sentences)
            vector = sentence_embeddings[0].tolist()
            return vector

        return BertAttGen('BERT_VECT', TYPE_LST, get_bert_vector)
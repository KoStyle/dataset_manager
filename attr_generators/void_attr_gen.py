from attr_generators.attr_funcs import *
from constants import TYPE_LST


class VoidAttGen:
    '''
    This class contains a generator following the att_generator ducktyping for the wordcount of a text
    '''

    def __init__(self, gen_id, gen_type, f):
        self.id = gen_id
        self.type = gen_type
        self.func = f

    def __str__(self):
        return self.get_attr_id()

    def __repr__(self):
        return self.get_attr_id()

    def get_attr_type(self):
        return self.type

    def get_attr_id(self):
        return self.id

    def get_attr(self, text):
        return AttrValue(self.type, self.func(text))

    @staticmethod
    def gen_wc():
        return VoidAttGen('WORD_COUNT', TYPE_NUM, get_wc)

    @staticmethod
    def gen_noun():
        return VoidAttGen('NOUN_COUNT', TYPE_NUM, get_noun_count)

    @staticmethod
    def gen_verb():
        return VoidAttGen('VERB_COUNT', TYPE_NUM, get_verb_count)

    @staticmethod
    def gen_adj():
        return VoidAttGen('ADJ_COUNT', TYPE_NUM, get_adjective_count)

    @staticmethod
    def gen_pron():
        return VoidAttGen('PRON_COUNT', TYPE_NUM, get_pronoun_count)

    @staticmethod
    def gen_det():
        return VoidAttGen('DET_COUNT', TYPE_NUM, get_determinant_count)

    @staticmethod
    def gen_sentences():
        return VoidAttGen('SENT_COUNT', TYPE_NUM, get_sentence_count)

    @staticmethod
    def gen_test_vector():
        return VoidAttGen('TEST_VECT', TYPE_LST, get_test_vector_attr)

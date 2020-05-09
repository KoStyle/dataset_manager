import string
import nltk
from att_generators.attr_funcs import *

from constants import TAGSET, TAG_NOUN, TAG_ADJ, TAG_DET, TAG_VERB, TAG_PRON


class VoidAttGen:
    '''
    This class contains a generator following the att_generator ducktyping for the wordcount of a text
    '''

    def __init__(self, gen_id, gen_type, f):
        self.id = gen_id
        self.type = gen_type
        self.func = f

    @staticmethod
    def gen_wc():
        return VoidAttGen('WORD_COUNT', 'INT', get_wc)

    @staticmethod
    def gen_noun():
        return VoidAttGen('NOUN_COUNT', 'INT', get_noun_count)

    @staticmethod
    def gen_verb():
        return VoidAttGen('VERB_COUNT', 'INT', get_verb_count)

    @staticmethod
    def gen_adj():
        return VoidAttGen('ADJ_COUNT', 'INT', get_adjective_count)

    @staticmethod
    def gen_pron():
        return VoidAttGen('PRON_COUNT', 'INT', get_pronoun_count)

    @staticmethod
    def gen_det():
        return VoidAttGen('DET_COUNT', 'INT', get_determinant_count)

    @staticmethod
    def gen_sentences():
        def get_sentence_count(text):
            clean_txt = text.replace('...', '') #We delete three consecutive points
            return len(clean_txt.split('.'))

        return VoidAttGen('SENT_COUNT', 'INT', get_sentence_count)

    def get_attr_type(self):
        return self.type

    def get_attr_id(self):
        return self.id

    def get_attr(self, text):
        return self.func(text)

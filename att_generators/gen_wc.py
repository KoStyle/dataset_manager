import string
import nltk

from constants import TAGSET


def gen_wc(text):
    if type(text) is not string or text == "":
        return -1
    else:
        return len(nltk.word_tokenize(text))


def get_tag_count(text, tag):
    tags = nltk.pos_tag(nltk.word_tokenize(text), tagset=TAGSET)

    count = 0
    for pair in tags:
        if pair[1] == tag:
            count += 1

    return count




class VoidAttGen:
    '''
    This class contains a generator following the att_generator ducktyping for the wordcount of a text
    '''

    def __init__(self, gen_id, gen_type, f):
        self.id = gen_id
        self.type = gen_type
        self.func = f

    def get_attr_type(self):
        return self.type

    def get_attr_id(self):
        return self.id

    def get_attr(self, text):
        return self.func(text)

import string
import nltk


class GenWC:
    '''
    This class contains a generator following the att_generator ducktyping for the wordcount of a text
    '''

    def __init__(self):
        self.id = "WORD_COUNT"

    def get_attr_type(self):
        return "int"

    def get_attr_id(self):
        return self.id

    def get_attr(self, text):
        if type(text) is not string or text == "":
            return -1
        else:
            return len(nltk.word_tokenize(text))

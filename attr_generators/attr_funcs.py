import nltk

from constants import TAG_PRON, TAG_VERB, TAG_ADJ, TAG_NOUN, TAGSET, TAG_DET, TYPE_NUM


class AttrValue:
    def __init__(self, attrtype, attrvalue):
        self.type = attrtype
        self.value = attrvalue


def get_wc(text):
    if type(text) is not str or text == "":
        return -1
    else:
        # We clean up the text from characters so the word count gets only proper stuff.
        for c in '.,:;-_?!\n':
            text = text.replace(c, ' ')
        return len(nltk.word_tokenize(text))


def __get_tag_count(text, tag):
    tags = nltk.pos_tag(nltk.word_tokenize(text), tagset=TAGSET)

    count = 0
    for pair in tags:
        if pair[1] == tag:
            count += 1

    return count


def get_noun_count(text):
    return __get_tag_count(text, TAG_NOUN)


def get_adjective_count(text):
    return __get_tag_count(text, TAG_ADJ)


def get_determinant_count(text):
    return __get_tag_count(text, TAG_DET)


def get_verb_count(text):
    return __get_tag_count(text, TAG_VERB)


def get_pronoun_count(text):
    return __get_tag_count(text, TAG_PRON)


def get_sentence_count(text):
    # clean_txt = text.replace('...', '')  # We delete three consecutive points
    sentence_count = 0
    sentences = text.split('.')
    for sent in sentences:
        if sent != '':
            sentence_count += 1

    return sentence_count


def get_test_vector_attr(text):
    test = [1, 2, 3, 420, 69, 666, 2666618]
    return test

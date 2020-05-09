import string

import nltk

from constants import TAG_PRON, TAG_VERB, TAG_ADJ, TAG_NOUN, TAGSET, TAG_DET


def get_wc(text):
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


def get_noun_count(text):
    return get_tag_count(text, TAG_NOUN)


def get_adjective_count(text):
    return get_tag_count(text, TAG_ADJ)


def get_determinant_count(text):
    return get_tag_count(text, TAG_DET)


def get_verb_count(text):
    return get_tag_count(text, TAG_VERB)


def get_pronoun_count(text):
    return get_tag_count(text, TAG_PRON)

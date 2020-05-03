import nltk

from constants import REVSET
from io_management import *

RUTA_BASE = 'ficheros_entrada/'

if __name__ == "__main__":
    # entries_socal_app = read_partial_set(RUTA_BASE + 'result-APP-SOCAL.txt')
    # entries_svr_app = read_partial_set(RUTA_BASE + 'result-APP-SVR62.txt')
    # entries_comments = read_partial_set(RUTA_BASE + 'revs_app.txt', REVSET)
    #
    # sorted_by_class = split_set_by_class(join_result_sets(entries_socal_app, entries_svr_app))
    # cases = join_partial_set_entries(entries_comments, sorted_by_class[TAG_SOCAL], sorted_by_class[TAG_SVR])
    print(len(sorted_by_class[TAG_SVR]))
    # print(len(sorted_by_class[TAG_SOCAL]))

    #nltk.download('averaged_perceptron_tagger')
    #nltk.download('punkt')
    tokensnltk.word_tokenize("My name is Phillip Douglass, bitch.")
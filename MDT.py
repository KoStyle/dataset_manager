import nltk

from constants import REVSET
from io_management import *

RUTA_BASE = 'ficheros_entrada/'

if __name__ == "__main__":
    entries_socal_app = read_partial_set(RUTA_BASE + 'result-IMDB-SOCAL.txt')
    entries_svr_app = read_partial_set(RUTA_BASE + 'result-IMDB-SVR62.txt')
    entries_comments = read_partial_set(RUTA_BASE + 'revs_imdb.txt', REVSET)
    complete_results = assing_class(join_result_sets(entries_socal_app, entries_svr_app))
    user_cases = join_partial_set_entries(complete_results)
    print(len(user_cases))
# print(len(sorted_by_class[TAG_SOCAL]))

# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# tokens = nltk.word_tokenize("My name is Phillip Douglass, bitch.")

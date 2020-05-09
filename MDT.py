import nltk
import timeit
from constants import REVSET
from io_management import *
from util import chronometer
from util import print_chrono

RUTA_BASE = 'ficheros_entrada/'


# TODO: una integración de textos de un UserCase
# TODO: Crear un fichero de set nuevo de user cases, una sola linea de texto, usuario, maeps o clase directamente, y review texto (el texto a parte quiza)
# TODO: usar el nuevo set para attribute_management y seguir adelante como planeado





# if __name__ == "__main__":
#     entries_socal_app = read_partial_set(RUTA_BASE + 'result-IMDB-SOCAL.txt')
#     entries_svr_app = read_partial_set(RUTA_BASE + 'result-IMDB-SVR62.txt')
#     entries_comments = read_partial_set(RUTA_BASE + 'revs_imdb.txt', REVSET)
#     complete_results = assing_class(join_result_sets(entries_socal_app, entries_svr_app))
#     user_cases = join_partial_set_entries(complete_results)
#     # print(len(user_cases))
#
#     @chronometer
#     def test():
#         for tupla_user in user_cases.items():
#             user_case = tupla_user[1]
#             user_case.get_maep_socal()
#
#     test()
#     print_chrono()
#
#     for tupla in user_cases.items():
#         user = tupla[1]
#
#         print("Maeps %s %s" % (user.get_maep_socal(), user.get_maep_svr()))

# print(len(sorted_by_class[TAG_SOCAL]))

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('taggers/universal_tagset')
except LookupError:
    nltk.download('universal_tagset')

# nltk.download('averaged_perceptron_tagger')
# nltk.download('universal_tagset')
tokens = nltk.word_tokenize("My name is Phillip Douglass, the most beautilful in this amazing world, bitch.")

tagged = nltk.pos_tag(tokens, tagset='universal')

print(tagged)
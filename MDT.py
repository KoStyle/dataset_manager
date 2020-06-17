import nltk
import timeit
import sqlite3

import user_case
from attribute_management import generate_attributes
from constants import REVSET
from io_management import read_partial_set, assing_class, join_result_sets, join_partial_set_entries, \
    create_database_schema

from util import chronometer
from util import print_chrono

from att_generators.void_attr_gen import VoidAttGen

RUTA_BASE = 'ficheros_entrada/'


# TODO: Crear un fichero de set nuevo de user cases, una sola linea de texto, usuario, maeps o clase directamente, y review texto (el texto a parte quiza)
# TODO: Base de datos sql para guardar las combinaciones de usertexts, leer de ahi.

def setup_nltk():
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


def generate_genlist():
    list_generators = [VoidAttGen.gen_verb(), VoidAttGen.gen_det(), VoidAttGen.gen_pron(), VoidAttGen.gen_sentences(),
                       VoidAttGen.gen_wc(), VoidAttGen.gen_noun(), VoidAttGen.gen_adj()]
    return list_generators


def load_usercase_set_IMBD():
    entries_socal_app = read_partial_set(RUTA_BASE + 'result-IMDB-SOCAL.txt')
    entries_svr_app = read_partial_set(RUTA_BASE + 'result-IMDB-SVR62.txt')
    entries_comments = read_partial_set(RUTA_BASE + 'revs_imdb.txt', REVSET)
    complete_results = assing_class(join_result_sets(entries_socal_app, entries_svr_app))
    return join_partial_set_entries(complete_results, entries_comments, "IMBD")


if __name__ == "__main__":
    create_database_schema()

    user_cases = load_usercase_set_IMBD()
    uc = list(user_cases.items())[0][1]

    conn = sqlite3.Connection('example.db')

    for x in range(20):
        uc.get_text(x)
        uc.db_log_instance(conn)

    instances = uc.db_list_instances(conn)
    uc.db_load_instance(conn, 6)
    print(instances)

    # print(len(user_cases))

    # @chronometer
    # def test():
    #     user_case = list(user_cases.items())[0][1]
    #     texto = user_case.get_text()
    #     # print("Longitud del texto: %d" % len(texto))
    #     list_generators = []
    #     gen_tmp = VoidAttGen.gen_wc()
    #     gen_noun = VoidAttGen.gen_noun()
    #     gen_adj = VoidAttGen.gen_adj()
    #
    #     list_generators.append(gen_tmp)
    #     list_generators.append(gen_noun)
    #     list_generators.append(gen_adj)
    #
    #     dict_temporal = {user_case.get_id(): user_case}
    #     # generate_attributes(dict_temporal, entries_comments, list_generators)
    #
    #     print(user_case.attributes)

    # test()
    # print_chrono()

# print(len(sorted_by_class[TAG_SOCAL]))


# tokens = nltk.word_tokenize("My name is Phillip Douglass, the most beautilful in this amazing world, bitch.")
#
# tagged = nltk.pos_tag(tokens, tagset='universal')
#
# print(tagged)

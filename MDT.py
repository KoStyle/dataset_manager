import nltk
import timeit
import sqlite3

import user_case
from attribute_management import generate_attributes, attribute_generator_publisher, get_active_attr_generators
from constants import REVSET
from io_management import __read_partial_set, __assign_class, __join_result_sets, join_partial_set_entries, \
    create_database_schema, load_dataset_files_IMBD

from util import chronometer
from util import print_chrono

from att_generators.void_attr_gen import VoidAttGen

RUTA_BASE = 'ficheros_entrada/'


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


if __name__ == "__main__":
    create_database_schema()
    setup_nltk()

    user_cases = load_dataset_files_IMBD()
    uc = list(user_cases.items())[0][1]

    conn = sqlite3.Connection('example.db')

    attribute_generator_publisher(conn)
    list_active_generators = get_active_attr_generators(conn)

    for x in range(20):
        generate_attributes([uc], list_active_generators, 10)
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

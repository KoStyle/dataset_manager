import sqlite3

import nltk

from constants import DATASET_IMDB
from io_management import create_database_schema, load_dataset
from util import print_chrono

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


def generate_user_instances(conn: sqlite3.Connection, instance_redundancy=1, instance_size=1):
    user_cases = load_dataset(conn, DATASET_IMDB)
    j = 0
    max_j = len(user_cases)
    print("Generating user instances")
    print("Instances per user: %d" % instance_redundancy)
    print("User reviews per instance: %d" % instance_size)
    for key in user_cases:

        for i in range(instance_redundancy):
            user_cases[key].gen_instance(instance_size)
            user_cases[key].db_log_instance(conn)
        j += 1
        print("Generated instances of user %s. %d/%d of users completed" % (key, j, max_j))


if __name__ == "__main__":
    create_database_schema()
    setup_nltk()
    conn = sqlite3.connect("example.db")
    generate_user_instances(conn, instance_redundancy=3, instance_size=3)
    print_chrono()
    conn.close()
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

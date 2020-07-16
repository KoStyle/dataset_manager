import sqlite3

import nltk
from sentence_transformers import SentenceTransformer

from attribute_management import get_active_attr_generators, attribute_generator_publisher, generate_attributes
from constants import DATASET_IMDB
from io_management import create_database_schema, load_dataset, load_all_db_instances
from util import print_chrono, chrono, get_chrono

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


def generate_user_instances(conn: sqlite3.Connection, dataset, instance_redundancy=1, instance_size=1):
    user_cases = load_dataset(conn, dataset)
    j = 0
    max_j = len(user_cases)
    attribute_generator_publisher(conn) #we create the available attrgenerators in the db
    print("Generating user instances")
    print("Instances per user: %d" % instance_redundancy)
    print("User reviews per instance: %d" % instance_size)
    for key in user_cases:

        for i in range(instance_redundancy):
            user_cases[key].gen_instance(instance_size)
            user_cases[key].db_log_instance(conn)
        j += 1
        print("Generated instances of user %s. %d/%d of users completed" % (key, j, max_j))


def generate_intances_attributes(conn : sqlite3.Connection, dataset):
    attribute_generator_publisher(conn)  #just in case, but we will not get any active one though
    list_active_generators = get_active_attr_generators(conn)

    user_instances = load_all_db_instances(conn, dataset)

    #TODO test speeds with different commit strategies (a lot of small ones or a big chunkus)
    generate_attributes(user_instances, list_active_generators)
    for instance in user_instances:
        instance.db_log_instance(conn)

@chrono
def test_bert_sentence():
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentences = ["This is a test sentence, to see if I'll drown in a river bank, or if I'll work in a bank. Also I want to see what does this return"]
    sentence_embeddings = model.encode(sentences)
    print("Ya")

if __name__ == "__main__":

    test_bert_sentence()
    get_chrono(test_bert_sentence)


    # create_database_schema()
    # setup_nltk()
    # conn = sqlite3.connect("example.db")
    # #generate_user_instances(conn, DATASET_IMDB, instance_redundancy=3, instance_size=3)
    # #print_chrono()
    # generate_intances_attributes(conn, DATASET_IMDB)
    # conn.close()
    # # print(len(user_cases))

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

import math
import sqlite3

from sklearn import svm, tree
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPClassifier

from attribute_management import get_active_attr_generators, attribute_generator_publisher, generate_attributes
from constants import DATASET_IMDB
from io_management import create_database_schema, load_dataset, load_all_db_instances, read_dataset_from_setup

RUTA_BASE = 'ficheros_entrada/'


# def setup_nltk():
#     try:
#         nltk.data.find('tokenizers/punkt')
#     except LookupError:
#         nltk.download('punkt')
#
#     try:
#         nltk.data.find('taggers/averaged_perceptron_tagger')
#     except LookupError:
#         nltk.download('averaged_perceptron_tagger')
#
#     try:
#         nltk.data.find('taggers/universal_tagset')
#     except LookupError:
#         nltk.download('universal_tagset')


def generate_user_instances(conn: sqlite3.Connection, dataset, instance_redundancy=1, instance_perc=1):
    user_cases = load_dataset(conn, dataset)
    j = 0
    max_j = len(user_cases)
    attribute_generator_publisher(conn)  # we create the available attrgenerators in the db
    print("Generating user instances")
    print("Instances per user: %d" % instance_redundancy)
    print("User reviews percentage per instance: %d" % instance_perc)
    for key in user_cases:
        uc = user_cases[key]
        for i in range(instance_redundancy):
            num_instances = math.floor(instance_perc * len(uc.reviews))
            uc.gen_instance(num_instances)
            uc.db_log_instance(conn)
        j += 1
        print("Generated instances of user %s. %d/%d of users completed" % (key, j, max_j))


def generate_intances_attributes(conn: sqlite3.Connection, dataset):
    attribute_generator_publisher(conn)  # activated by default
    list_active_generators = get_active_attr_generators(conn)

    user_instances = load_all_db_instances(conn, dataset)
    # result= Parallel(n_jobs=12)(delayed(generate_attributes)([row], list_active_generators) for row in user_instances)
    generate_attributes(user_instances, list_active_generators, conn=conn)
    # for instance in user_instances:
    #     instance.db_log_instance(conn)


# @chrono
# def test_bert_sentence():
#     global model
#     sentences = ["This is a test sentence, to see if I'll drown in a river bank, or if I'll work in a bank. Also I want to see what does this return"]
#     sentence_embeddings = model.encode(sentences)
#     vector = sentence_embeddings[0].tolist()
#     print("Ya")
#     return vector

def measure_model(model, the_matrix, the_expected, label):
    matrix_classified = model.predict(the_matrix)
    correct = 0
    for i in range(len(the_expected)):
        classi = matrix_classified[i]
        if classi > 0.5:
            classi = 1
        else:
            classi = 0

        if the_expected[i] == classi:
            correct += 1

    print(label +" set acierto (correctos/totales):")
    print(float(correct) / len(the_expected))

def model_run(models, id_setup, trainperc, conn):
    print("Let's get them entries!")
    the_matrix_train, the_expected_train, the_matrix_test, the_expected_test, the_matrix, the_expected = read_dataset_from_setup(conn, id_setup, train_perc=trainperc)
    for model in models:
        print("Let's fitness " + str(model))
        model.fit(the_matrix_train, the_expected_train)
        measure_model(model, the_matrix_train, the_expected_train, "Train")
        measure_model(model, the_matrix_test, the_expected_test, "Test")
        measure_model(model, the_matrix, the_expected, "Complete")
        print()
        print("-------")
        print()

if __name__ == "__main__":
    # test_bert_sentence()
    # get_chrono(test_bert_sentence)


    create_database_schema()
    # setup_nltk()
    conn = sqlite3.connect("example.db")
    models = []
    # models.append(LogisticRegression(penalty="none", max_iter=80000))
    # # models.append(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(6, 3), random_state=1))
    # # models.append(svm.SVC())
    # # models.append(Ridge(max_iter=80000, solver='sag'))
    # # models.append(tree.DecisionTreeClassifier())
    # models.append(ExtraTreesClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0))
    # models.append(RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0))
    # #
    # model_run(models, 3, 0.5, conn)
    # model_run(models, 4, 0.5, conn)

   # generate_user_instances(conn, DATASET_IMDB, instance_redundancy=2, instance_perc=0.25)
   # generate_user_instances(conn, DATASET_IMDB, instance_redundancy=2, instance_perc=0.5)
   # generate_user_instances(conn, DATASET_IMDB, instance_redundancy=2, instance_perc=0.75)
    # generate_user_instances(conn, DATASET_IMDB, instance_redundancy=1, instance_perc=1.0)
    # print_chrono()
    generate_intances_attributes(conn, DATASET_IMDB)
    conn.close()

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

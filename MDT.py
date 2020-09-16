import math
import random
import sqlite3
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier

from att_generators.void_attr_gen import VoidAttGen, nltk
from attribute_management import get_active_attr_generators, attribute_generator_publisher, generate_attributes
from constants import SOCAL_REPRES, SVR_REPRES, DATASET_IMDB
from dataset_entry import get_entries_of, map_predicted_values
from io_management import create_database_schema, read_matrixes_from_setup, __get_expected_array, __datalist_to_datamatrix, read_dataset_from_setup
from io_management import load_dataset, load_all_db_instances
from user_case import UserCase
from util import chronometer



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


def anna_run_this_please(ids):
    procs = []
    for id in ids:
        procs.append(subprocess.Popen(['C:\\Users\\konom\\Downloads\\openjdk-14+36_windows-x64_bin\\jdk-14\\bin\\java', '-jar', 'ANNA.jar', 'example.db', str(id), '0']))
    for p in procs:
        p.wait()
    # subprocess.Popen(['C:\\Users\\konom\\Downloads\\openjdk-14+36_windows-x64_bin\\jdk-14\\bin\\java', '-jar', 'ANNA.jar', 'example.db', '1', '0'])
    # subprocess.Popen(['C:\\Users\\konom\\Downloads\\openjdk-14+36_windows-x64_bin\\jdk-14\\bin\\java', '-jar', 'ANNA.jar', 'example.db', '1', '0'])
    # subprocess.Popen(['C:\\Users\\konom\\Downloads\\openjdk-14+36_windows-x64_bin\\jdk-14\\bin\\java', '-jar', 'ANNA.jar', 'example.db', '2', '0'])
    # subprocess.Popen(['C:\\Users\\konom\\Downloads\\openjdk-14+36_windows-x64_bin\\jdk-14\\bin\\java', '-jar', 'ANNA.jar', 'example.db', '1', '0'])
    # subprocess.Popen(['C:\\Users\\konom\\Downloads\\openjdk-14+36_windows-x64_bin\\jdk-14\\bin\\java', '-jar', 'ANNA.jar', 'example.db', '2', '0']).wait()


def generate_intances_attributes(conn: sqlite3.Connection, dataset):
    attribute_generator_publisher(conn)  # activated by default
    list_active_generators = get_active_attr_generators(conn)
    user_instances = load_all_db_instances(conn, dataset)
    # result= Parallel(n_jobs=12)(delayed(generate_attributes)([row], list_active_generators) for row in user_instances)
    generate_attributes(user_instances, list_active_generators, conn=conn)  # logs instances on generation


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

    print(label + " set acierto (correctos/totales): {:.6f}".format(correct / len(the_expected)))
    return correct, matrix_classified


def convert_ds_to_mx(dsentries):
    the_matrix = __datalist_to_datamatrix(dsentries)
    the_expected = __get_expected_array(dsentries)
    return the_matrix, the_expected


class ModelRun():
    def __init__(self, model, id_setup, kfolds, dataset):
        self.model = model
        self.id_setup = id_setup
        self.kfolds = kfolds
        self.dataset = dataset
        self.precision = 0.
        self.classified_entries = []
        self.user_beans = []

    def log_result(self, conn: sqlite3.Connection):
        select_max_resid = "SELECT MAX(id_result) as max_id FROM NNRESULTS WHERE id_setup=?"
        insert_statement = "INSERT INTO NNRESULTS (id_setup, id_result, perc_success, config_summary) VALUES (?, ?, ?, ?)"
        insert_classif_statement = "INSERT INTO RESULT_CLASSIFICATIONS VALUES (?, ?, ?, ?, ?, ?)"
        c = conn.cursor()

        c.execute(select_max_resid, (self.id_setup,))
        id_result = c.fetchone()[0]
        if not id_result:
            id_result = 0
        else:
            id_result += 1

        c.execute(insert_statement, (self.id_setup, id_result, self.precision, str(self.model)))

        for bean in self.user_beans:
            bean: UserIRRBean
            c.execute(insert_classif_statement, (self.id_setup, id_result, bean.user_id, bean.IRR_socal_maep, bean.IRR_svr_maep, bean.IRR_adaptative_maep))

        c.close()
        conn.commit()

    def plot(self):
        list_of_points = []
        self.user_beans.sort(key=lambda x: x.IRR_svr_maep, reverse=True)
        for ub in self.user_beans:
            ub:UserIRRBean
            list_of_points.append([ub.IRR_socal_maep, ub.IRR_svr_maep, ub.IRR_adaptative_maep])
        df = pd.DataFrame(list_of_points, columns=["SOCAL", "SVR", "ADAPTIVE"])
        fig = df.plot().get_figure()
        fig.savefig('graf.pdf')


def model_run_but_better(models, conn):
    print("Let's play!")
    for m in models:
        m: ModelRun
        print("Reading the dataset with setup {}. Give me a second there...".format(m.id_setup))
        ds = read_dataset_from_setup(conn, m.id_setup, m.dataset)
        folds = stratify(ds, m.kfolds)
        print("Done! (yay!)")

        testfolds = []
        correct_classifications = 0
        for i in range(len(folds)):
            testset = folds[i]
            trainset = []
            for t in range(len(folds)):
                if t != i:
                    trainset = trainset + folds[t]

            train_mx, train_expected = convert_ds_to_mx(trainset)
            test_mx, test_expected = convert_ds_to_mx(testset)

            m.model.fit(train_mx, train_expected)

            new_matches, the_classified = measure_model(m.model, test_mx, test_expected, "Fold {}".format(i))
            correct_classifications += new_matches
            testset = map_predicted_values(testset, the_classified)
            testfolds += testset  # this is used to evaluate the performance of the original IRR algorithm with our modification.

        m.precision = correct_classifications / len(ds)
        m.classified_entries = testfolds
        print("Total model precision {}-folds: {:.6f}".format(m.kfolds, m.precision))


def model_run(models, id_setup, trainperc, conn):
    print("Let's get them entries!")
    the_matrix_train, the_expected_train, the_matrix_test, the_expected_test, the_matrix, the_expected = read_matrixes_from_setup(conn, id_setup, trainperc, DATASET_IMDB)
    for model in models:
        print("Let's fitness " + str(model))
        model.fit(the_matrix_train, the_expected_train)
        measure_model(model, the_matrix_train, the_expected_train, "Train")
        measure_model(model, the_matrix_test, the_expected_test, "Test")
        measure_model(model, the_matrix, the_expected, "Complete")
        print()
        print("-------")
        print()


def stratify(entries_list, k):
    folds = []
    positive_cases = get_entries_of(entries_list, 1)
    negative_cases = get_entries_of(entries_list, 0)

    posi_per_fold = math.floor(len(positive_cases) / k)
    posi_extra = len(positive_cases) % k

    nega_per_fold = math.floor(len(negative_cases) / k)
    nega_extra = len(negative_cases) % k

    random.shuffle(positive_cases)
    random.shuffle(negative_cases)

    for i in range(k):
        pperfold = 0
        if posi_extra > 0:
            pperfold = posi_per_fold + 1
            posi_extra += -1
        else:
            pperfold = posi_per_fold

        nperfold = 0
        if nega_extra > 0:
            nperfold = nega_per_fold + 1
            nega_extra += -1
        else:
            nperfold = nega_per_fold

        # Stratified fold comprised of (aproximately) the same ratio of positive and negative cases as the original set
        fold = positive_cases[:pperfold] + negative_cases[:nperfold]
        folds.append(fold)

        # We delete from the list the folded cases
        del positive_cases[:pperfold]
        del negative_cases[:nperfold]

    return folds


class UserIRRBean:
    def __init__(self, user_id, socal, svr, classified_instances):
        self.user_id = user_id
        self.IRR_socal_maep = socal
        self.IRR_svr_maep = svr
        # the average of class assigned (the closer to 1, positive class, the closer to 0, negative
        average_predicted_class = sum(x.predicted_class for x in classified_instances) / len(classified_instances)

        # If the average prediction si closer to socal, use the socal maep, else, the svr maep
        if abs(SOCAL_REPRES - average_predicted_class) < abs(SVR_REPRES - average_predicted_class):
            self.IRR_adaptative_maep = self.IRR_socal_maep
        else:
            self.IRR_adaptative_maep = self.IRR_svr_maep


def generate_user_instances_for_percentiles(conn, dataset, redundancy):
    generate_user_instances(conn, dataset, redundancy, instance_perc=0.5)
    generate_user_instances(conn, dataset, redundancy, instance_perc=0.25)
    generate_user_instances(conn, dataset, redundancy, instance_perc=0.75)
    generate_user_instances(conn, dataset, redundancy, instance_perc=1.0)


# DO NOT USE, YOU FOOL!
@chronometer
def test():
    user_cases = ""
    user_case = list(user_cases.items())[0][1]
    texto = user_case.get_text()
    # print("Longitud del texto: %d" % len(texto))
    list_generators = []
    gen_tmp = VoidAttGen.gen_wc()
    gen_noun = VoidAttGen.gen_noun()
    gen_adj = VoidAttGen.gen_adj()

    list_generators.append(gen_tmp)
    list_generators.append(gen_noun)
    list_generators.append(gen_adj)

    dict_temporal = {user_case.get_id(): user_case}
    # generate_attributes(dict_temporal, entries_comments, list_generators)

    print(user_case.attributes)


def generate_model_examples():
    models = []
    models.append(LogisticRegression(penalty="none", max_iter=80000))
    models.append(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(6, 3), random_state=1))
    models.append(svm.SVC())
    models.append(Ridge(max_iter=80000, solver='sag'))
    models.append(tree.DecisionTreeClassifier())
    models.append(ExtraTreesClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0))

if __name__ == "__main__":
    # test_bert_sentence()
    # get_chrono(test_bert_sentence)
    create_database_schema()
    # setup_nltk()
    conn = sqlite3.connect("example.db")
    models = []

    models.append(ModelRun(RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0), id_setup=4, kfolds=10, dataset=DATASET_IMDB))

    model_run_but_better(models, conn)

    # For every model, we get the maep-value for socal, svr, and the adaptative method (ours) which will be one or the other depending
    # on the average predicted class for that given user's instances
    for m in models:
        users = load_dataset(conn, m.dataset, True)
        user_irr_beans = []
        for key, u in users.items():
            u: UserCase
            user_predictions = [x for x in m.classified_entries if x.user_id == u.user_id]
            ub = UserIRRBean(u.user_id, u.maep_socal, u.maep_svr, user_predictions)
            user_irr_beans.append(ub)
        m.user_beans = user_irr_beans
        # m.log_result(conn)
        m.plot()
    # generate_intances_attributes(conn, DATASET_IMDB)
    conn.close()

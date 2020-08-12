import math
import sqlite3
from random import random

import numpy
from scipy.sparse import csr_matrix

from base_case import BaseCase
from constants import CLASS_SOCAL, CLASS_SVR, SEPARATOR, TAG_RID, TAG_SVR, TAG_SOCAL, TAG_CLASS, \
    TAG_UID, TAG_PID, TAG_UR, TAG_REVIEW, RUTA_BASE, MUSR_UID, MUSR_CLASS, DBT_MUSR, MUSR_DS, MUSR_MAEP_SVR, \
    MUSR_MAEP_SOCAL, DATASET_IMDB, DATASET_APP
from dataset_entry import DsEntry
from user_case import UserCase
from util import chronometer2


def read_dataset_from_setup(conn: sqlite3.Connection, id_setup, train_perc):
    select_setup_statement = "SELECT select_statement from NNSETUPS where id_setup=?"
    c = conn.cursor()

    try:
        c.execute(select_setup_statement, (id_setup,))
        select_cases = c.fetchone()[0]
        c.execute(select_cases)
        results = c.fetchall()
        datalist = []
        expected_outputs = []
        entry_list = []
        for result in results:
            values = result[2].split('@')
            values = [float(element) for element in values]
            datalist.append(values)
            entry_list.append(DsEntry(values, float(result[3])))  # new and exciting stuff!!!
            # expected_outputs.append([float(element) for element in result[3].split('@')])
            expected_outputs.append(float(result[3]))

        the_train, the_test = __split_dsentries(entry_list, train_perc)
        the_train_mx = __datalist_to_datamatrix(the_train)
        the_test_mx = __datalist_to_datamatrix(the_test)
        the_train_exp = __get_expected_array(the_train)
        the_test_exp = __get_expected_array(the_test)

        # the_matrix = __datalist_to_datamatrix(datalist)
        # # expectations = numpy.array([numpy.array(xi) for xi in expected_outputs])
        # expectations = numpy.array(expected_outputs)

        return the_train_mx, the_train_exp, the_test_mx, the_test_exp

    except sqlite3.OperationalError as e:
        print(e)
        # TODO Something something error


def __split_dsentries(dsentries, train_perc):
    positive_entries = []
    negative_entries = []
    train_entries = []
    test_entries = []
    train_cases_amount = math.floor(len(dsentries) * train_perc)
    test_cases_amount = len(dsentries) - train_cases_amount

    for entry in dsentries:
        entry: DsEntry
        if entry.output_value == 0:
            negative_entries.append(entry)
        elif entry.output_value == 1:
            positive_entries.append(entry)

    random.shuffle(positive_entries)
    random.shuffle(negative_entries)

    #We add positive and negative cases for the training set until we fill this bad boy up
    for i in range(math.floor(train_cases_amount/2)):
        train_entries.append(positive_entries[i % len(positive_entries)])
        train_entries.append(negative_entries[i % len(negative_entries)])

    #We delete cases that are used in test
    positive_entries = list(set(positive_entries) - set(train_entries))
    negative_entries = list(set(negative_entries) - set(train_entries))

    #we put what's left into testing (good enough I guess..)
    test_entries.append(positive_entries)
    test_entries.append(negative_entries)

    #we shuffle this bad booois
    random.shuffle(train_entries)
    random.shuffle(test_entries)

    return train_entries, test_entries

def __get_expected_array(dsentries):
    expected_list = []
    for entry in dsentries:
        entry: DsEntry
        expected_list.append(entry.output_value)
    return numpy.array(expected_list)

# converts a list of lists of floats (or a list of DsEntries) to a csr matrix
def __datalist_to_datamatrix(datalist):
    # Use of the list to create a sparse matrix readable by the Pandora models
    rows = []
    cols = []
    mxdata = []
    for u in range(len(datalist)):
        vect = datalist[u]
        if isinstance(vect, DsEntry):
            vect = vect.input_values
        for i in range(len(vect)):
            if vect[i] != 0:
                rows.append(u)
                cols.append(i)
                mxdata.append(vect[i])
    enter_the_matrix = csr_matrix((mxdata, (rows, cols)), shape=(len(datalist), len(datalist[0])))
    return enter_the_matrix


def __load_dataset_files_IMBD():
    entries_socal_app = __read_partial_set(RUTA_BASE + 'result-IMDB-SOCAL.txt')
    entries_svr_app = __read_partial_set(RUTA_BASE + 'result-IMDB-SVR62.txt')
    entries_comments = __read_partial_set(RUTA_BASE + 'revs_imdb.txt')
    complete_results = __assign_class(__join_result_sets(entries_socal_app, entries_svr_app))
    return join_partial_set_entries(complete_results, entries_comments, DATASET_IMDB)


def __load_dataset_files_APP():
    entries_socal_app = __read_partial_set(RUTA_BASE + 'result-APP-SOCAL.txt')
    entries_svr_app = __read_partial_set(RUTA_BASE + 'result-APP-SVR62.txt')
    entries_comments = __read_partial_set(RUTA_BASE + 'revs_APP.txt')
    complete_results = __assign_class(__join_result_sets(entries_socal_app, entries_svr_app))
    return join_partial_set_entries(complete_results, entries_comments, DATASET_APP)


def load_dataset_from_files(dataset):
    if dataset == DATASET_APP:
        return __load_dataset_files_APP()
    elif dataset == DATASET_IMDB:
        return __load_dataset_files_IMBD()


def load_dataset(conn: sqlite3.Connection, dataset):
    user_cases = load_dataset_from_db(conn, dataset)
    if not user_cases:
        user_cases = load_dataset_from_files(dataset)
        for key in user_cases:
            uc: UserCase = user_cases[key]
            uc.db_log_user(conn)
        conn.commit()
    return user_cases


def load_all_db_instances(conn: sqlite3.Connection, dataset):
    select_instances = "SELECT C.tid, C.uid, C.numrevs, C.revstr, U.dataset FROM CONCATS C INNER JOIN MUSR U ON C.uid=U.uid WHERE U.dataset=?"
    c = conn.cursor()
    c.execute(select_instances, (dataset,))
    results = c.fetchall()
    user_instances = []
    for qr in results:
        ui = UserCase(qr[1])
        ui.txt_instance_id = qr[0]
        ui.rev_text_amount = qr[2]
        ui.rev_text_concat = qr[3]
        ui.dataset = dataset
        ui.db_load_attr(conn)
        user_instances.append(ui)
    return user_instances


@chronometer2
def load_dataset_from_db(conn: sqlite3.Connection, dataset):
    select_user_headers = "SELECT %s, %s, %s, %s FROM %s WHERE %s = ?" % (
        MUSR_UID, MUSR_CLASS, MUSR_MAEP_SVR, MUSR_MAEP_SOCAL, DBT_MUSR, MUSR_DS)
    c = conn.cursor()
    try:
        c.execute(select_user_headers, (dataset,))
        results = c.fetchall()
        c.close()
        user_cases = {}
        for query_result in results:
            uc = UserCase(query_result[0])
            uc.dataset = dataset
            uc.irr_class = query_result[1]
            uc.db_load_reviews(conn)
            user_cases[uc.user_id] = uc
        if user_cases:
            return user_cases
        else:
            return None
    except sqlite3.OperationalError as e:
        print(e)
        return None


def __read_partial_set(file_name):
    '''
    This function reads a .csv file with the field descriptions in the first line and returns an array
    of dictionaries with the entries in the file
    :param file_name: The path to the csv file
    :return: array of dictionaries with all the entries in the file
    '''
    entries = {}

    f = open(file_name, "r", encoding='utf-8')

    buffer = f.readlines()
    f.close()
    data_header_tokens = buffer.pop(0).split(SEPARATOR)

    header_list = []

    for data_token in data_header_tokens:
        header_list.append(data_token.strip().lower())  # we create a list of header names, cleaned and lowered from
        # the first line of the file

    for line in buffer:
        values = {}
        tokens = line.split(SEPARATOR)
        index = 0
        for token in tokens:
            values[header_list[index]] = token.strip().lower()
            index += 1

        entries[values[TAG_RID]] = values

    return entries


def __assign_class(entries):
    '''
    This function splits a set of examples in accordance to their performance with diferent training methods
    (labeled TAG_SVR and TAG_SOCAL)
    :param entries: All mixed class entries
    :return: A dictionary with two elements, the first corresponding to the SVR instances, the second corresponding to
    the SOCAL instances. They can be accessed using the constants TAG_SVR and TAG_SOCAL as keys.
    '''

    for key in entries:
        entry = entries[key]
        if entry[TAG_SVR] < entry[TAG_SOCAL]:  # because the absolute error is smaller in SVR, assings svr class
            entry[TAG_CLASS] = CLASS_SVR
        else:
            entry[TAG_CLASS] = CLASS_SOCAL

    return entries


def __get_entry(entry_id, entry_dict, remove=False):
    '''
    This function searches for an entry in a given set that matches a given id
    :param entry_id: The id to search (int)
    :param entry_dict: A set of entries
    :param remove: A flag that deletes the found entry from the original list if enabled. False by default.
    :return: The entry in case the id exists, None if not found
    '''
    if not entry_dict:
        return None

    try:
        result = entry_dict[entry_id]

        if remove:  # We remove the key if requested
            del entry_dict[entry_id]

        return result
    except KeyError:
        return None


def __check_set_algorithm(result_set, tag):
    if tag in list(result_set.items())[0][1]:
        return True
    else:
        return False


def __join_result_sets(set1, set2):
    '''
    This method takes to sets with only one estimation and combines the in a set of cases with both estimations.
    This method is faster if both sets are sorted by TAG_RID
    :param set_socal: first set
    :param set_svr: second set
    :return: combined set
    '''

    if __check_set_algorithm(set1, TAG_SVR):
        set_svr = set1
    elif __check_set_algorithm(set2, TAG_SVR):
        set_svr = set2
    else:
        raise Exception("No SVR set in join")

    if __check_set_algorithm(set1, TAG_SOCAL):
        set_socal = set1
    elif __check_set_algorithm(set2, TAG_SOCAL):
        set_socal = set2
    else:
        raise Exception("No SOCAL set in join")

    # We continue if we have both sets

    for sockey in set_socal:
        socentry = set_socal[sockey]
        svr_partner = __get_entry(sockey, set_svr)
        if svr_partner:
            socentry[TAG_SVR] = svr_partner[
                TAG_SVR]  # We add the svr data to the socal entry, this function is destructive
        else:
            # TODO Tabla log para escribir estas cosas quizas
            print("id {} no encontrado en SVR, asisgnando 99".format(sockey))
            del set_socal[sockey]
    return set_socal


# This method joins two sets that are linked through the field id. We search for the same ID in both sets and join said
# entries to create a complete set
def join_partial_set_entries(set_results, set_comments=None, ds=None):
    '''
    This function creates a definitive learning set containing the information from the three initial ones.
    :param ds: String with the name of the dataset
    :param set_comments: Set with the actual text reviews
    :param set_results: The dictionary with all the reviews as stated in the input file
    :return: A list of "UserCase" objects containing all the initial information, i.e.: without calculated attributes
    '''

    users_dict = {}

    base_cases = []
    for entry_key in set_results:
        entry = set_results[entry_key]
        case = BaseCase()
        case.rev_id = entry[TAG_RID]
        case.user_id = entry[TAG_UID]
        case.product_id = entry[TAG_PID]
        if set_comments:
            case.review = set_comments[entry_key][TAG_REVIEW]  # Only exists in comment sets (we separated that)
        case.irr_socal = float(entry[TAG_SOCAL])
        case.irr_svr = float(entry[TAG_SVR])
        case.user_rating = float(entry[TAG_UR])

        # We add the review to the corresponding user
        if case.user_id not in users_dict:
            user_case = UserCase(case.user_id)
            user_case.dataset = ds
            users_dict[user_case.get_id()] = user_case
        else:
            user_case = users_dict[case.user_id]

        user_case.add_review(case)

        base_cases.append(case)

    return users_dict


# TODO change magic numbers for constants
def create_database_schema():
    conn = sqlite3.connect('example.db')

    c = conn.cursor()
    # c.execute("DROP TABLE texto_combi")

    try:

        try:
            c.execute("CREATE TABLE MUSR ("
                      "uid text , "
                      "dataset text,"
                      "class text, "
                      "maep_svr real,"
                      "maep_socal real,"
                      "PRIMARY KEY (uid,dataset))")
        except sqlite3.OperationalError as e:
            print(e)

        try:
            c.execute("CREATE TABLE MREVS ("
                      "rid text , "
                      "dataset text,"
                      "uid text, "
                      "pid text, "
                      "review text, "
                      "socal real, "
                      "svr real, "
                      "PRIMARY KEY (rid),"
                      "FOREIGN KEY (uid) REFERENCES MUSR (uid))")
        except sqlite3.OperationalError as e:
            print(e)

        try:
            c.execute("CREATE TABLE CONCATS ("
                      "tid INTEGER PRIMARY KEY, "
                      "uid text, "
                      "numrevs int, "
                      "revstr text,"
                      "FOREIGN KEY (uid) REFERENCES MUSR (uid))")
        except sqlite3.OperationalError as e:
            print(e)

        try:
            c.execute("CREATE TABLE MATTR ("
                      "aid text PRIMARY KEY, "
                      "desc text, "
                      "active BOOLEAN,"
                      "type text)")
        except sqlite3.OperationalError as e:
            print(e)

        try:
            c.execute("CREATE TABLE ATTGEN ("
                      "tid INTEGER, "
                      "aid text, "
                      "aseq INTEGER,"  # Sequential for complex attributes
                      "value text, "
                      "cdate date, "
                      "udate date, "
                      "version int, "
                      "PRIMARY KEY(tid,aid,aseq), "
                      "FOREIGN KEY (tid) REFERENCES CONCATS (tid), "
                      "FOREIGN KEY (aid) REFERENCES MATTR (aid))")
        except sqlite3.OperationalError as e:
            print(e)
    except sqlite3.Error as e:
        print(e.__class__)
        print(e)
    finally:
        c.close()
        conn.commit()
        conn.close()

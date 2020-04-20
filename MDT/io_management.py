from MDT.base_case import BaseCase
from MDT.constants import CLASS_NOCLASS, CLASS_SOCAL, CLASS_SVR

SEPARATOR = ";"

TAG_RID = "id"
TAG_UID = "uid"
TAG_PID = "pid"
TAG_SVR = "svr"
TAG_SOCAL = "socal"
TAG_TEXT = "comment"


# TODO Rework header reading to return it somehow
# TODO al leer los resultados, meter en TAG_SVR/TAG_SOCAL el error, no el rating. para no leer el rating real. El programa funciona con error de metodo
# TODO intetar crear una unica funcion de lectura (que distinga por nombre de fichero quizá)

def read_og_set_file(file_name):
    '''
    This function reads a .csv file with the field descriptions in the first line and returns an array
    of dictionaries with the entries in the file
    :param file_name: The path to the csv file
    :return: array of dictionaries with all the entries in the file
    '''
    entries = []
    f = open(file_name, "r")

    buffer = f.readlines()
    data_header = buffer.pop(0)
    data_header_tokens = data_header.split(SEPARATOR)

    header_list = []

    for data_token in data_header_tokens:
        header_list.append(data_token.strip().lower())

    for line in buffer:
        values = {}
        tokens = line.split(SEPARATOR)
        index = 0
        for token in tokens:
            values[header_list[index]] = token.strip().lower()
            index += 1
        entries.append(values)

    return entries


def split_set(entries):
    '''
    This function splits a set of examples in accordance to their performance with diferent training methods
    (labeled TAG_SVR and TAG_SOCAL)
    :param entries: All mixed class entries
    :return: A list with two sublists, the first corresponding to the SVR instances, the second corresponding to the
    SOCAL instances.
    '''
    set1 = []
    set2 = []
    sets = [set1, set2]
    for entry in entries:
        if entry[TAG_SVR] > entry[TAG_SOCAL]:
            set1.append(entry)
        else:
            set2.append(entry)

    return sets


def get_entry(entry_id, entry_set):
    '''
    This function searches for an entry in a given set that matches a given id
    :param entry_id: The id to search (int)
    :param entry_set: A set of entries
    :return: The entry in case the id exists, None if not found
    '''
    i = 0
    entry = entry_set[i]
    max_len = len(entry_set)
    while not entry[TAG_RID] == entry_id and i < max_len:
        i += 1
        entry = entry_set[i]

    if i == max_len:
        return None
    else:
        return entry


def join_result_sets(set_socal, set_svr):
    for socentry in set_socal:
        svr_partner = get_entry(socentry[TAG_RID], set_svr)
        socentry[TAG_SVR] = svr_partner[TAG_SVR]  # We add the svr data to the socal entry, this function is destructive
    return set_socal


# This method joins two sets that are linked through the field id. We search for the same ID in both sets and join said entries to create a complete set
def join_partial_set_entries(set_comments, set_results_socal, set_results_svr):
    base_cases = []
    for entry in set_comments:
        case = BaseCase()
        case.rev_id = entry[TAG_RID]
        case.user_id = entry[TAG_UID]
        case.product_id = entry[TAG_PID]
        case.review = entry[TAG_TEXT]

        # We assing a class to the entry by searching for it in one of the subsets. If not foud, something not good happened
        if get_entry(case.rev_id, set_results_socal) is not None:
            case.classification_class = CLASS_SOCAL
        elif get_entry(case.rev_id, set_results_svr) is not None:
            case.classification_class = CLASS_SVR
        else:
            case.classification_class = CLASS_NOCLASS

        base_cases.append(case)

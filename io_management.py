from base_case import BaseCase
from constants import CLASS_NOCLASS, CLASS_SOCAL, CLASS_SVR, RESSET

SEPARATOR = "\t"

TAG_RID = "rev_id"
TAG_UID = "user_id"
TAG_PID = "product_id"
TAG_UR = "user_rating"
TAG_SVR = "svr"
TAG_SOCAL = "socal"
TAG_REVIEW = "review"


def read_partial_set(file_name, mode=RESSET):
    '''
    This function reads a .csv file with the field descriptions in the first line and returns an array
    of dictionaries with the entries in the file
    :param file_name: The path to the csv file
    :param mode: The reading mode, by defaut "RESSET" (for RESult SET). The other option would be "REVSET" (REView SET)
    :return: array of dictionaries with all the entries in the file
    '''
    entries = []

    f = open(file_name, "r")

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

        if mode == RESSET:  # we update the socal or svr field with absolute error instead of rating
            if TAG_SOCAL in values:
                estimation_method = TAG_SOCAL
            elif TAG_SVR in values:
                estimation_method = TAG_SVR
            else:
                raise Exception('No estimation tag found')
            values[estimation_method] = abs(float(values[estimation_method]) - float(values[TAG_UR]))

        entries.append(values)

    return entries


def split_set_by_class(entries):
    '''
    This function splits a set of examples in accordance to their performance with diferent training methods
    (labeled TAG_SVR and TAG_SOCAL)
    :param entries: All mixed class entries
    :return: A dictionary with two elements, the first corresponding to the SVR instances, the second corresponding to
    the SOCAL instances. They can be accessed using the constants TAG_SVR and TAG_SOCAL as keys.
    '''
    set_svr = []
    set_socal = []
    sets = {TAG_SVR: set_svr, TAG_SOCAL: set_socal}
    for entry in entries:
        if entry[TAG_SVR] < entry[TAG_SOCAL]:  # because the absolute error is smaller in SVR, goes to setSVR
            set_svr.append(entry)
        else:
            set_socal.append(entry)

    return sets


def get_entry(entry_id, entry_set, remove=False):
    '''
    This function searches for an entry in a given set that matches a given id
    :param entry_id: The id to search (int)
    :param entry_set: A set of entries
    :param remove: A flag that deletes the found entry from the original list if enabled. False by default.
    :return: The entry in case the id exists, None if not found
    '''
    if not entry_set:
        return None

    i = 0
    entry = entry_set[i]
    max_len = len(entry_set)
    while not entry[TAG_RID] == entry_id and i < max_len - 1:
        i += 1
        entry = entry_set[i]

    if i == max_len - 1:
        return None
    else:
        if remove:
            entry_set.remove(entry)
        return entry


# TODO hacer el orden de los argumentos intercambiables como en la lectura de fichero (comprobando que clave existe en
#   arg1
def join_result_sets(set_socal, set_svr):
    '''
    This method takes to sets with only one estimation and combines the in a set of cases with both estimations.
    This method is faster if both sets are sorted by TAG_RID
    :param set_socal: first set
    :param set_svr: second set
    :return: combined set
    '''
    for socentry in set_socal:  # Remove flat to true to optimize search of partner speed (works best with sorted sets)
        svr_partner = get_entry(socentry[TAG_RID], set_svr, True)
        if svr_partner:
            socentry[TAG_SVR] = svr_partner[
                TAG_SVR]  # We add the svr data to the socal entry, this function is destructive
        else:
            print("id {} no encontrado en SVR, asisgnando 99".format(socentry[TAG_RID]))
            socentry[TAG_SVR] = 99
    return set_socal


# This method joins two sets that are linked through the field id. We search for the same ID in both sets and join said
# entries to create a complete set
def join_partial_set_entries(set_comments, set_results_socal, set_results_svr):
    '''
    This function creates a definitive learning set containing the information from the three initial ones.
    :param set_comments: The file from where we get the IDs for review, product, user and the review text itself.
    :param set_results_socal: The file from where we get the SOCAL absolute error of the case following the formula
    absolute_error = (user_ratin - socal_estimate)
    :param set_results_svr: The file from where we get the SVR absolute error of the case following the formula
    absolute_error = (user_ratin - svr_estimate)
    :return: A list of "BaseCase" objects containing all the initial information, i.e.: without calculated attributes
    '''
    base_cases = []
    for entry in set_comments:
        case = BaseCase()
        case.rev_id = entry[TAG_RID]
        case.user_id = entry[TAG_UID]
        case.product_id = entry[TAG_PID]
        case.review = entry[TAG_REVIEW]

        # We assing a class to the entry by searching for it in one of the subsets. If not foud, something not good
        # happened. Found entries are removed from result sets for optimization reasons
        if get_entry(case.rev_id, set_results_socal, True) is not None:
            case.classification_class = CLASS_SOCAL
        elif get_entry(case.rev_id, set_results_svr, True) is not None:
            case.classification_class = CLASS_SVR
        else:
            case.classification_class = CLASS_NOCLASS

        base_cases.append(case)
    return base_cases

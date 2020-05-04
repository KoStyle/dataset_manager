from base_case import BaseCase
from constants import CLASS_NOCLASS, CLASS_SOCAL, CLASS_SVR, RESSET
from user_case import UserCase

SEPARATOR = "\t"

TAG_RID = "rev_id"
TAG_UID = "user_id"
TAG_PID = "product_id"
TAG_UR = "user_rating"
TAG_SVR = "svr"
TAG_SOCAL = "socal"
TAG_REVIEW = "review"
TAG_CLASS = "class"


def read_partial_set(file_name, mode=RESSET):
    '''
    This function reads a .csv file with the field descriptions in the first line and returns an array
    of dictionaries with the entries in the file
    :param file_name: The path to the csv file
    :param mode: The reading mode, by defaut "RESSET" (for RESult SET). The other option would be "REVSET" (REView SET)
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


def assing_class(entries):
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


def get_entry(entry_id, entry_dict, remove=False):
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


def check_method_set(result_set, tag):
    if tag in list(result_set.items())[0][1]:
        return True
    else:
        return False


def join_result_sets(set1, set2):
    '''
    This method takes to sets with only one estimation and combines the in a set of cases with both estimations.
    This method is faster if both sets are sorted by TAG_RID
    :param set_socal: first set
    :param set_svr: second set
    :return: combined set
    '''

    if check_method_set(set1, TAG_SVR):
        set_svr = set1
    elif check_method_set(set2, TAG_SVR):
        set_svr = set2
    else:
        raise Exception("No SVR set in join")

    if check_method_set(set1, TAG_SOCAL):
        set_socal = set1
    elif check_method_set(set2, TAG_SOCAL):
        set_socal = set2
    else:
        raise Exception("No SOCAL set in join")

    # We continue if we have both sets

    for sockey in set_socal:
        socentry = set_socal[sockey]
        svr_partner = get_entry(sockey, set_svr)
        if svr_partner:
            socentry[TAG_SVR] = svr_partner[
                TAG_SVR]  # We add the svr data to the socal entry, this function is destructive
        else:
            # TODO volcar esto a un log que se sobreescriba
            print("id {} no encontrado en SVR, asisgnando 99".format(sockey))
            del set_socal[sockey]
    return set_socal


# This method joins two sets that are linked through the field id. We search for the same ID in both sets and join said
# entries to create a complete set
def join_partial_set_entries(set_results):
    '''
    This function creates a definitive learning set containing the information from the three initial ones.
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
        #case.review = entry[TAG_REVIEW]  #Only exists in comment sets (we separated that)
        case.irr_socal = float(entry[TAG_SOCAL])
        case.irr_svr = float(entry[TAG_SVR])
        case.user_rating = float(entry[TAG_UR])

        # We add the review to the corresponding user
        if case.user_id not in users_dict:
            user_case = UserCase(case.user_id)
            users_dict[user_case.get_id()] = user_case
        else:
            user_case = users_dict[case.user_id]

        user_case.add_review(case)

        base_cases.append(case)

    return users_dict




def generate_attributes(dict_cases, dict_text, list_attgenerators):
    if not dict_cases or not dict_text or not list_attgenerators:
        raise Exception("An empty parameter received")

    for case_keys in dict_cases:

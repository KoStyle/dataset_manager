def generate_attributes(dict_cases, dict_text, list_attgenerators):
    '''
    This function generates as many attributes for the text in a user case based on the generator objects passed in
    the arguments. These objects must ducktype the method "get_Attr_id()" and "get_Attr( text)"
    :param dict_cases:
    :param dict_text:
    :param list_attgenerators:
    :return:
    '''
    if not dict_cases or not dict_text or not list_attgenerators:
        raise Exception("An empty parameter received")

    for case_keys, case_data in dict_cases.items():
        for generator in list_attgenerators:
            case_data.add_attribute(generator.get_attr_id(), generator.get_attr(case_data.get_text(dict_text)))

    return

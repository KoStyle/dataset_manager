import sqlite3

from att_generators.void_attr_gen import VoidAttGen
from constants import DBT_MATTR

published_attributes = []

#TODO probar
def attribute_generator_publisher():
    # Here we add a line for every attribute generator we want available (whether we use it or not)
    published_attributes.append(VoidAttGen.gen_wc())
    published_attributes.append(VoidAttGen.gen_adj())
    published_attributes.append(VoidAttGen.gen_noun())
    published_attributes.append(VoidAttGen.gen_sentences())

    conn = sqlite3.Connection('example.db')
    __log_attributes(conn)
    conn.close()

#TODO probar
def get_active_attr_generators():
    active_generators = []
    select_attr = "SELECT aid FROM %s WHERE active = ?" % DBT_MATTR
    conn = sqlite3.Connection('example.db')
    c = conn.cursor()
    c.execute(select_attr, (True,))

    data = c.fetchall()
    if data is not None or len(data) > 0:
        for att in published_attributes:

            # We check if the attribute is in the list of active ones
            flag_active = False
            i = 0
            while not flag_active or i < len(data):
                if data[i]["aid"] == att.get_attr_id():
                    flag_active = True
                i += 1

            if flag_active:
                active_generators.append(att)

    c.close()
    conn.close()
    return active_generators

# TODO Probar
def __log_attributes(conn: sqlite3.Connection):
    insert_attr = "INSERT INTO %s VALUES(?, ?, ?, ?)" % DBT_MATTR
    uninserter_attr = []
    c = conn.cursor()
    for attr in published_attributes:
        attr: VoidAttGen
        try:
            c.execute(insert_attr, (attr.get_attr_id(), "", False, attr.get_attr_type()))
        except sqlite3.Error:
            uninserter_attr.append(attr)
    c.close()
    conn.commit()


def generate_attributes(list_user_cases, list_attgenerators, sample_size=-1):
    '''
    This function generates as many attributes for the text in a user case based on the generator objects passed in
    the arguments. These objects must ducktype the method "get_Attr_id()" and "get_Attr( text)"
    :param sample_size:
    :param list_user_cases:
    :param list_attgenerators:
    :return:
    '''
    if not list_user_cases or not list_attgenerators:
        raise Exception("An empty parameter received")

    for user_case in list_user_cases:
        for generator in list_attgenerators:
            user_case.add_attribute(generator.get_attr_id(), generator.get_attr(user_case.get_text(sample_size)))

    return

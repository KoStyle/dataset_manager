import sqlite3

from att_generators.void_attr_gen import VoidAttGen
from constants import DBT_MATTR
from user_case import UserCase

published_attributes = []


def attribute_generator_publisher(optionalConn: sqlite3.Connection = None):
    # Here we add a line for every attribute generator we want available (whether we use it or not)
    published_attributes.append(VoidAttGen.gen_wc())
    published_attributes.append(VoidAttGen.gen_adj())
    published_attributes.append(VoidAttGen.gen_noun())
    published_attributes.append(VoidAttGen.gen_sentences())

    if optionalConn:
        conn = optionalConn
    else:
        conn = sqlite3.Connection('example.db')
    print(__log_attribute_headers(conn))

    conn.commit()

    if not optionalConn:
        conn.close()


def get_active_attr_generators(optionalConn: sqlite3.Connection = None):
    active_generators = []
    select_attr = "SELECT aid FROM %s WHERE active = ?" % DBT_MATTR

    if optionalConn:
        conn = optionalConn
    else:
        conn = sqlite3.Connection('example.db')
    c = conn.cursor()
    c.execute(select_attr, (True,))

    data = c.fetchall()
    if data is not None and len(data) > 0:
        for att in published_attributes:

            # We check if the attribute is in the list of active ones
            flag_active = False
            i = 0
            while not flag_active and i < len(data):
                if data[i][0] == att.get_attr_id():
                    flag_active = True
                i += 1

            if flag_active:
                active_generators.append(att)

    c.close()
    if not optionalConn:
        conn.close()
    return active_generators


def __log_attribute_headers(conn: sqlite3.Connection):
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
    return uninserter_attr


def generate_attributes(list_user_instances, list_attgenerators):
    '''
    This function generates as many attributes for the text in a user case based on the generator objects passed in
    the arguments. These objects must ducktype the method "get_Attr_id()" and "get_Attr( text)"
    :param list_user_instances:
    :param list_attgenerators:
    :return:
    '''
    if not list_user_instances or not list_attgenerators:
        raise Exception("An empty parameter received")

    print("Calculating attributes:")
    print(list_attgenerators)
    print("")
    i = 1
    max_i = len(list_user_instances)
    for user_case in list_user_instances:
        user_case: UserCase
        print("Generating attributes for instance %d/%d" % (i, max_i))
        for generator in list_attgenerators:
            if not user_case.exists_attribute(generator.get_attr_id()):
                user_case.add_attribute(generator.get_attr_id(), generator.get_attr(user_case.get_instance_text()))
        i += 1

    return

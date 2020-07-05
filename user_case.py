from random import random
from random import sample
import sqlite3
from datetime import datetime

from att_generators.attr_funcs import AttrValue
from base_case import BaseCase
from constants import TAG_REVIEW, CONCATS_TID, DBT_CONCATS, DBT_ATTGEN, ATTGEN_TID, ATTGEN_AID, CONCATS_UID, \
    CONCATS_NUMRE, CONCATS_REVST, TAG_RID, TAG_PID, TAG_SOCAL, TAG_SVR, DBT_MREVS, DBT_MUSR, CLASS_NOCLASS, CLASS_SOCAL, \
    CLASS_SVR, TYPE_LST


class UserCase:

    def __init__(self, user_id):
        self.user_id = user_id
        self.dataset = None
        self.reviews = {}
        self.rev_text_concat = None
        self.rev_text_amount = 0
        self.maep_socal = None
        self.maep_svr = None
        self.attributes = {}
        self.txt_instance_id = -1

    def get_id(self):
        return self.user_id

    def get_rev(self, rev_id):
        return self.reviews[rev_id]

    def get_text(self, amount=-1):
        '''
        This method concatenates a random number of text reviews into a single string attribute of the case.
        :param amount: Amount of reviews to concat
        :return:
        '''
        if amount < 0:
            amount = len(list(self.reviews))

        self.rev_text_concat = ""
        self.rev_text_amount = amount
        self.txt_instance_id = -1  # we mark it as -1 because the to-be-created instance won't be in the db (probably)

        if amount == len(list(self.reviews)):
            sample_revs = self.reviews
        else:
            sample_revs = sample(list(self.reviews), amount)

        for rev_key in sample_revs:
            rev_text = self.reviews[rev_key].review
            self.rev_text_concat = self.rev_text_concat + " " + rev_text
        self.rev_text_concat = self.rev_text_concat.strip()

        return self.rev_text_concat

    def get_maep_socal(self):
        if not self.maep_socal:
            self.calculate_maep()
        return self.maep_socal

    def get_maep_svr(self):
        '''
        This function gets the existing value for the SVR method's MAEP. If it doesn't exists, it calculates both MAEPs
        (SOCAL and SVR)
        :return: Returns the SVR MAEP value of the UserCase
        '''
        if not self.maep_svr:
            self.calculate_maep()
        return self.maep_svr

    def calculate_maep(self):
        '''
        This method calculates the Mean Average Error in Pairing order (MAEP) for a user based on its reviews. For all posible pairings between
        its reviews, we check if the IRR method (SOCAL or SVR) orders them in the same way than the user rating. The value represents
        the number of incorrectly ordered pairs divided by the total number of pairs. It stores the values in the attributes
        of the UserCase.
        :return: Nothing
        '''
        review_tuples = list(self.reviews.items())
        i = 0
        list_size = len(review_tuples)
        pairs = ((list_size * (
                list_size + 1)) / 2) - list_size  # (((n+1)*n)/2) -n Sumatory from 1 to n minus N (no pairs with themselves)
        correct_socal_pairs = 0
        correct_svr_pairs = 0
        while i < list_size:
            sample = review_tuples[i][1]
            j = i + 1
            while j < list_size:
                pair = review_tuples[j][1]

                # TODO check speed differences with different if structure
                # check socal pair
                if sample.user_rating > pair.user_rating and sample.irr_socal > pair.irr_socal:
                    correct_socal_pairs += 1
                elif sample.user_rating <= pair.user_rating and sample.irr_socal <= pair.irr_socal:
                    correct_socal_pairs += 1

                # check svr pair
                if sample.user_rating > pair.user_rating and sample.irr_svr > pair.irr_svr:
                    correct_svr_pairs += 1
                elif sample.user_rating <= pair.user_rating and sample.irr_svr <= pair.irr_svr:
                    correct_svr_pairs += 1

                j += 1
            i += 1

        self.maep_socal = (pairs - correct_socal_pairs) / float(pairs)
        self.maep_svr = (pairs - correct_svr_pairs) / float(pairs)

    def add_review(self, review):
        if type(review) is BaseCase:
            self.reviews[review.rev_id] = review

    ##CAREFUL, attr_value can be a list in case of BERT or PANDORA
    def add_attribute(self, attr_id, attr_object):
        self.attributes[attr_id] = attr_object

    def db_log_instance(self, conn: sqlite3.Connection):
        select_max_tid = "SELECT MAX(%s) FROM %s" % (CONCATS_TID, DBT_CONCATS)
        select_max_amount = "SELECT MAX(%s) FROM %s" % (CONCATS_NUMRE, DBT_CONCATS)
        insert_header = "INSERT INTO %s VALUES (?, ?, ?, ?)" % DBT_CONCATS
        select_attrs = "SELECT %s FROM %s WHERE %s=?" % (ATTGEN_AID, DBT_ATTGEN, ATTGEN_TID)
        flag_insert = False
        c = conn.cursor()

        # We create the header if it didn't exist
        if self.txt_instance_id == -1:
            c.execute(select_max_tid)
            max_tid = c.fetchone()[0]
            c.execute(select_max_amount)
            max_amount = c.fetchone()[0]

            # We insert the instance with autoincremented TID unless it is a tid of all reviews and
            # we already have one of those (the calculation of attributes would be redundant, because the results would
            # be the same)
            if not (self.rev_text_amount == len(self.reviews) and self.rev_text_amount == max_amount):
                if max_tid:
                    self.txt_instance_id = max_tid + 1
                    flag_insert = True
                elif not max_tid:
                    self.txt_instance_id = 0
                    flag_insert = True

                if flag_insert:
                    c.execute(insert_header,
                              (self.txt_instance_id, self.user_id, self.rev_text_amount, self.rev_text_concat))

        # TODO test
        # We log the attributes once we are sure we have a header (only the ones not logged, based in the Attribute ID
        self.__db_log_attr(conn)
        c.close()
        conn.commit()

    def db_log_user(self, conn: sqlite3.Connection):
        insert_user = "INSERT INTO %s VALUES(?, ?, ?)" % DBT_MUSR
        if self.maep_svr is None or self.maep_socal is None:
            self.calculate_maep()
        user_class = CLASS_NOCLASS

        # We calculate the class to be expected in training for this user
        if self.maep_svr < self.maep_socal:
            user_class = CLASS_SVR
        elif self.maep_svr > self.maep_socal:  # If the maeps are equal, the class is NOCLASS
            user_class = CLASS_SOCAL

        c = conn.cursor()
        try:
            c.execute(insert_user, (self.user_id, self.dataset, user_class))
        except sqlite3.Error:
            c.close()
            return False

        uninserted_revs = self.__db_log_reviews(conn)
        print(uninserted_revs)
        c.close()
        return True

    def __db_log_reviews(self, conn: sqlite3.Connection):
        insert_reviews = "INSERT INTO %s VALUES(?, ?, ?, ?, ?, ?, ?)" % DBT_MREVS
        uninserted_list = []
        c = conn.cursor()
        for key, value in self.reviews.items():
            try:
                c.execute(insert_reviews, (
                    value.rev_id, self.dataset, self.user_id, value.product_id, value.review,
                    float(value.irr_socal),
                    float(value.irr_svr)))
            except sqlite3.Error:
                uninserted_list.append(key)
        return uninserted_list

    # TODO test
    def __db_log_attr(self, conn: sqlite3.Connection):
        insert_attributtes = "INSERT INTO %s VALUES(?, ?, ?, ?, ?, ?, ?)" % DBT_ATTGEN
        delete_failed_complex_attr = "DELETE FROM %s WHERE %s = ? AND %s = ?" % (DBT_ATTGEN, ATTGEN_TID, ATTGEN_AID)
        uninserted_list = []
        c = conn.cursor()
        for key, value in self.attributes.items():
            value: AttrValue
            if value.type == TYPE_LST:
                try:
                    for i in range(len(
                            value.value)):  # I know is a pain to read, it means when the attrValue object contains a list
                        c.execute(insert_attributtes,
                                  (self.txt_instance_id, key, i, value.value[i], datetime.now(), None, 1))
                except sqlite3.Error:
                    uninserted_list.append(key)
                    c.execute(delete_failed_complex_attr, (self.txt_instance_id,
                                                           key))  # In case only one component in the list fails to insert but the rest did (we wipe off the entire attr)
            else:
                try:
                    c.execute(insert_attributtes, (self.txt_instance_id, key, -1, value.value, datetime.now(), None, 1))
                except sqlite3.Error as e:
                    uninserted_list.append(key)
        c.close()
        return uninserted_list

    def db_load_user(self, conn: sqlite3.Connection):
        # TODO Implement
        return

    def __db_load_reviews(self, conn: sqlite3.Connection):
        # TODO Implement
        return

    def __db_load_attr(self, conn: sqlite3.Connection):
        # TODO Implement
        return

    def db_load_instance(self, conn: sqlite3.Connection, tid: int):
        # TODO also load the attributes calling the method
        select_header = "SELECT %s, %s, %s, %s FROM %s WHERE %s=?" % (
            CONCATS_TID, CONCATS_UID, CONCATS_NUMRE, CONCATS_REVST, DBT_CONCATS, CONCATS_TID)

        c = conn.cursor()
        c.execute(select_header, (tid,))
        data = c.fetchone()
        if data is not None:
            if self.user_id == data[1]:
                self.txt_instance_id = data[0]
                self.rev_text_concat = data[3]
                self.rev_text_amount = data[2]
            else:
                print("Mismatched user_id!")
        else:
            print("TID not found")

    def db_list_instances(self, conn: sqlite3.Connection):
        select_instances = "SELECT %s FROM %s WHERE %s=?" % (CONCATS_TID, DBT_CONCATS, CONCATS_UID)

        c = conn.cursor()
        c.execute(select_instances, (self.user_id,))
        instances = c.fetchall()
        list_instances = []
        for inst in list(instances):
            list_instances.append(inst[0])
        return list_instances

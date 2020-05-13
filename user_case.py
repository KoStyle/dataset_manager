from random import random
from random import sample
import sqlite3
import datetime
from base_case import BaseCase
from constants import TAG_REVIEW, CONCATS_TID, DBT_CONCATS, DBT_ATTGEN, ATTGEN_TID, ATTGEN_AID, CONCATS_UID, \
    CONCATS_NUMRE, CONCATS_REVST


class UserCase:

    def __init__(self, user_id):
        self.user_id = user_id
        self.reviews = {}
        self.rev_text_concat = None
        self.rev_text_amount = 0
        self.maep_socal = None
        self.maep_svr = None
        self.attributes = {}
        self.txt_instance = -1

    def get_id(self):
        return self.user_id

    def get_rev(self, rev_id):
        return self.reviews[rev_id]

    def get_text(self, amount=-1):
        '''
        This method concatenates a random number of text reviews into a single string attribute of the case. It stores the
        last result and only recalculates if the amount required is different.
        :param amount: Amount of reviews to concat
        :return:
        '''
        if amount < 0:
            amount = len(list(self.reviews))

        if not self.rev_text_concat or self.rev_text_amount != amount:
            self.rev_text_concat = ""
            self.rev_text_amount = amount

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
            self.calculate_maep_svr()
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

    def add_attribute(self, attr_id, attr_value):
        self.attributes[attr_id] = attr_value

    def db_log_instance(self, conn: sqlite3.Connection):
        select_max_tid = "SELECT MAX(%s) FROM %s" % (CONCATS_TID, DBT_CONCATS)
        insert_header = "INSERT INTO %s VALUES (?, ?, ?, ?)" % DBT_CONCATS
        insert_attr = "INSERT INTO %s VALUES (?, ?, ?, ?, ?, ?)" % DBT_ATTGEN
        select_attrs = "SELECT %s FROM %s WHERE %s=?" % (ATTGEN_AID, DBT_ATTGEN, ATTGEN_TID)
        flag_insert = False
        c = conn.cursor()

        if self.txt_instance == -1:
            c.execute(select_max_tid)
            max_tid = c.fetchone()[0]
            if max_tid and self.rev_text_amount != len(self.reviews):  # we just want 1 full instance (all reviews in)
                self.txt_instance = max_tid + 1
                flag_insert = True
            elif not max_tid:
                self.txt_instance = 1
                flag_insert = True

            if flag_insert:
                c.execute(insert_header, (self.txt_instance, self.user_id, self.rev_text_amount, self.rev_text_concat))
                c.execute(select_attrs, (self.txt_instance,))
                logged_attr = c.fetchall()

                for attkey, attdata in self.attributes.items():
                    if attkey not in logged_attr:
                        c.execute(insert_attr,
                                  (self.txt_instance, attkey, attdata[1], datetime.datetime.now(), None, 1))
        c.close()
        conn.commit()

    def db_list_instances(self, conn: sqlite3.Connection):
        select_instances = "SELECT %s FROM %s WHERE %s=?" % (CONCATS_TID, DBT_CONCATS, CONCATS_UID)

        c = conn.cursor()
        c.execute(select_instances, (self.user_id,))
        instances = c.fetchall()
        return instances

    def db_load_instance(self, conn: sqlite3.Connection, tid):
        select_header = "SELECT %s, %s, %s, %s FROM %s WHERE %s=?" % (
            CONCATS_TID, CONCATS_UID, CONCATS_NUMRE, CONCATS_REVST, DBT_CONCATS, CONCATS_TID)

        c = conn.cursor()
        c.execute(select_header, (tid,))

        if c.rowcount == 1:
            data = c.fetchone()
            if self.user_id == data[1]:
                self.txt_instance = data[0]
                self.rev_text_concat = data[3]
                self.rev_text_amount = data[2]
            else:
                print("Mismatched user_id!")
        else:
            print("TID not found")


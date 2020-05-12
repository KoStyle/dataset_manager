from random import random
from random import sample
import sqlite3
from base_case import BaseCase
from constants import TAG_REVIEW


class UserCase:

    def __init__(self, user_id):
        self.user_id = user_id
        self.reviews = {}
        self.rev_text_concat = None
        self.rev_text_amount = 0
        self.maep_socal = None
        self.maep_svr = None
        self.attributes = {}

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
        c = conn.cursor()
        # TODO: build sql inset statement for the concat header, and a loop for the attributes (if exists, update)

    def db_list_instances(self, conn: sqlite3.Connection):
        print("Unimplemented")
        # TODO: build sql select to list all the TIDs for this user

    def db_load_instance(self, conn: sqlite3.Connection, tid):
        print("Unimplemented")
        # TODO: Build sql select for a specific TID of this user and load its values (attributes included)

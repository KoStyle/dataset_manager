from random import random

from base_case import BaseCase


class UserCase:

    def __init__(self, user_id):
        self.user_id = user_id
        self.reviews = {}
        self.maep_socal = None
        self.maep_svr = None

    def get_id(self):
        return self.user_id

    def get_rev(self, rev_id):
        return self.reviews[rev_id]

    def get_maep_socal(self):
        if not self.maep_socal:
            self.calculate_maep()
        return self.maep_socal

    def get_maep_svr(self):
        if not self.maep_svr:
            self.calculate_maep_svr()
        return self.maep_svr

    def calculate_maep(self):
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

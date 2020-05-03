from random import random

from base_case import BaseCase


class UserCase:
    reviews = {}
    user_id = None
    maep_socal = None
    maep_svr = None

    def get_id(self):
        return self.user_id

    def get_rev(self, rev_id):
        return rev_id[rev_id]

    def get_maep_socal(self):
        if not self.maep_socal:
            self.calculate_maep_socal()
        return self.maep_socal

    def get_maep_svr(self):
        if not self.maep_svr:
            self.calculate_maep_svr()
        return self.maep_svr

    def calculate_maep_socal(self):
        # TODO implement
        self.maep_socal = random.rand()

    def calculate_maep_svr(self):
        # TODO implement
        self.maep_svr = random.rand()

    def __init__(self, user_id):
        self.user_id = user_id

    def add_review(self, review):
        if isinstance(BaseCase, review):
            self.reviews[review.rev_id] = review

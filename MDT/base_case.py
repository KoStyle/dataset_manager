from MDT.constants import CLASS_NOCLASS


class BaseCase:

    def __init__(self):
        self.rev_id = ''  # primary key
        self.user_id = ''
        self.product_id = ''
        self.review = ''
        self.classification_class = CLASS_NOCLASS
        self.attributes = {}

    def compute_attr(self, attributer):
        self.attributes[attributer.att_name] = attributer.eval_review(self.review)

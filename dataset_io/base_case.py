from constants import CLASS_NOCLASS


class BaseCase:
    '''
    This class is used to store all the information needed for a case in our set, including (but not limited to):
    the id of a review (that is a unique pair of user id and product id)
    the id of an user
    the id of the product being reviewed (in our case, it would be either an app or a film)
    the method that classified this review more accurately (SVR and SOCAL)
    the review itself
    the attributtes that can be extracted from the value of the review text itself
    '''

    def __init__(self):
        self.rev_id = ''  # primary key
        self.user_id = ''
        self.product_id = ''
        self.review = ''
        self.classification_class = CLASS_NOCLASS
        self.irr_socal = None
        self.irr_svr = None
        self.user_rating= None
        self.attributes = {}

    def compute_attr(self, attributer):
        self.attributes[attributer.att_id()] = attributer.eval_review(self.review)

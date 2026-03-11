# model for logistic growth
class Logistic:
    """
    Model for the logistic growth defined by parameters alpha and R.
    """    

    def __init__(self, alpha, R):
        self.aplha, self.R = alpha, float(R)

    def __call__(self, t, u):
        return self.alpha * u * (1-u/self.R)



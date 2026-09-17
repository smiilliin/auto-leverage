from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def create_model():

    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0),
    )

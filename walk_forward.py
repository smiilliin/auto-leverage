import numpy as np
import pandas as pd

from model import create_model
from strategy import make_strategy_returns
from backtest import calculate_performance


def walk_forward_predict(
    X,
    y,
    train_size=756,
    test_size=63,
):
    """
    과거 train 구간으로 학습하고
    바로 다음 test 구간을 예측하는 과정을 반복한다.

    train_size = 약 3년
    test_size  = 약 3개월
    """

    predictions = []

    start = train_size

    while start < len(X):

        train_start = start - train_size
        train_end = start

        test_end = min(
            start + test_size,
            len(X),
        )

        X_train = X.iloc[train_start:train_end]

        y_train = y.iloc[train_start:train_end]

        X_test = X.iloc[start:test_end]

        model = create_model()

        model.fit(
            X_train,
            y_train,
        )

        prediction = model.predict(X_test)

        predictions.append(
            pd.Series(
                prediction,
                index=X_test.index,
            )
        )

        start = test_end

    return pd.concat(predictions)

import requests
import pandas
import scipy
import numpy
import sys
from sklearn.linear_model import LinearRegression

TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    df = pandas.read_csv(TRAIN_DATA_URL,sep=',',header=None)
    df1=df.to_numpy()
    a=numpy.asarray(df1)
    X= a[0][1:]
    Y=a[1][1:]
    x1 = numpy.array([X])

    y1 = numpy.array([Y])
    #print(numpy.shape(x1))
    arr=numpy.array([area])
    #print(numpy.shape(arr))
    input1 = numpy.zeros((266,1))
    for i in range(266):
        input1[i,0]=x1[0,i]

    reg = LinearRegression().fit(input1,numpy.transpose(y1))
    input2 = numpy.zeros((24,1))
    for i in range(24):
        input2[i,0]=arr[0,i]

    fin =reg.predict(input2)
    finn = numpy.array(fin)
    #print(finn)
    return (finn)
    # YOUR IMPLEMENTATION HERE
    ...


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    #from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")

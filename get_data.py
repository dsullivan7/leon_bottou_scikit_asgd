import urllib.request
import os

label_url = "ftp://largescale.ml.tu-berlin.de/largescale/alpha/alpha_train.lab.bz2"
data_url = "ftp://largescale.ml.tu-berlin.de/largescale/alpha/alpha_train.dat.bz2"


def main():
    print("downloading labels...")
    # labels
    urllib.request.urlretrieve(label_url, os.path.join("data", "alpha_train.lab.bz2"))

    print("downloading data...")
    # data
    urllib.request.urlretrieve(data_url, os.path.join("data", "alpha_train.dat.bz2"))


if __name__ == "__main__":
    main()

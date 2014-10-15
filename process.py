import numpy as np
import pandas as p
import os


def main():
    print("converting alpha.txt to labels and plain data...")

    reader = p.read_csv(os.path.join("data", "alpha.txt"), sep=" ", iterator=True)

    try:
        os.remove(os.path.join("data", "data_alpha.txt"))
    except OSError:
        pass
    try:
        os.remove(os.path.join("data", "labels_alpha.txt"))
    except OSError:
        pass

    for i in range(200):
        print("processing chunk:", i + 1, "of 200")
        X = np.array(reader.get_chunk(2500))
        y = X[:, 0]
        X = X[:, 1:]
        for idx1, row in enumerate(X):
            for idx2, col in enumerate(row):
                X[idx1, idx2] = np.double(X[idx1, idx2].split(":")[1])

        print("     writing")
        fi = open(os.path.join("data", "data_alpha.txt"), "a")
        fil = open(os.path.join("data", "labels_alpha.txt"), "a")
        df = p.DataFrame(X)
        dfl = p.DataFrame(y.reshape(-1, 1))
        df.to_csv(fi, header=False, index=False)
        dfl.to_csv(fil, header=False, index=False)

    print("converting data to numpy binaries...")
    readerX = p.read_csv(os.path.join("data", "data_alpha.txt"),
                         iterator=True)
    readery = p.read_csv(os.path.join("data", "labels_alpha.txt"),
                         iterator=True)

    for i in range(10):
        print("converting chunk:", i + 1, "of 10")
        X = np.array(readerX.get_chunk(50000))
        y = np.array(readery.get_chunk(50000))
        np.save(os.path.join("data", "data_batch" + str(i) + ".npy"), X)
        np.save(os.path.join("data", "labels_batch" + str(i) + ".npy"), y)


if __name__ == "__main__":
    main()

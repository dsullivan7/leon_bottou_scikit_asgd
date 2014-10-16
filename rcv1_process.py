import numpy as np
import pandas as p
from scipy.sparse import csr_matrix
import os

chunk_num = 5000

def main():
    print("converting rcv1.txt to labels and plain data...")

    # reader = p.read_csv(os.path.join("data", "rcv1.train.txt"), sep=" ", iterator=True)

    try:
        os.remove(os.path.join("data", "data_rcv1.txt"))
    except OSError:
        pass
    try:
        os.remove(os.path.join("data", "labels_rcv1.txt"))
    except OSError:
        pass

    # maxrow = 0
    counter = 0

    chunk = np.zeros((chunk_num, 47153))
    y = []
    for line in open(os.path.join("data", "rcv1.test.txt"), "r"):
        if counter % 2500 == 0:
            print("at", counter)

        if counter % chunk_num == 0 and counter != 0:
            write_chunk(chunk, y, int(counter / chunk_num))
        X = line.split(" ")
        y.append(int(X[0]))
        for entry in X[1:]:
            entry_array = entry.split(":")
            chunk[counter % chunk_num, int(entry_array[0])] = entry_array[1]
        counter += 1

    # print("converting data to numpy binaries...")
    # readerX = p.read_csv(os.path.join("data", "data_alpha.txt"),
    #                      iterator=True)
    # readery = p.read_csv(os.path.join("data", "labels_alpha.txt"),
    #                      iterator=True)

    # for i in range(1):
    #     print("converting chunk:", i + 1, "of 10")
    #     X = csr_matrix(readerX.get_chunk(50000))
    #     y = csr_matrix(readery.get_chunk(50000))
    #     np.save(os.path.join("data", "data_batch" + str(i) + ".npy"), X)
    #     np.save(os.path.join("data", "labels_batch" + str(i) + ".npy"), y)


def write_chunk(chunk, y, i):
    print("writing")
    # fi = open(os.path.join("data", "data_alpha.txt"), "a")
    # fil = open(os.path.join("data", "labels_alpha.txt"), "a")
    # df = p.DataFrame(chunk)
    # dfl = p.DataFrame(np.array(y).reshape(-1, 1))
    # df.to_csv(fi, header=False, index=False)
    # dfl.to_csv(fil, header=False, index=False)
    X = csr_matrix(chunk)
    chunk[:] = np.zeros((chunk_num, 47153))
    print("converted to csr")
    y_copy = np.array(y).reshape(-1, 1)
    y[:] = []
    save_sparse_csr(os.path.join("data", "rcv1_test", "data_batch" + str(i)), X)
    np.save(os.path.join("data", "rcv1_test", "labels_batch" + str(i)), y_copy)


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

# def load_sparse_csr(filename):
#     loader = np.load(filename)
#     return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
#                          shape = loader['shape'])


if __name__ == "__main__":
    main()

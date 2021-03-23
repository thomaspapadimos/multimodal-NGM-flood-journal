import numpy as np
import math


def batch_gen(X, batch_size):
    n_batches = X.shape[0] / float(batch_size)
    n_batches = int(math.ceil(n_batches))
    end = int(X.shape[0] / float(batch_size)) * batch_size
    n = 0
    for i in range(0, n_batches):
        if i <= n_batches - 1:
            if len(X.shape) > 1:
                batch = X[i * batch_size:(i + 1) * batch_size, :]
                yield batch
            else:
                batch = X[i * batch_size:(i + 1) * batch_size]
                yield batch
        else:
            if len(X.shape) > 1:
                batch = X[end:, :]
                n += X[end:, :].shape[0]
                yield batch
            else:
                batch = X[end:]
                n += X[end:].shape[0]
                yield batch


if __name__ == "__main__":
  metrics = []

  for i in range(10):
    metrics.append([j for j in range(6)])


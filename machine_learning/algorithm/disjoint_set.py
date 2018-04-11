import numpy as np

__all__ = ['DisjointSet']


class DisjointSet:
    def __init__(self, n):
        self.p = np.arange(n)
        self.rank = np.zeros(n, dtype=np.int64)

    def find(self, i):
        s = self._find(i)
        self._compress(i, s)
        return s

    def union(self, i, j):
        s = self._find(i)
        t = self._find(j)
        if s == t:
            self._compress(i, s)
            self._compress(j, t)
            return False
        else:
            if self.rank[s] <= self.rank[t]:
                self.p[t] = s
                self.rank[s] += 1
                self._compress(i, s)
                self._compress(j, s)
            else:
                self.p[s] = t
                self.rank[t] += 1
                self._compress(i, t)
                self._compress(j, t)
            return True

    def get_labels(self):
        for i in range(len(self.p)):
            self.find(i)
        return self.p

    def _find(self, i):
        while i != self.p[i]:
            i = self.p[i]
        return i

    def _compress(self, i, s):
        j = self.p[i]
        while j != s:
            self.p[i] = s
            i, j = j, self.p[j]
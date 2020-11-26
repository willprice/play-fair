from itertools import combinations


def powerset(xs):
    for s in range(0, len(xs) + 1):
        yield from combinations(xs, s)

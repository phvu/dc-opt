from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from dcopt import optimizer


def simple():
    result = optimizer.minimize(f1=lambda x: 4 * np.power(x[0], 2),
                                f2=lambda x: 0.1 * np.power(x[0], 4) - np.sqrt(x[1]),
                                bounds=[(0, 0), (1, 2)],
                                constraints=dict(type='ineq', fun=lambda x: x[0] + x[1] - 1),
                                max_iterations=10)

    print('Best value {} found at {}'.format(result.values[-1], result.solutions[-1]))
    print(result)


def main():
    result = optimizer.minimize(f1=lambda x: np.power(x[0], 2),
                                f2=lambda x: np.power(x[1], 2),
                                bounds=[(-5, -5), (5, 2)],
                                constraints=dict(type='ineq', fun=lambda x: x[1] - x[0] - 2),
                                max_iterations=100)
    print('Best value {} found at {}'.format(result.values[-1], result.solutions[-1]))
    print(result)


if __name__ == '__main__':
    main()

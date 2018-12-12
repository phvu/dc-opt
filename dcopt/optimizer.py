from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


import itertools
import numpy as np
import scipy.optimize


class OptimizationResult(object):

    def __init__(self):
        self.values = []
        self.solutions = []
        self.lower_rects = []
        self.success = True
        self.message = ''

    def fail(self, message):
        self.success = False
        self.message = message

    def __str__(self):
        return """Success: {}
        values: {}
        solutions: {}
        lower_rects: {}
        message: {}
        """.format(self.success, self.values, self.solutions,
                   '\n'.join('{}'.format(r) for r in self.lower_rects), self.message)


class Rectangle(object):
    def __init__(self, bounds):
        self.bounds = bounds
        if not isinstance(self.bounds, scipy.optimize.Bounds):
            bounds_arr = np.asarray(bounds, dtype=np.float)
            self.bounds = scipy.optimize.Bounds(bounds_arr[0, :], bounds_arr[1, :])

        self.lower_bound = None
        self.upper_bound = None
        self.lower_solution = None
        self.upper_solution = None

    def __str__(self):
        return 'Rect({},{},{},{},{},{})'.format(self.bounds.lb, self.bounds.ub,
                                                self.lower_bound, self.upper_bound,
                                                self.lower_solution, self.upper_solution)

    def feasible_point(self):
        """Returns a numpy array of feasible point."""
        return np.asarray([next(x for x in (l, b, 0) if x is not None) for (l, b) in
                           zip(self.bounds.lb, self.bounds.ub)])

    def corners(self):
        """Return a generator of corners."""
        # needs to handle unbounded cases.
        return (np.asarray(x) for x in itertools.product(*zip(self.bounds.lb, self.bounds.ub)))

    def set_solution(self, lower_bound, upper_bound, lower_solution, upper_solution):
        """

        :param lower_bound:
        :param upper_bound:
        :param lower_solution:
        :param upper_solution:
        :return:
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.lower_solution = lower_solution
        self.upper_solution = upper_solution

    def split(self):
        """Split this rectangle along the maximum difference dimension."""
        split_dim = int(np.asscalar(np.argmax(np.abs(self.upper_solution - self.lower_solution))))
        new_bound_val = (self.upper_solution[split_dim] + self.lower_solution[split_dim]) / 2.
        assert self.bounds.lb[split_dim] <= new_bound_val <= self.bounds.ub[split_dim]

        r1_ub = self.bounds.ub.copy()
        r1_ub[split_dim] = new_bound_val
        r1 = Rectangle(scipy.optimize.Bounds(self.bounds.lb, r1_ub))

        r2_lb = self.bounds.lb.copy()
        r2_lb[split_dim] = new_bound_val
        r2 = Rectangle(scipy.optimize.Bounds(r2_lb, self.bounds.ub))
        return [r1, r2]


def _solve_rectangle(f1, f2, rect, constraints):
    # Maybe should explicitly pick the SLSQP solver.
    f1_result = scipy.optimize.minimize(f1, x0=rect.feasible_point(), bounds=rect.bounds, constraints=constraints,
                                        options=dict(maxiter=1000, ftol=1e-06))
    if not f1_result.success:
        raise ValueError('Failed to minimize f1: {}'.format(f1_result.message))
    f1_solution = f1_result.x
    f1_min_val = f1_result.fun

    f2_solution = None
    f2_max = -np.infty
    for c in rect.corners():
        if not any(x is None for x in c):
            c_val = f2(c)
            if c_val > f2_max:
                f2_solution = c
                f2_max = c_val
        else:
            print('Ignore corner: {}'.format(c))
    if f2_solution is None:
        # Set the solution to be infty
        raise ValueError('Failed to minimize f2 in the given rectangle')

    lower_bound = f1_min_val - f2_max  # beta, relaxed solution
    upper_bound = f1_min_val - f2(f1_solution)  # alpha

    rect.set_solution(lower_bound, upper_bound, f2_solution, f1_solution)
    return rect


def _get_min_rect(rectangles, rect_to_val):
    min_rect = rectangles[0]
    min_val = rect_to_val(min_rect)
    for rect in rectangles[1:]:
        rect_val = rect_to_val(rect)
        if rect_val < min_val:
            min_val = rect_val
            min_rect = rect
    return min_rect


def minimize(f1, f2, bounds=None, constraints=None, max_iterations=10):
    """
    Minimize a function f = f1 - f2 where f1 and f2 are convex.

    `bounds` needs to be as many as number of variables.

    :param f1:
    :param f2:
    :param bounds:
    :param constraints:
    :param max_iterations:
    :return:
    """
    result = OptimizationResult()
    try:
        rect = _solve_rectangle(f1, f2, rect=Rectangle(bounds), constraints=constraints)
    except ValueError as ex:
        result.fail('{}'.format(ex))
        return result

    rectangles = [rect]

    # Take the pessimistic solution (upper bound).
    best_val = rect.upper_bound
    best_solution = rect.upper_solution

    for k in range(max_iterations):
        # Pessimistic estimation: the true min has to be smaller than this.
        upper_rect = _get_min_rect(rectangles, lambda r: r.upper_bound)

        if upper_rect.upper_bound < best_val:
            best_val = upper_rect.upper_bound
            best_solution = upper_rect.upper_solution

        # Optimistic estimation: the true min has to be greater than this relaxed bound.
        lower_rect = _get_min_rect(rectangles, lambda r: r.lower_bound)

        # book-keeping
        result.values.append(best_val)
        result.solutions.append(best_solution)
        result.lower_rects.append(lower_rect)

        if lower_rect.lower_bound >= upper_rect.upper_bound:
            break
        else:
            try:
                split_rects = [_solve_rectangle(f1, f2, r, constraints) for r in lower_rect.split()]
            except ValueError as ex:
                result.fail('{}'.format(ex))
                return result

            rectangles = [r for r in rectangles if r != lower_rect and r.lower_bound <= best_val] + split_rects

    return result

import numpy as np
import random

def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.
    You might find numpy functions np.exp, np.sum, np.reshape,
    np.max, and numpy broadcasting useful for this task. (numpy
    broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

    You should also make sure that your code works for one
    dimensional inputs (treat the vector as a row), you might find
    it helpful for your later problems.

    You must implement the optimization in problem 1(a) of the
    written assignment!
    """
    # see http://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

    # A nice way to avoid this problem is by normalizing the inputs to be not too large
    # or too small, by observing that we can use an arbitrary constant C like:
    # np.exp(x + log(some_constant))

    # This will shift the inputs to a range close to zero,
    # assuming the inputs themselves are not too far from each other.
    # Crucially, it shifts them all to be negative (except the maximal
    # which turns into a zero). Negatives with large exponents "saturate" to
    # zero rather than infinity, so we have a better chance of avoiding NaNs.

    # Picking the constant
    # "A good choice is the maximum between all inputs, negated:"

    shiftx = x - np.max(x, axis=x.ndim-1, keepdims=True)
    exps = np.exp(shiftx)
    return exps / np.sum(exps, axis=x.ndim-1, keepdims=True)

def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print( "Running basic tests...")
    test1 = softmax(np.array([1,2]))
    print( test1)
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print('test2: ', test2)
    assert np.amax(np.fabs(test2 - np.array(
        [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6

    test3 = softmax(np.array([[-1001,-1002]]))
    print( test3)
    assert np.amax(np.fabs(test3 - np.array(
        [0.73105858, 0.26894142]))) <= 1e-6

    print( "You shoul1d verify these results!\n")


if __name__ == "__main__":
    test_softmax_basic()

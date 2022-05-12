# Projected Gradient Descent onto the probabilty simplex
import numpy as np

EPS = 0.0005
DEBUG = False

# p may already be on the simplex, but we don't know that yet
# We assume that p is non-negative, which should be true during the descent
#   since the value of mu is strictly positive, and an agent's contribution
#   to d(log(NSW(p,mu)))/dp_l is proportional to mu_i,l and inversely
#   proportional to the reward that agent receives from p, both of which are
#   non-negative. As such, sum(p) >= 1 should be true, but easy test in case
def project_to_simplex(p, iter=0):
    if DEBUG:
        print("  Projecting ", p)
    d = len(p)
    s = np.sum(p)
    # easy case, since p >= 0
    if s < 1:
        return p + ((1 - s) / d)
    # We have reached our target, or are accurate enough
    if (np.abs(s - 1) < EPS):
        # Close enough
        if DEBUG:
            print ("  s is initially ", s, ", stopping")
        return p / s
    # If we reach here, p >= 0 and s > 1
    is_zero = [False] * d    # We will keep track of 0s
    n_zeros = 0
    for i in range(d):
        if p[i] == 0:
            is_zero[i] = True
            n_zeros = n_zeros + 1
    # Now, we should decrement
    # After each loop, n_zeros increases by >= 1 and can be <= d
    while (np.abs(s - 1) > EPS and iter < 25):
        # offset to subtract from all non-zeros
        # We need to contribute (1 - s), split over the non-zeros
        offset = (1 - s) / (d - n_zeros)
        # Check each index
        for i in range(d):
            # if zero, do nothing
            if not is_zero[i]:
                # Update, and update 0 data if necessary
                p[i] = p[i] + offset
                if p[i] < 0:
                    p[i] = 0
                    is_zero[i] = True
                    n_zeros = n_zeros + 1
        s = np.sum(p)
        iter = iter + 1
        #loop
    if DEBUG:
        print("   Projected to ", p / s)
    return p / s




# func is the function to maximize  (R^d -> R)
# grad is the gradient function     (R^d -> R^d)
# proj is the projection function   (R^d -> R^d)
# d is the dimension
#
# maxIter is the maximum number of iterations we will allow
# delt is a threshold we use to mark slowing progress, and halt
# lookback is how far back we look for delt
# lr is the learning rate
# init_guess can be provided. This could be useful if we don't expect
#            large changes between multiple calls to proj_grad_asc
def proj_grad_asc(d, func, grad, proj=project_to_simplex, maxIter=50000, delt=0.00001, lookback=20, lr=0.05, init_guess=None):
    if init_guess is None:
        init_guess = proj(np.array([1/k] * k))
    last_guess = init_guess
    iter = 0
    iter_val = []
    while iter < maxIter:
        if DEBUG:
            print("Iter: ", iter, ", Last Guess = ", last_guess)
        # get the function value
        # this append puts us at iter_val[iter], starting at [0]
        iter_val.append(func(last_guess))
        # check for stalled progress
        if iter >= lookback:
            if np.max(iter_val[iter - lookback:]) - np.min(iter_val[iter-lookback:]) < delt:
                if DEBUG:
                    print("PGD RAN FOR ", iter, " ITERATIONS")
                return last_guess
        # otherwise, we continue
        if DEBUG:
            print("Getting gradient")
        nab = grad(last_guess)
        if DEBUG:
            print(nab)
        iter = iter + 1
        step_size = lr / np.sqrt(iter)
        if DEBUG:
            print("Updating:")
            print(last_guess + (nab * step_size))
        next_guess = proj(last_guess + (nab * step_size))
        # continue
        last_guess = next_guess
    return last_guess

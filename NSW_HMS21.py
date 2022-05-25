import numpy as np
import pgd

SAFE_CONST = 8
DEBUG = False

# helper functions ======================================================
def NSW(mu, p, n=None, k=None):
    if n is None or k is None:
        (n, k) = np.shape(mu)
    rewards = np.matmul(mu, p)
    return np.prod(rewards)

def gradNSW(mu, p, n=None, k=None):
    if n is None or k is None:
        (n, k) = np.shape(mu)
    nsw = NSW(mu, p, n=n, k=k)
    rewards = np.matmul(mu, p)
    for i in range(len(rewards)):
        if rewards[i] < 0.001:
            rewards[i] = 0.001
    rewards_less_i = nsw / rewards
    nab = np.matmul(rewards_less_i, mu)
    return nab

def NSWplusR(mu, p, alpha, r_vec, n=None, k=None):
    nsw = NSW(mu, p, n=n, k=k)
    return (nsw + (alpha * np.dot(p, r_vec)))

def gradNSWplusR(mu, p, alpha, r_vec, n=None, k=None):
    nab_NSW = gradNSW(mu, p, n=n, k=k)
    if n is None or k is None:
        (n, k) = np.shape(mu)
    nab_R = alpha * r_vec
    return nab_NSW + nab_R
    

# beta function, with added edge cases for 0 and 1
def my_beta(mu):
    v = []
    for i in range(len(mu)):
        if mu[i] == 1:
            v.append(1)
        elif mu[i] == 0:
            v.append(0)
        else:
            v.append(np.random.beta(mu[i], 1-mu[i]))
    return np.array(v)


def my_binom(mu):
    v = []
    for i in range(len(mu)):
        if mu[i] == 1:
            v.append(1)
        elif mu[i] == 0:
            v.append(0)
        else:
            v.append(np.random.choice([0,1], p=[1-mu[i], mu[i]]))
    return v


# Our problem has learning rate D/Gsqrt(t)
# proj_grad_asc handles the sqrt(t)
# D is the diameter, which is bounded by sqrt(2)
# G is the Lipschitz Condition, which can be quite large
#    Since we optimize log(NSW), small NSW gives very negative values
#    If we initialize to p uniform, our reward is the minimum of any
#    person's average reward, which we can compute
#    We also scale that person's rewards to [0,1], this directly
#    scales NSWs. This means at least 1 mu_(i,_) is 1, so the reward
#    is at least 1/k, and G can be capped at roughly nk (plus some alpha*r)
def max_NSWplusR(mu, alpha, r_vec, init_guess=None, n=None, k=None, delt=0.0005):
    if n is None or k is None:
        (n, k) = np.shape(mu)
    if init_guess is None:
        init_guess = np.array([1/k] * k)
    G = SAFE_CONST * np.linalg.norm(gradNSWplusR(mu, init_guess, alpha, r_vec, n=n, k=k))
    if G > n * k:
        G = n * k
    D = np.sqrt(2)
    lr = D / G
    # define functions with fixed mu
    def local_NSWplusR(p):
        return NSWplusR(mu, p, alpha, r_vec, n=n, k=k)
    def local_gradNSWplusR(p):
        return gradNSWplusR(mu, p, alpha, r_vec, n=n, k=k)
    return pgd.proj_grad_asc(k, local_NSWplusR, local_gradNSWplusR, lr=lr, lookback=25, delt=delt, init_guess=init_guess)



# class to handle the UCB algorithm =====================================
class MAFB_UCB:
    def __init__(self, mu_star, alpha_fn, initiate=True, dist_mu=my_beta, checkpointing=5000, delt=0.000001, C=1):
        # the true mu, passed as a paramter
        self.true_mu = mu_star
        self.alpha_fn = alpha_fn
        self.distr = dist_mu   # pulls from distribution which has
        #                      # range [0,1] and mean as input parameter
        # initate data values, including shape from mu
        self.initiated = False
        (self.n, self.k) = np.shape(mu_star)
        self.rounds = 0
        # do not have values
        self.mu_sum = None
        self.mu_hat = None
        self.cumul_rewards = None
        self.ntj = np.zeros(self.k, dtype=int)
        self.NSWs = []
        self.NSW_max = 0
        self.best_p = None
        self.last_p = None
        self.checkpointing = checkpointing
        self.delt = delt
        self.C = C
        self.t = 1      # t is the next iteration to run,
        if initiate:    # so we've already run t-1 iterations
            self.initiate()

    def num_steps_run(self):
        return (self.t - 1)

    def cumul_rewards(self):
        return self.mu_sum
    
    # draw from Beta(mu,(1-mu)) for each mu(_,j)
    def pull_arm(self, j):
        if DEBUG:
            print(" Pulling arm ", j)
        assert 0 <= j and j < self.k
        mu_j = self.true_mu[:,j]
        rewards =  self.distr(mu_j)
        self.ntj[j] += 1
        self.t += 1
        self.mu_sum[:,j] += rewards
        self.mu_hat[:,j] = self.mu_sum[:,j] / self.ntj[j]

    # pull each arm once, to get initial values
    def initiate(self):
        self.initiated = True
        mu_shape = np.shape(self.true_mu)
        self.mu_sum = np.zeros(mu_shape)
        self.mu_hat = np.zeros(mu_shape)
        for j in range(self.k):
            self.pull_arm(j)
            # compute NSW by hand
            nsw = np.prod(self.true_mu[:,j])
            self.NSWs.append(nsw)

    # wrapper to the function which maximizes NSW
    def compute_p(self, init_guess=None, alpha=None, r_vec=None):
        if r_vec is None:
            r_vec = np.zeros(self.k)
        if alpha is None:
            alpha = 1
        return max_NSWplusR(self.mu_hat, alpha, r_vec * self.C, delt=self.delt, init_guess=init_guess, n=self.n, k=self.k)
    
    # run a single step of the UCB algorithm
    def run_step(self):
        assert (self.mu_hat is not None) # Make sure we've initiated
        # We keep mu_hat up-to-date, so just find p
        if DEBUG:
            print("Iter: ",self.t)
            print("Guess: ",self.last_p)
        # Compute UCB offsets
        r_vec = np.full((self.k), (np.log(self.n * self.k * self.t)))
        r_vec = np.sqrt(r_vec / self.ntj)
        p = self.compute_p(init_guess=self.last_p, alpha=self.alpha_fn(self.t), r_vec=r_vec)
        self.last_p = p
        arm_to_pull = np.random.choice(range(self.k), p=p)
        # pull arm and update state
        self.pull_arm(arm_to_pull)
        # update ongoing list of NSW
        thisNSW = NSW(self.true_mu, p, n=self.n, k=self.k)
        if thisNSW > self.NSW_max:
            self.NSW_max = thisNSW
            self.best_p = p
        self.NSWs.append(thisNSW)

    # run several steps
    def run_steps(self, n_steps):
        if self.checkpointing is None:
            for tm in range(n_steps):
                self.run_step()
        else:
            for tm in range(n_steps):
                if self.t % self.checkpointing == 0:
                    print("Running UCB step ", self.t)
                    print("p_hat is currently:\n", self.last_p)
                self.run_step()

    # run to time T
    # will crash if we have called run_step() at all
    def run_to_T(self, T):
        assert (self.t == 1 and not self.initiated) or (self.t == self.k + 1 and self.initiated)
        if self.t == 1:
            self.initiate()
        self.run_steps(T - self.t - 1)

# =======================================================================


# wrapper for MAFB_UCB __init__
def setup_UCB(mu_star, alpha_fn, checkpointing=None, delt=0.000001, C=1):
    return MAFB_UCB(mu_star, alpha_fn, dist_mu=my_binom, checkpointing=checkpointing, delt=delt, C=C)

# run UCB for either one step, n_steps steps, or up to T total steps
def run_UCB(UCB, n_steps=None, T=None):
    assert n_steps is None or T is None # make sure we call one or the other
    if n_steps is not None:
        UCB.run_steps(n_steps)
    elif T is not None:
        UCB.run_to_T(T)
    else:
        UCB.run_step()

# get NSWs up to time T
def results(mus, T, alpha_fn, checkpointing=None, delt=0.000001, C=1):
    UCB = setup_UCB(mus, alpha_fn, checkpointing=checkpointing, delt=delt, C=C)
    run_UCB(UCB, T=T)
    return UCB.NSWs, UCB.last_p, UCB.NSW_max, UCB.best_p

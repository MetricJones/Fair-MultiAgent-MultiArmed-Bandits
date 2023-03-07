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


def logNSW(mu, p, n=None, k=None):
    return np.log(NSW(mu, p, n=n, k=k))


def grad_logNSW(mu, p, n=None, k=None):
    #print(" Computing gradient on mu = ", mu, ", p = ", p)
    if n is None or k is None:
        (n, k) = np.shape(mu)
    rewards = np.matmul(mu, p)
    mu_t = np.copy(mu)
    for i in range(n):
        mu_t[i,:] = mu_t[i,:] / rewards[i]
    nab = np.sum(mu_t, axis = 0)
    #print(" Got ", nab)
    return nab
#    for i in range(n):
#        scale = 1 / rewards[i]
#        mu_temp[i,:] = mu_temp[i,:] * scale
#    return np.sum(mu_temp, axis=0)
    
    

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

# binomial function
def my_binom(mu):
    v = []
    for i in range(len(mu)):
        if mu[i] == 1:
            v.append(1)
        elif mu[i] == 0:
            v.append(0)
        else:
            v.append(np.random.choice([0,1], p=[1-mu[i], mu[i]]))
    return np.array(v)

# Our problem has learning rate D/Gsqrt(t)
# proj_grad_asc handles the sqrt(t)
# D is the diameter, which is bounded by sqrt(2)
# G is the Lipschitz Condition, which can be quite large
#    Since we optimize log(NSW), small NSW gives very negative values
#    If we initialize to p uniform, our reward is the minimum of any
#    person's average reward, which we can compute
#    We also scale that person's rewards to [0,1], this directly
#    scales NSWs. This means at least 1 mu_(i,_) is 1, so the reward
#    is at least 1/k, and G is at most nk
def max_NSW(mu, init_guess=None, n=None, k=None, delt=0.0002):
    #print("------------ COMPUTING max_NSW -------, mu = ", mu)
    if n is None or k is None:
        (n, k) = np.shape(mu)
    if init_guess is None:
        init_guess = np.array([1/k] * k)
    # Approximate G using local gradient's l2
    G = SAFE_CONST * np.linalg.norm(grad_logNSW(mu, init_guess, n=n, k=k))
    # if this is too little sauce, use the conservative bound
    if G > n * k:
        G = n * k
    D = np.sqrt(2)
    lr = D / G
    # define our functions without mu
    #print("Defining functions with mu = ", mu)
    def local_logNSW(p):
        return logNSW(mu, p, n=n, k=k)
    def local_grad_logNSW(p):
        return grad_logNSW(mu, p, n=n, k=k)
    return pgd.proj_grad_asc(k, local_logNSW, local_grad_logNSW, lr=lr, delt=delt, init_guess=init_guess)


# class to handle the UCB algorithm =====================================
class MAFB_UCB_EG:
    def __init__(self, mu_star, initiate=True, dist_mu=my_beta, checkpointing=5000, C=2, eps_0=None):
        # the true mu, passed as a paramter
        self.true_mu = mu_star
        self.distr = dist_mu   # pulls from distribution which has
        #                      # range [0,1] and mean as input parameter
        # initate data values, including shape from mu
        self.initiated = False
        (self.n, self.k) = np.shape(mu_star)
        self.rounds = 0
        self.round_robin = 0
        if eps_0 is None:
            eps_0 = (self.n ** (1/3)) * (self.k ** (2/3))
        self.eps_0 = eps_0
        self.log_help = np.log(self.n * self.k)
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
        self.C=C
        self.delt = 0.001
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
    def compute_p(self, init_guess=None, r_vec=None):
        if r_vec is None:
            r_vec = np.zeros((self.n, self.k))
        mu_temp = self.mu_hat + r_vec
        cap = np.ones(mu_temp.shape)
        mu_temp = np.minimum(mu_temp, cap)
        return max_NSW(mu_temp, init_guess=init_guess, n=self.n, k=self.k)
    
    # run a single step of the UCB algorithm
    def run_step(self):
        assert (self.mu_hat is not None) # Make sure we've initiated
        # We keep mu_hat up-to-date, so just find p
        if DEBUG:
            print("Iter: ",self.t)
            print("Guess: ",self.last_p)
        # Calculate eps_t
        eps_t = self.eps_0 * (self.t ** (-1/3)) * ((self.log_help + np.log(self.t)) ** (1/3))
        if eps_t >= 1 or np.random.uniform() <= eps_t:
            # Round-robin
            arm_to_pull = self.round_robin
            p = np.zeros(self.k)
            p[arm_to_pull] = 1
            self.round_robin = (self.round_robin + 1) % self.k
        else:
            p = self.compute_p(init_guess=self.last_p)
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
def setup_UCB_EG(mu_star, checkpointing=None, C=2):
    return MAFB_UCB_EG(mu_star, dist_mu=my_binom, checkpointing=checkpointing, C=C)

# run UCB for either one step, n_steps steps, or up to T total steps
def run_UCB_EG(UCB, n_steps=None, T=None):
    assert n_steps is None or T is None # make sure we call one or the other
    if n_steps is not None:
        UCB.run_steps(n_steps)
    elif T is not None:
        UCB.run_to_T(T)
    else:
        UCB.run_step()

# get NSWs up to time T
def results_EG(mus, T, checkpointing=None, C=2):
    UCB = setup_UCB_EG(mus, checkpointing=checkpointing, C=C)
    run_UCB_EG(UCB, T=T)
    return UCB.NSWs, UCB.last_p, UCB.NSW_max, UCB.best_p


# ==========================================================================

# class to handle the UCB algorithm =====================================
class MAFB_UCB_EF:
    def __init__(self, mu_star, initiate=True, dist_mu=my_beta, checkpointing=5000, C=2, L=None, T=50000):
        # the true mu, passed as a paramter
        self.true_mu = mu_star
        self.distr = dist_mu   # pulls from distribution which has
        #                      # range [0,1] and mean as input parameter
        # initate data values, including shape from mu
        self.initiated = False
        (self.n, self.k) = np.shape(mu_star)
        self.rounds = 0
        if L is None:
            L = (self.k ** (-1/3)) * (self.n ** (1/3)) * (T ** (2/3)) * ((np.log(self.n * self.k * T)) ** (2/3))
        self.L = L
        print("L: ", self.L)
        if (self.L * self.k) >= T:
            self.L = T / self.k
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
        self.C=C
        self.delt = 0.001
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
            nsw = np.prod(self.true_mu[:,j])
            for x in range(int(self.L)):
                self.pull_arm(j)
                # compute NSW by hand
                self.NSWs.append(nsw)

    # wrapper to the function which maximizes NSW
    def compute_p(self, init_guess=None, r_vec=None):
        if r_vec is None:
            r_vec = np.zeros((self.n, self.k))
        mu_temp = self.mu_hat + r_vec
        cap = np.ones(mu_temp.shape)
        mu_temp = np.minimum(mu_temp, cap)
        return max_NSW(mu_temp, init_guess=init_guess, n=self.n, k=self.k)
    
    # run a single step of the UCB algorithm
    def run_step(self):
        assert (self.mu_hat is not None) # Make sure we've initiated
        # We keep mu_hat up-to-date, so just find p
        if DEBUG:
            print("Iter: ",self.t)
            print("Guess: ",self.last_p)
        p = self.compute_p(init_guess=self.last_p)
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
        if self.t == 1:
            self.initiate()
        self.run_steps(T - self.t - 1)

# =======================================================================


# wrapper for MAFB_UCB __init__
def setup_UCB_EF(mu_star, checkpointing=None, C=2, T=50000):
    return MAFB_UCB_EF(mu_star, dist_mu=my_binom, checkpointing=checkpointing, C=C, T=T)

# run UCB for either one step, n_steps steps, or up to T total steps
def run_UCB_EF(UCB, n_steps=None, T=None):
    assert n_steps is None or T is None # make sure we call one or the other
    if n_steps is not None:
        UCB.run_steps(n_steps)
    elif T is not None:
        UCB.run_to_T(T)
    else:
        UCB.run_step()

# get NSWs up to time T
def results_EF(mus, T, checkpointing=None, C=2):
    UCB = setup_UCB_EF(mus, checkpointing=checkpointing, C=C, T=T)
    run_UCB_EF(UCB, T=T)
    print(len(UCB.NSWs))
    return UCB.NSWs, UCB.last_p, UCB.NSW_max, UCB.best_p


import NSW
import NSW_HMS21 as NSW2
import NSW_ineff as NSWb
import NSW_ineff as NSWEFG
import pgd
import numpy as np
import matplotlib.pyplot as plt
import csv
import time

seed = time.time_ns()
print(seed)
rng = np.random.default_rng(seed)

# Default settings
T = 10000
DEBUG = False

# random mu sample functions
# Beta with parameters (0.85, 0.15)
def closer_to_1(size):
    return rng.beta(0.85, 0.15, size=size)

# uniform between 0.9 and 1
def all_large(size):
    return rng.uniform(0.9, 1, size=size)

# 1 - exponential(0.04), where each value will be at least 0.9 whp
def exp_large(size):
    return 1 - rng.exponential(scale=0.04, size=size)


# Either takes (n,k) and makes a random mu using function r, or takes mu and finds n,k
# Tests the FMAMAB functions with given iterations (T)
#  nsw1 -> UCB JNN
#  nsw2 -> UCB HMS21 (alpha = N)
#  nsw3 -> UCB HMS21 (alpha = sqrt(12*NK*log(NKt))
#  prnt is a tag to print some info to console, including mu and testing checkpoints
#  checkpointing prints an iteration count every "checkpointing" steps, or no printing if None
#  plotfile is the file to save the plot too, if None then show the plot
def test_MAB(n=None, k=None, mu=None, T=800000, r=rng.random, prnt=False, nsw1=True, nsw2=True, nsw3=True, checkpointing=5000, plotfile=None, timed=False):
    # initialize
    NSWs_1, p_hat_1, NSWs_2, p_hat_2 = None, None, None, None
    # fill in mu
    if mu is not None:
        (nt,kt) = np.array(mu).shape
        assert (n is None or nt == n)
        assert (k is None or kt == k)
    else:
        if n is None:
            n = 5
        if k is None:
            k = 5
        mu = r(size=(n,k))
    # scale mu such that each agent's rewards have a maximum of 1
    mu_max = np.max(mu, axis=1)
    for i in range(n):
        mu[i,:] = mu[i,:] / mu_max[i]
    if prnt:
        print(" mu:\n", mu)
    # Get an initial test, knowing mu
    p_star = NSW.max_NSW(mu, init_guess=([1/k] * k), n=n, k=k, delt=0.000000001)
    NSW_star = NSW.NSW(mu, p_star, n=n, k=k)
    # Run FMAB UCB
    if nsw1:
        if prnt:
            print(" Testing UCB_JNN")
        s_t = time.time()
        NSWs_1, p_hat_1, NSW_max_1, best_p_1 = NSW.results(mu, T, checkpointing=checkpointing, C=0.5)
        if NSW_max_1 > NSW_star:
            NSW_star = NSW_max_1
            p_star = best_p_1
        f_t = time.time()
        if timed and prnt:
        	print(' s elapsed: ', f_t - s_t)
    # UCB from HMS21
    def alphaN(t):
        return n
    def alphaNotN(t):
        return np.sqrt(12 * n * k * np.log(n * k * t))
    if nsw2:
        if prnt:
            print(" Testing UCB_HMS21 w/ alpha = N")
        s_t = time.time()
        NSWs_2, p_hat_2, NSW_max_2, best_p_2 = NSW2.results(mu, T, alphaN, checkpointing=checkpointing, delt=0.000001, C=0.8)
        if NSW_max_2 > NSW_star:
            NSW_star = NSW_max_2
            p_star = best_p_2
        f_t = time.time()
        if timed and prnt:
        	print(' s elapsed: ', f_t - s_t)
    if nsw3:
        if prnt:
            print(" Testing UCB_HMS21 w/ alpha = sqrt(12NKlog(NKt))")
        s_t = time.time()
        NSWs_3, p_hat_3, NSW_max_3, best_p_3 = NSW2.results(mu, T, alphaNotN, checkpointing=checkpointing)
        if NSW_max_3 > NSW_star:
            NSW_star = NSW_max_3
            p_star = best_p_3
        f_t = time.time()
        if timed and prnt:
        	print(' s elapsed: ', f_t - s_t)
    # get the regrets
    if nsw1:
        i_regs_1 = NSW_star - NSWs_1
        i_regs_1 = np.insert(i_regs_1, 0, 0)
        cumul = 0
        cumul_regs_1 = []
        for t in range(len(i_regs_1)):
            cumul += i_regs_1[t]
            cumul_regs_1.append(cumul)
    if nsw2:
        i_regs_2 = NSW_star - NSWs_2
        i_regs_2 = np.insert(i_regs_2, 0, 0)
        cumul = 0
        cumul_regs_2 = []
        for t in range(len(i_regs_2)):
            cumul += i_regs_2[t]
            cumul_regs_2.append(cumul)
    if nsw3:
        i_regs_3 = NSW_star - NSWs_3
        i_regs_3 = np.insert(i_regs_3, 0, 0)
        cumul = 0
        cumul_regs_3 = []
        for t in range(len(i_regs_3)):
            cumul += i_regs_3[t]
            cumul_regs_3.append(cumul)
    # plot
    if not nsw1 or not nsw2:
        if not nsw1 and not nsw2:
            return None, None, None
        else:
            if nsw1:
                i_plot = i_regs_1
                c_plot = cumul_regs_1
            if nsw2:
                i_plot = i_regs_2
                c_plot = cumul_regs_2
            fig, ax1 = plt.subplots()
            ax1.set_xlabel('Iterations')
            ax1.plot(c_plot, label='Cumul. Regret', color='tab:blue')
            ax2 = ax1.twinx()
            ax2.plot(i_plot, label='Instant Regret', color='tab:red')
            plt.legend()
            if plotfile is None:
                #plt.show()
                x = 1
            else:
                plt.savefig(plotfile)
            plt.close()
            if nsw1:
                return (NSWs_1, p_hat_1, i_regs_1, cumul_regs_1), None, None, (NSW_star, p_star)
            if nsw2:
                return None, (NSWs_2, p_hat_2, i_regs_2, cumul_regs_2), None, (NSW_star, p_star)
    else:
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Iterations')
        if nsw1:
            ax1.plot(cumul_regs_1, label='Our C. Regret', color='tab:blue')
        if nsw2:
            ax1.plot(cumul_regs_2, label='HMS21 C. Regret, a = n', color='tab:red')
        if nsw3:
            ax1.plot(cumul_regs_3, label='HMS21 C. Regret, a != n', color='tab:green')
        plt.legend()
        if plotfile is None:
            #plt.show()
            x = 1
        else:
            plt.savefig(plotfile)
        plt.close()
        if nsw3:
            return (NSWs_1, p_hat_1, best_p_1, i_regs_1, cumul_regs_1), (NSWs_2, p_hat_2, best_p_2, i_regs_2, cumul_regs_2), (NSWs_3, p_hat_3, best_p_3, i_regs_3, cumul_regs_3), (NSW_star, p_star), mu
        else:
            return (NSWs_1, p_hat_1, best_p_1, i_regs_1, cumul_regs_1), (NSWs_2, p_hat_2, best_p_2, i_regs_2, cumul_regs_2), None, (NSW_star, p_star), mu
            

# Tests many levels: for each N in n, K in k, run n_iter tests using test_MAB
# print the results in a file to be read by the other .py files, and save the plots in a folder
#  given by plotfolder
def test_MAB_many(n_iter, n, k, T_max, r=rng.random, csvfile='results.csv', plotfolder='', nsw3=True):
    # Open the write file
    write_file = open(csvfile, 'w', newline='')
    writer = csv.writer(write_file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['TestData'])
    writer.writerow(['n', 'k', 'n_iter'])
    writer.writerow([n, k, n_iter])
    # For each (N,K) pair
    for i in n:
        for j in k:
            for x in range(n_iter):
                print('n = ', i, ', k = ', j, ', iteration ', x)
                writer.writerow([i, j, x])
                # run tests
                plotfile = plotfolder + 'results_' + str(int(i)) + '_' + str(int(j)) + '_' + str(int(x)) + '.png'
                # Run the tests
                if nsw3:
                    (NSWs1, pHat1, bestP1, iRegs1, cRegs1), (NSWs2, pHat2, bestP2, iRegs2, cRegs2), (NSWs3, pHat3, bestP3, iRegs3, cRegs3), (NSWStar, PStar), mu = test_MAB(n=i, k=j, mu=None, T=T_max, r=r, prnt=True, checkpointing=None, plotfile=plotfile, nsw3=nsw3)
                else:
                    (NSWs1, pHat1, bestP1, iRegs1, cRegs1), (NSWs2, pHat2, bestP2, iRegs2, cRegs2), _, (NSWStar, PStar), mu = test_MAB(n=i, k=j, mu=None, T=T_max, r=r, prnt=True, checkpointing=None, plotfile=plotfile, nsw3=nsw3)
                writer.writerow(mu)
                writer.writerow(cRegs1)
                writer.writerow(pHat1)
                writer.writerow(cRegs2)
                writer.writerow(pHat2)
                if nsw3:
                    writer.writerow(cRegs3)
                    writer.writerow(pHat3)
                writer.writerow(PStar)
                writer.writerow([NSWStar])
    write_file.close()


# Similar to test_MAB_many, but takes an array of (N,K) pairs instead of N and K arrays to
#  narrow the testing
def test_MAB_pairs(n_iter, nk_pairs, T_max, r=all_large, csvfile='pair_results.csv', plotfolder=''):
    write_file = open(csvfile, 'w', newline='')
    writer = csv.writer(write_file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['TestData'])
    writer.writerow(['n', 'k', 'n_iter'])
    writer.writerow([nk_pairs, n_iter])
    for m in nk_pairs:
        (i,j) = m
        for x in range(n_iter):
            print('n = ', i, ', k = ', j, ', iteration ', x)
            writer.writerow([i,j,x])
            # run tests
            #plotfile = plotfolder + 'results_' + str(int(i)) + '_' + str(int(j)) + '_' + str(int(x)) + '.png'
            plotfile500000 = plotfolder + 'results_' + str(int(i)) + '_' + str(int(j)) + '_' + str(int(x)) + '_500000.png'
            plotfile100000 = plotfolder + 'results_' + str(int(i)) + '_' + str(int(j)) + '_' + str(int(x)) + '_100000.png'
            plotfile20000 = plotfolder + 'results_' + str(int(i)) + '_' + str(int(j)) + '_' + str(int(x)) + '_20000.png'
            #(NSWs1, pHat1, bestP1, iRegs1, cRegs1), (NSWs2, pHat2, bestP2, iRegs2, cRegs2), _, (NSWStar, PStar), mu = test_MAB(n=i, k=j, mu=None, T=T_max, r=r, prnt=True, checkpointing=50000, plotfile=plotfile, nsw3=False)
            (NSWs1, pHat1, bestP1, iRegs1, cRegs1), (NSWs2, pHat2, bestP2, iRegs2, cRegs2), _, (NSWStar, PStar), mu = test_MAB(n=i, k=j, mu=None, T=T_max, r=r, prnt=True, checkpointing=100000, plotfile=None, nsw3=False)
            fig, ax1 = plt.subplots()
            ax1.set_xlabel('Iterations')
            ax1.plot(cRegs1, label='Our C. Regret', color='tab:blue')
            ax1.plot(cRegs2, label='HMS21 C. Regret, a = n', color='tab:red')
            plt.legend()
            plt.savefig(plotfile500000)
            plt.close()
            ax1.clear()
            fig, ax2 = plt.subplots()
            ax2.set_xlabel('Iterations')
            ax2.plot(cRegs1[:100000], label='Our C. Regret', color='tab:blue')
            ax2.plot(cRegs2[:100000], label='HMS21 C. Regret, a = n', color='tab:red')
            plt.legend()
            plt.savefig(plotfile100000)
            plt.close()
            ax2.clear()
            fig, ax3 = plt.subplots()
            ax3.set_xlabel('Iterations')
            ax3.plot(cRegs1[:20000], label='Our C. Regret', color='tab:blue')
            ax3.plot(cRegs2[:20000], label='HMS21 C. Regret, a = n', color='tab:red')
            plt.legend()
            plt.savefig(plotfile20000)
            plt.close()
            ax3.clear()
            writer.writerow(mu)
            writer.writerow(cRegs1)
            writer.writerow(pHat1)
            writer.writerow(cRegs2)
            writer.writerow(pHat2)
            writer.writerow(PStar)
            writer.writerow([NSWStar])
    write_file.close()


def test_MAB_time(n=None, k=None, mu=None, T=800000, r=rng.random, prnt=False, nsw1=True, nsw2=True, nsw3=True, checkpointing=5000, plotfile=None):
    # initialize
    NSWs_1, p_hat_1, NSWs_2, p_hat_2 = None, None, None, None
    # fill in mu
    if mu is not None:
        (nt,kt) = np.array(mu).shape
        assert (n is None or nt == n)
        assert (k is None or kt == k)
    else:
        if n is None:
            n = 5
        if k is None:
            k = 5
        mu = r(size=(n,k))
    # scale mu such that each agent's rewards have a maximum of 1
    mu_max = np.max(mu, axis=1)
    for i in range(n):
        mu[i,:] = mu[i,:] / mu_max[i]
    if prnt:
        print(" mu:\n", mu)
    # Get an initial test, knowing mu
    p_star = NSW.max_NSW(mu, init_guess=([1/k] * k), n=n, k=k, delt=0.000000001)
    NSW_star = NSW.NSW(mu, p_star, n=n, k=k)
    # Run FMAB UCB
    if nsw1:
        if prnt:
            print(" Testing UCB_JNN")
        s_t = time.time()
        NSWs_1, p_hat_1, NSW_max_1, best_p_1 = NSW.results(mu, T, checkpointing=checkpointing, C=0.5)
        if NSW_max_1 > NSW_star:
            NSW_star = NSW_max_1
            p_star = best_p_1
        f_t = time.time()
        print(" Time elapsed: ", f_t - s_t)
    # UCB from HMS21
    def alphaN(t):
        return n
    def alphaNotN(t):
        return np.sqrt(12 * n * k * np.log(n * k * t))
    if nsw2:
        if prnt:
            print(" Testing UCB_HMS21 w/ alpha = N")
        s_t = time.time()
        NSWs_2, p_hat_2, NSW_max_2, best_p_2 = NSW2.results(mu, T, alphaN, checkpointing=checkpointing, delt=0.0001, C=0.8)
        if NSW_max_2 > NSW_star:
            NSW_star = NSW_max_2
            p_star = best_p_2
        f_t = time.time()
        print(" Time elapsed: ", f_t - s_t)
    if nsw3:
        if prnt:
            print(" Testing UCB_HMS21 w/ alpha = sqrt(12NKlog(NKt))")
        s_t = time.time()
        NSWs_3, p_hat_3, NSW_max_3, best_p_3 = NSW2.results(mu, T, alphaNotN, checkpointing=checkpointing)
        if NSW_max_3 > NSW_star:
            NSW_star = NSW_max_3
            p_star = best_p_3
        f_t = time.time()
        print(" Time elapsed: ", f_t - s_t)        
    return None, None, None, (NSW_star, p_star)

    

# Similar to test_MAB_many, but takes an array of (N,K) pairs instead of N and K arrays to
#  narrow the testing
def test_MAB_pairs_time(n_iter, nk_pairs, T_max, r=all_large):
    for m in nk_pairs:
        (i,j) = m
        for x in range(n_iter):
            print('n = ', i, ', k = ', j, ', iteration ', x)
            # run tests
            (NSWs1, pHat1, bestP1, iRegs1, cRegs1), (NSWs2, pHat2, bestP2, iRegs2, cRegs2), _, (NSWStar, PStar), mu = test_MAB(n=i, k=j, mu=None, T=T_max, r=r, prnt=True, checkpointing=150000, nsw3=False, timed=True)
            print(cRegs1[T_max - 10])
            print(cRegs2[T_max - 10])
    


# Test to find the optimal parameters
# on a given (N,K), generate a mu and test the same function with different parameters
#    Either tests the constants C on the added values in the NSW algorithms, or
#    tests deltSet as values of delta in the PGA for NSW2 (to deal with pseudo-concavity)
def test_MAB_stability(n, k, T_max, r=rng.random, csvfile='results.csv', CSet=None, deltSet=None):
    if CSet is None and deltSet is None:
        return None
    if not (CSet is None or deltSet is None):
        return None
    # Initialize values
    write_file = open(csvfile, 'w')
    print("TestData", file=write_file)
    mu = r(size=(n,k))
    mu_max = np.max(mu, axis=1)
    for i in range(n):
        mu[i,:] = mu[i,:] / mu_max[i]
    print(mu)
    def alphaN(t):
        return n
    p_star = NSW.max_NSW(mu, init_guess=([1/k] * k), n=n, k=k, delt=0.000000001)
    NSW_star = NSW.NSW(mu, p_star, n=n, k=k)
    print(" Computed p_star: \n", p_star)

    # initialize accumulators
    NSWs2 = []
    NSWs1 = []

    if CSet is not None:
        lCSet = len(CSet)
    if deltSet is not None:
        ldeltSet = len(deltSet)

    # test
    if CSet is not None:
        # test delta
        for i in range(lCSet):
            NSWs_1, p_hat_1, NSW_max_1, best_p_1 = NSW.results(mu, T_max, checkpointing=200000, C=CSet[i])
            NSWs_2, p_hat_2, NSW_max_2, best_p_2 = NSW2.results(mu, T_max, alphaN, checkpointing=200000, C=CSet[i])
            if NSW_max_1 > NSW_star:
                NSW_star = NSW_max_1
                p_star = best_p_1
            if NSW_max_2 > NSW_star:
                NSW_star = NSW_max_2
                p_star = best_p_2
            NSWs1.append(NSWs_1)
            NSWs2.append(NSWs_2)
        for i in range(lCSet):
            cregs1 = []
            cregs2 = []
            creg = 0
            cregs1.append(creg)
            cregs2.append(creg)
            for j in range(len(NSWs1[i])):
                creg = creg + (NSW_star - NSWs1[i][j])
                cregs1.append(creg)
            creg = 0
            for j in range(len(NSWs2[i])):
                creg = creg + (NSW_star - NSWs2[i][j])
                cregs2.append(creg)
            print(" C = " + str(CSet[i]), file=write_file)
            print(" UCB JNN", file=write_file)
            print("  T = 200,000: " + str(cregs1[200000]), file=write_file)
            print("  T = 500,000: " + str(cregs1[499995]), file=write_file)
            print(" UCB HMS21", file=write_file)
            print("  T = 200,000: " + str(cregs2[200000]), file=write_file)
            print("  T = 500,000: " + str(cregs2[499995]), file=write_file)
        return None
    elif deltSet is not None:
        # test C
        for j in range(ldeltSet):
            NSWs_2, p_hat_2, NSW_max_2, best_p_2 = NSW2.results(mu, T_max, alphaN, checkpointing=50000, delt=deltSet[j])
            if NSW_max_2 > NSW_star:
                NSW_star = NSW_max_2
                p_star = best_p_2
            NSWs2.append(NSWs_2)
        for j in range(ldeltSet):
            cregs2 = []
            creg = 0
            cregs2.append(creg)
            for i in range(len(NSWs2[j])):
                creg = creg + (NSW_star - NSWs2[j][i])
                cregs2.append(creg)
            plt.plot(cregs2, label=('Delt = ' + str(deltSet[j])))
            plt.show()
            plt.clf()
        return None
        
        
# Either takes (n,k) and makes a random mu using function r, or takes mu and finds n,k
# Tests the FMAMAB functions with given iterations (T)
#  nsw1 -> UCB JNN
#  nsw2 -> UCB HMS21 (alpha = N)
#  nsw3 -> UCB HMS21 (alpha = sqrt(12*NK*log(NKt))
#  prnt is a tag to print some info to console, including mu and testing checkpoints
#  checkpointing prints an iteration count every "checkpointing" steps, or no printing if None
#  plotfile is the file to save the plot too, if None then show the plot
def test_MAB2(n=None, k=None, mu=None, T=800000, r=rng.random, prnt=False, nsw1=True, nsw2=True, checkpointing=5000, plotfile=None, timed=False):
    # initialize
    NSWs_1, p_hat_1, NSWs_2, p_hat_2 = None, None, None, None
    # fill in mu
    if mu is not None:
        (nt,kt) = np.array(mu).shape
        assert (n is None or nt == n)
        assert (k is None or kt == k)
    else:
        if n is None:
            n = 5
        if k is None:
            k = 5
        mu = r(size=(n,k))
    # scale mu such that each agent's rewards have a maximum of 1
    mu_max = np.max(mu, axis=1)
    for i in range(n):
        mu[i,:] = mu[i,:] / mu_max[i]
    if prnt:
        print(" mu:\n", mu)
    # Get an initial test, knowing mu
    p_star = NSW.max_NSW(mu, init_guess=([1/k] * k), n=n, k=k, delt=0.000000001)
    NSW_star = NSW.NSW(mu, p_star, n=n, k=k)
    # Run FMAB UCB
    if nsw1:
        if prnt:
            print(" Testing UCB_JNN")
        s_t = time.time()
        NSWs_1, p_hat_1, NSW_max_1, best_p_1 = NSW.results(mu, T, checkpointing=checkpointing, C=0.5)
        if NSW_max_1 > NSW_star:
            NSW_star = NSW_max_1
            p_star = best_p_1
        f_t = time.time()
        if timed and prnt:
        	print(' s elapsed: ', f_t - s_t)
    # UCB from HMS21
    def alphaN(t):
        return n
    def alphaNotN(t):
        return np.sqrt(12 * n * k * np.log(n * k * t))
    if nsw2:
        if prnt:
            print(" Testing UCB_HMS21 w/ alpha = N")
        s_t = time.time()
        NSWs_2, p_hat_2, NSW_max_2, best_p_2 = NSWb.results(mu, T, alphaN, checkpointing=checkpointing, delt=0.000001, C=0.8, full_startup=False)
        if NSW_max_2 > NSW_star:
            NSW_star = NSW_max_2
            p_star = best_p_2
        f_t = time.time()
        if timed and prnt:
        	print(' s elapsed: ', f_t - s_t)
    # get the regrets
    if nsw1:
        i_regs_1 = NSW_star - NSWs_1
        i_regs_1 = np.insert(i_regs_1, 0, 0)
        cumul = 0
        cumul_regs_1 = []
        for t in range(len(i_regs_1)):
            cumul += i_regs_1[t]
            cumul_regs_1.append(cumul)
    if nsw2:
        i_regs_2 = NSW_star - NSWs_2
        i_regs_2 = np.insert(i_regs_2, 0, 0)
        cumul = 0
        cumul_regs_2 = []
        for t in range(len(i_regs_2)):
            cumul += i_regs_2[t]
            cumul_regs_2.append(cumul)
    # plot
    if not nsw1 or not nsw2:
        if not nsw1 and not nsw2:
            return None, None, None
        else:
            if nsw1:
                i_plot = i_regs_1
                c_plot = cumul_regs_1
            if nsw2:
                i_plot = i_regs_2
                c_plot = cumul_regs_2
            fig, ax1 = plt.subplots()
            ax1.set_xlabel('Iterations')
            ax1.plot(c_plot, label='Cumul. Regret', color='tab:blue')
            ax2 = ax1.twinx()
            ax2.plot(i_plot, label='Instant Regret', color='tab:red')
            plt.legend()
            if plotfile is None:
                #plt.show()
                x = 1
            else:
                plt.savefig(plotfile)
            plt.close()
            if nsw1:
                return (NSWs_1, p_hat_1, i_regs_1, cumul_regs_1), None, None, (NSW_star, p_star)
            if nsw2:
                return None, (NSWs_2, p_hat_2, i_regs_2, cumul_regs_2), None, (NSW_star, p_star)
    else:
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Iterations')
        if nsw1:
            ax1.plot(cumul_regs_1, label='Our C. Regret', color='tab:blue')
        if nsw2:
            ax1.plot(cumul_regs_2, label='Ineff. C. Regret, a = n', color='tab:red')
        plt.legend()
        if plotfile is None:
            #plt.show()
            x = 1
        else:
            plt.savefig(plotfile)
        plt.close()
        return (NSWs_1, p_hat_1, best_p_1, i_regs_1, cumul_regs_1), (NSWs_2, p_hat_2, best_p_2, i_regs_2, cumul_regs_2), None, (NSW_star, p_star), mu



    # Similar to test_MAB_many, but takes an array of (N,K) pairs instead of N and K arrays to
#  narrow the testing
def test_MAB_pairs2(n_iter, nk_pairs, T_max, r=all_large, csvfile='pair_results.csv', plotfolder=''):
    write_file = open(csvfile, 'w', newline='')
    writer = csv.writer(write_file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['TestData'])
    writer.writerow(['n', 'k', 'n_iter'])
    writer.writerow([nk_pairs, n_iter])
    for m in nk_pairs:
        (i,j) = m
        for x in range(n_iter):
            print('n = ', i, ', k = ', j, ', iteration ', x)
            writer.writerow([i,j,x])
            # run tests
            #plotfile = plotfolder + 'results_' + str(int(i)) + '_' + str(int(j)) + '_' + str(int(x)) + '.png'
            plotfile500000 = plotfolder + 'results_' + str(int(i)) + '_' + str(int(j)) + '_' + str(int(x)) + '_500000.png'
            plotfile100000 = plotfolder + 'results_' + str(int(i)) + '_' + str(int(j)) + '_' + str(int(x)) + '_100000.png'
            plotfile20000 = plotfolder + 'results_' + str(int(i)) + '_' + str(int(j)) + '_' + str(int(x)) + '_20000.png'
            #(NSWs1, pHat1, bestP1, iRegs1, cRegs1), (NSWs2, pHat2, bestP2, iRegs2, cRegs2), _, (NSWStar, PStar), mu = test_MAB(n=i, k=j, mu=None, T=T_max, r=r, prnt=True, checkpointing=50000, plotfile=plotfile, nsw3=False)
            (NSWs1, pHat1, bestP1, iRegs1, cRegs1), (NSWs2, pHat2, bestP2, iRegs2, cRegs2), _, (NSWStar, PStar), mu = test_MAB2(n=i, k=j, mu=None, T=T_max, r=r, prnt=True, checkpointing=100000, plotfile=None)
            fig, ax1 = plt.subplots()
            ax1.set_xlabel('Iterations')
            ax1.plot(cRegs1, label='Our C. Regret', color='tab:blue')
            ax1.plot(cRegs2, label='HMS21 C. Regret, a = n', color='tab:red')
            plt.legend()
            plt.savefig(plotfile500000)
            plt.close()
            ax1.clear()
            fig, ax2 = plt.subplots()
            ax2.set_xlabel('Iterations')
            ax2.plot(cRegs1[:100000], label='Our C. Regret', color='tab:blue')
            ax2.plot(cRegs2[:100000], label='HMS21 C. Regret, a = n', color='tab:red')
            plt.legend()
            plt.savefig(plotfile100000)
            plt.close()
            ax2.clear()
            fig, ax3 = plt.subplots()
            ax3.set_xlabel('Iterations')
            ax3.plot(cRegs1[:20000], label='Our C. Regret', color='tab:blue')
            ax3.plot(cRegs2[:20000], label='HMS21 C. Regret, a = n', color='tab:red')
            plt.legend()
            plt.savefig(plotfile20000)
            plt.close()
            ax3.clear()
            writer.writerow(mu)
            writer.writerow(cRegs1)
            writer.writerow(pHat1)
            writer.writerow(cRegs2)
            writer.writerow(pHat2)
            writer.writerow(PStar)
            writer.writerow([NSWStar])
    write_file.close()


# Either takes (n,k) and makes a random mu using function r, or takes mu and finds n,k
# Tests the FMAMAB functions with given iterations (T)
#  nsw1 -> UCB JNN
#  nsw2 -> UCB HMS21 (alpha = N)
#  nsw3 -> UCB HMS21 (alpha = sqrt(12*NK*log(NKt))
#  prnt is a tag to print some info to console, including mu and testing checkpoints
#  checkpointing prints an iteration count every "checkpointing" steps, or no printing if None
#  plotfile is the file to save the plot too, if None then show the plot
def test_MAB3(n=None, k=None, mu=None, T=800000, r=rng.random, prnt=False, nsw1=True, nsw2=True, nsw3=True, checkpointing=5000, plotfile=None, timed=False):
    # initialize
    NSWs_1, p_hat_1, NSWs_2, p_hat_2, NSWs_3, p_hat_3 = None, None, None, None, None, None
    # fill in mu
    if mu is not None:
        (nt,kt) = np.array(mu).shape
        assert (n is None or nt == n)
        assert (k is None or kt == k)
    else:
        if n is None:
            n = 5
        if k is None:
            k = 5
        mu = r(size=(n,k))
    # scale mu such that each agent's rewards have a maximum of 1
    mu_max = np.max(mu, axis=1)
    for i in range(n):
        mu[i,:] = mu[i,:] / mu_max[i]
    if prnt:
        print(" mu:\n", mu)
    # Get an initial test, knowing mu
    p_star = NSW.max_NSW(mu, init_guess=([1/k] * k), n=n, k=k, delt=0.000000001)
    NSW_star = NSW.NSW(mu, p_star, n=n, k=k)
    # Run FMAB UCB
    if nsw1:
        if prnt:
            print(" Testing UCB_JNN")
        s_t = time.time()
        NSWs_1, p_hat_1, NSW_max_1, best_p_1 = NSW.results(mu, T, checkpointing=checkpointing, C=0.5)
        if NSW_max_1 > NSW_star:
            NSW_star = NSW_max_1
            p_star = best_p_1
        f_t = time.time()
        if timed and prnt:
        	print(' s elapsed: ', f_t - s_t)
    if nsw2:
        if prnt:
            print(" Testing Explore-First")
        s_t = time.time()
        NSWs_2, p_hat_2, NSW_max_2, best_p_2 = NSWEFG.results_EF(mu, T=T, checkpointing=checkpointing, C=0.8)
        if NSW_max_2 > NSW_star:
            NSW_star = NSW_max_2
            p_star = best_p_2
        f_t = time.time()
        if timed and prnt:
        	print(' s elapsed: ', f_t - s_t)
    if nsw3:
        if prnt:
            print(" Testing Epsilon-Greedy")
        s_t = time.time()
        NSWs_3, p_hat_3, NSW_max_3, best_p_3 = NSWEFG.results_EG(mu, T=T, checkpointing=checkpointing, C=0.8)
        if NSW_max_3 > NSW_star:
            NSW_star = NSW_max_3
            p_star = best_p_3
        f_t = time.time()
        if timed and prnt:
        	print(' s elapsed: ', f_t - s_t)
    # get the regrets
    if nsw1:
        i_regs_1 = NSW_star - NSWs_1
        i_regs_1 = np.insert(i_regs_1, 0, 0)
        cumul = 0
        cumul_regs_1 = []
        for t in range(len(i_regs_1)):
            cumul += i_regs_1[t]
            cumul_regs_1.append(cumul)
    if nsw2:
        i_regs_2 = NSW_star - NSWs_2
        i_regs_2 = np.insert(i_regs_2, 0, 0)
        cumul = 0
        cumul_regs_2 = []
        for t in range(len(i_regs_2)):
            cumul += i_regs_2[t]
            cumul_regs_2.append(cumul)
    if nsw3:
        i_regs_3 = NSW_star - NSWs_3
        i_regs_3 = np.insert(i_regs_3, 0, 0)
        cumul = 0
        cumul_regs_3 = []
        for t in range(len(i_regs_3)):
            cumul += i_regs_3[t]
            cumul_regs_3.append(cumul)
    # plot
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Iterations')
    if nsw1:
        ax1.plot(cumul_regs_1, label='Our C. Regret', color='tab:blue')
    if nsw2:
        ax1.plot(cumul_regs_2, label='EF C. Regret', color='tab:red')
    if nsw3:
        ax1.plot(cumul_regs_3, label='EG C. Regret', color='tab:green')
    plt.legend()
    if plotfile is None:
        #plt.show()
        x = 1
    else:
        plt.savefig(plotfile)
    plt.close()
    return (NSWs_1, p_hat_1, best_p_1, i_regs_1, cumul_regs_1), (NSWs_2, p_hat_2, best_p_2, i_regs_2, cumul_regs_2), (NSWs_3, p_hat_3, best_p_3, i_regs_3, cumul_regs_3), (NSW_star, p_star), mu


    
    # Similar to test_MAB_pairs2, but for Explore-First and Epsilon-Greedy
def test_MAB_pairs_EFG(n_iter, nk_pairs, T_max, r=all_large, csvfile='pair_results.csv', plotfolder=''):
    write_file = open(csvfile, 'w', newline='')
    writer = csv.writer(write_file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['TestData'])
    writer.writerow(['n', 'k', 'n_iter'])
    writer.writerow([nk_pairs, n_iter])
    for m in nk_pairs:
        (i,j) = m
        for x in range(n_iter):
            print('n = ', i, ', k = ', j, ', iteration ', x)
            writer.writerow([i,j,x])
            # run tests
            #plotfile = plotfolder + 'results_' + str(int(i)) + '_' + str(int(j)) + '_' + str(int(x)) + '.png'
            plotfile500000 = plotfolder + 'results_' + str(int(i)) + '_' + str(int(j)) + '_' + str(int(x)) + '_500000.png'
            plotfile100000 = plotfolder + 'results_' + str(int(i)) + '_' + str(int(j)) + '_' + str(int(x)) + '_100000.png'
            plotfile20000 = plotfolder + 'results_' + str(int(i)) + '_' + str(int(j)) + '_' + str(int(x)) + '_20000.png'
            #(NSWs1, pHat1, bestP1, iRegs1, cRegs1), (NSWs2, pHat2, bestP2, iRegs2, cRegs2), _, (NSWStar, PStar), mu = test_MAB(n=i, k=j, mu=None, T=T_max, r=r, prnt=True, checkpointing=50000, plotfile=plotfile, nsw3=False)
            (NSWs1, pHat1, bestP1, iRegs1, cRegs1), (NSWs2, pHat2, bestP2, iRegs2, cRegs2), (NSWs3, pHat3, bestP3, iRegs3, cRegs3), (NSWStar, PStar), mu = test_MAB3(n=i, k=j, mu=None, T=T_max, r=r, prnt=True, checkpointing=100000, plotfile=None)
            fig, ax1 = plt.subplots()
            ax1.set_xlabel('Iterations')
            ax1.plot(cRegs1, label='Our C. Regret', color='tab:blue')
            ax1.plot(cRegs2, label='EF C. Regret', color='tab:red')
            ax1.plot(cRegs3, label='EG C. Regret', color='tab:green')
            plt.legend()
            plt.savefig(plotfile500000)
            plt.close()
            ax1.clear()
            fig, ax2 = plt.subplots()
            ax2.set_xlabel('Iterations')
            ax2.plot(cRegs1[:100000], label='Our C. Regret', color='tab:blue')
            ax2.plot(cRegs2[:100000], label='EF C. Regret', color='tab:red')
            ax2.plot(cRegs3[:100000], label='EG C. Regret', color='tab:green')
            plt.legend()
            plt.savefig(plotfile100000)
            plt.close()
            ax2.clear()
            fig, ax3 = plt.subplots()
            ax3.set_xlabel('Iterations')
            ax3.plot(cRegs1[:20000], label='Our C. Regret', color='tab:blue')
            ax3.plot(cRegs2[:20000], label='EF C. Regret', color='tab:red')
            ax3.plot(cRegs3[:20000], label='EG C. Regret', color='tab:green')
            plt.legend()
            plt.savefig(plotfile20000)
            plt.close()
            ax3.clear()
            writer.writerow(mu)
            writer.writerow(cRegs1)
            #writer.writerow(pHat1)
            writer.writerow(cRegs2)
            #writer.writerow(pHat2)
            writer.writerow(cRegs3)
            #writer.writerow(pHat3)
            writer.writerow(PStar)
            writer.writerow([NSWStar])
    write_file.close()
    
# main
# example calls, right now it is set to test pairs but this can be switched
if __name__ == "__main__":
    # This line tests the timing:
    #test_MAB_pairs_time(2, [(4,2), (20,4), (80,8)], 500000, r=exp_large)
    # This line tests the performance:
    #test_MAB_pairs(10, [(80,8), (20,4), (4,2)], 500000, r=exp_large, plotfolder='pair_plots/')
    # This line tests various values of C on the two functions:
    #test_MAB_stability(80, 8, 500000, r=all_large, csvfile='results.csv', CSet=[0.2,0.4,0.6,0.8,1.0])
    test_MAB_pairs_EFG(3, [(4,2), (20,4), (80,8)], 500000, r=exp_large, plotfolder='EFG_plots/')

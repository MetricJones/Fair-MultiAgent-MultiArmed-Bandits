
import NSW
import NSW_HMS21 as NSW2
import pgd
import numpy as np
import matplotlib.pyplot as plt
import csv

# Default settings
T = 10000
DEBUG = False

# random mu sample functions
# Beta with parameters (0.85, 0.15)
def closer_to_1(size):
    return np.random.beta(0.85, 0.15, size=size)

# uniform between 0.9 and 1
def all_large(size):
    return np.random.uniform(0.9, 1, size=size)

# 1 - exponential(0.04), where each value will be at least 0.9 whp
def exp_large(size):
    return 1 - np.random.exponential(scale=0.04, size=size)


# Either takes (n,k) and makes a random mu using function r, or takes mu and finds n,k
# Tests the FMAMAB functions with given iterations (T)
#  nsw1 -> UCB JNN
#  nsw2 -> UCB HMS21 (alpha = N)
#  nsw3 -> UCB HMS21 (alpha = sqrt(12*NK*log(NKt))
#  prnt is a tag to print some info to console, including mu and testing checkpoints
#  checkpointing prints an iteration count every "checkpointing" steps, or no printing if None
#  plotfile is the file to save the plot too, if None then show the plot
def test_MAB(n=None, k=None, mu=None, T=800000, r=np.random.rand, prnt=False, nsw1=True, nsw2=True, nsw3=True, checkpointing=5000, plotfile=None):
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
        NSWs_1, p_hat_1, NSW_max_1, best_p_1 = NSW.results(mu, T, checkpointing=checkpointing, C=0.5)
        if NSW_max_1 > NSW_star:
            NSW_star = NSW_max_1
            p_star = best_p_1
    # UCB from HMS21
    def alphaN(t):
        return n
    def alphaNotN(t):
        return np.sqrt(12 * n * k * np.log(n * k * t))
    if nsw2:
        if prnt:
            print(" Testing UCB_HMS21 w/ alpha = N")
        NSWs_2, p_hat_2, NSW_max_2, best_p_2 = NSW2.results(mu, T, alphaN, checkpointing=checkpointing, delt=0.0001, C=0.8)
        if NSW_max_2 > NSW_star:
            NSW_star = NSW_max_2
            p_star = best_p_2
    if nsw3:
        if prnt:
            print(" Testing UCB_HMS21 w/ alpha = sqrt(12NKlog(NKt))")
        NSWs_3, p_hat_3, NSW_max_3, best_p_3 = NSW2.results(mu, T, alphaNotN, checkpointing=checkpointing)
        if NSW_max_3 > NSW_star:
            NSW_star = NSW_max_3
            p_star = best_p_3
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
                plt.show()
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
            plt.show()
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
def test_MAB_many(n_iter, n, k, T_max, r=np.random.rand, csvfile='results.csv', plotfolder='', nsw3=True):
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
            plotfile = plotfolder + 'results_' + str(int(i)) + '_' + str(int(j)) + '_' + str(int(x)) + '.png'
            (NSWs1, pHat1, bestP1, iRegs1, cRegs1), (NSWs2, pHat2, bestP2, iRegs2, cRegs2), _, (NSWStar, PStar), mu = test_MAB(n=i, k=j, mu=None, T=T_max, r=r, prnt=True, checkpointing=50000, plotfile=plotfile, nsw3=False)
            writer.writerow(mu)
            writer.writerow(cRegs1)
            writer.writerow(pHat1)
            writer.writerow(cRegs2)
            writer.writerow(pHat2)
            writer.writerow(PStar)
            writer.writerow([NSWStar])
    write_file.close()


# Test to find the optimal parameters
# on a given (N,K), generate a mu and test the same function with different parameters
def test_MAB_stability(n, k, T_max, r=np.random.rand, csvfile='results.csv'):
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
    
    delt = 0.00016
    delts = []
    NSWs2 = []
    for j in range(4):
        print("--------- J:", j, "---------")
        delt = delt / 4
        delts.append(delt)
        NSWs_2, p_hat_2, NSW_max_2, best_p_2 = NSW2.results(mu, T_max, alphaN, checkpointing=50000, delt=delt)
        if NSW_max_2 > NSW_star:
            NSW_star = NSW_max_2
            p_star = best_p_2
        NSWs2.append(NSWs_2)
        print(len(NSWs2))
        print(len(NSWs2[j]))

    
    for j in range(4):
        cregs2 = []
        creg = 0
        cregs2.append(creg)
        for i in range(len(NSWs2[j])):
            creg = creg + (NSW_star - NSWs2[j][i])
            cregs2.append(creg)
        plt.plot(cregs2, label=('Delt = ' + str(delts[j])))
        plt.show()
        plt.clf()
    
    Cs = []
    NSWs1 = []
    for i in range(6):
        print("--------- I:", i, "---------")
        C = 1 + 0.2 * (i)
        Cs.append(C)
        NSWs_1, p_hat_1, NSW_max_1, best_p_1 = NSW.results(mu, T_max, checkpointing=50000, C=C)
        if NSW_max_1 > NSW_star:
            NSW_star = NSW_max_1
            p_star = best_p_1
        NSWs1.append(NSWs_1)
    
    for i in range(6):
        cregs1 = []
        creg = 0
        cregs1.append(creg)
        for j in range(len(NSWs1[i])):
            creg = creg + (NSW_star - NSWs1[i][j])
            cregs1.append(creg)
        print(" C = " + str(C), file=write_file)
        print("  T = 200,000: " + str(cregs1[200000]), file=write_file)
        print("  T = 400,000: " + str(cregs1[400000]), file=write_file)

        
    
# main
# example calls, right now it is set to test pairs but this can be switched
if __name__ == "__main__":
    test_MAB_pairs(5, [(80,8), (20,4), (4,2)], 500000, r=exp_large, plotfolder='pair_plots/')
    #test_MAB_stability(4, 4, 400020, r=all_large, csvfile='results.csv')
    #test_MAB_many(8, [32, 64], [2, 4, 8, 16], 500000, r=all_large, csvfile='results.csv', plotfolder = 'plots/', nsw3=False)
    #_, _, _, _ = test_MAB(n=25, k=10, r=all_large, T=250000, prnt=True)
    #print("Last p_hat:\n",p_hat)
    #print("Computed p_star:\n",p_star)
    #main()

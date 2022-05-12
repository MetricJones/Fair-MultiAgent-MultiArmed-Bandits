
import NSW
import NSW_HMS21 as NSW2
import pgd
import numpy as np
import matplotlib.pyplot as plt
import csv

T = 10000
DEBUG = False

def closer_to_1(size):
    return np.random.beta(0.85, 0.15, size=size)

def all_large(size):
    return np.random.uniform(0.75, 1, size=size)

#mu = np.array([[0.00114021, 1], [1, 0.54500379]])

def main(n=5, k=5, mu=None, nsw1=True, nsw2=True):
    if mu is None:
        mu = np.random.rand(n,k)
    if DEBUG:
        print("mu is:\n", mu)
    # scale
    mu_max = np.max(mu, axis=1)
    for i in range(n):
        mu[i,:] = mu[i,:] / mu_max[i]
    # We have mu, let's run the algorithm
    NSWs, p_hat = NSW.results(mu, T)
    print("Last p_hat: ", p_hat)
    # Get the optimal
    p_star = NSW.max_NSW(mu, init_guess=p_hat, n=n, k=k, delt=0.0000001)
    print("p_star: ", p_star)
    NSW_star = NSW.NSW(mu, p_star, n=n, k=k)
    print("NSW_star: ",NSW_star)
    inst_regrets = NSW_star - NSWs
    inst_regrets = np.insert(inst_regrets, 0, 0)
    cumul = 0
    cumul_regrets = [0]
    for t in range(1,len(inst_regrets)):
        cumul += inst_regrets[t]
        cumul_regrets.append(cumul)
    plt.plot(inst_regrets, label='Instant regret')
    plt.plot(cumul_regrets, label='Cumul. Regret')
    plt.legend()
    plt.show()


def test_MAB(n=None, k=None, mu=None, T=800000, r=np.random.rand, prnt=False, nsw1=True, nsw2=True, nsw3=True, checkpointing=5000, plotfile=None):
    NSWs_1, p_hat_1, NSWs_2, p_hat_2 = None, None, None, None
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
    mu_max = np.max(mu, axis=1)
    for i in range(n):
        mu[i,:] = mu[i,:] / mu_max[i]
    if prnt:
        print(" mu:\n", mu)
    # Get an initial test
    p_star = NSW.max_NSW(mu, init_guess=([1/k] * k), n=n, k=k, delt=0.000000001)
    NSW_star = NSW.NSW(mu, p_star, n=n, k=k)
    # Run FMAB UCB
    if nsw1:
        if prnt:
            print(" Testing UCB_JNN")
        NSWs_1, p_hat_1, NSW_max_1, best_p_1 = NSW.results(mu, T, checkpointing=checkpointing)
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
        NSWs_2, p_hat_2, NSW_max_2, best_p_2 = NSW2.results(mu, T, alphaN, checkpointing=checkpointing)
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
        return (NSWs_1, p_hat_1, best_p_1, i_regs_1, cumul_regs_1), (NSWs_2, p_hat_2, best_p_2, i_regs_2, cumul_regs_2), (NSWs_3, p_hat_3, best_p_3, i_regs_3, cumul_regs_3), (NSW_star, p_star), mu


def test_MAB_many(n_iter, n, k, T_max, r=np.random.rand, csvfile='results.csv', plotfolder=''):
    write_file = open(csvfile, 'w', newline='')
    writer = csv.writer(write_file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['TestData'])
    writer.writerow(['n', 'k', 'n_iter'])
    writer.writerow([n, k, n_iter])
    for i in n:
        for j in k:
            for x in range(n_iter):
                print('n = ', i, ', k = ', j, ', iteration ', x)
                writer.writerow([i, j, x])
                # run tests
                plotfile = plotfolder + 'results_' + str(int(i)) + '_' + str(int(j)) + '_' + str(int(x)) + '.png'
                (NSWs1, pHat1, bestP1, iRegs1, cRegs1), (NSWs2, pHat2, bestP2, iRegs2, cRegs2), (NSWs3, pHat3, bestP3, iRegs3, cRegs3), (NSWStar, PStar), mu = test_MAB(n=i, k=j, mu=None, T=T_max, r=r, prnt=True, checkpointing=None, plotfile=plotfile)
                writer.writerow(mu)
                writer.writerow(cRegs1)
                writer.writerow(pHat1)
                writer.writerow(cRegs2)
                writer.writerow(pHat2)
                writer.writerow(cRegs3)
                writer.writerow(pHat3)
                writer.writerow(PStar)
                writer.writerow([NSWStar])
    close(write_file)
        
            

    
# main
if __name__ == "__main__":
    test_MAB_many(4, [2,4], [2,4], 50000, r=all_large, csvfile='test.csv', plotfolder = 'plots/')
    #_, _, _, _ = test_MAB(n=25, k=10, r=all_large, T=250000, prnt=True)
    #print("Last p_hat:\n",p_hat)
    #print("Computed p_star:\n",p_star)
    #main()

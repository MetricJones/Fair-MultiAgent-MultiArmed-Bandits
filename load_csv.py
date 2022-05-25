import pickle as pkl
import numpy as np
from merge_csv import TestDataItem

# Open and write the pkl file

if __name__=="__main__":
    f = open('all_tests_comp.pkl', 'rb')
    read_csvs = pkl.load(f)
    f.close()
    print("Read pickle")

    # compute statistics
    n_prev = 0
    k_prev = 0
    avg200000_1 = []
    avg500000_1 = []
    avg200000_2 = []
    avg500000_2 = []
    nsw_weights = []
    count = 0
    wins1_200000 = 0
    wins1_500000 = 0
    j = 0
    while j < len(read_csvs):
        #print("J: ", j)
        i = read_csvs[j]
        if n_prev == i.n and k_prev == i.k:
            avg200000_1.append(i.cregs1[10000])
            avg500000_1.append(i.cregs1[24998])
            avg200000_2.append(i.cregs2[10000])
            avg500000_2.append(i.cregs2[24998])
            nsw_weights.append(i.nsw_star)
            if i.cregs1[10000] < i.cregs2[10000]:
                wins1_200000 = wins1_200000 + 1
            if i.cregs1[24998] < i.cregs2[24998]:
                wins1_500000 = wins1_500000 + 1
            count = count + 1
            j = j + 1
        else:
            if count != 0: # reset and continue
                print("N =", n_prev, "K =", k_prev, "Count = ", count)
                print("200,000: ", (wins1_200000), " better JNN")
                print("500,000: ", (wins1_500000), " better JNN")
                print(" Unweighted:")
                print("  Our alg, T=200000: ", np.mean(avg200000_1), np.std(avg200000_1))
                print("  Our alg, T=500000: ", np.mean(avg500000_1), np.std(avg500000_1))
                print("  HMS alg, T=200000: ", np.mean(avg200000_2), np.std(avg200000_2))
                print("  HMS alg, T=500000: ", np.mean(avg500000_2), np.std(avg500000_2))
                nsw_weights = np.array(nsw_weights)
                avg200000_1 = avg200000_1 / nsw_weights
                avg500000_1 = avg500000_1 / nsw_weights
                avg200000_2 = avg200000_2 / nsw_weights
                avg500000_2 = avg500000_2 / nsw_weights
                print(" Weighted:")
                print("  Our alg, T=200000: ", np.mean(avg200000_1), np.std(avg200000_1))
                print("  Our alg, T=500000: ", np.mean(avg500000_1), np.std(avg500000_1))
                print("  HMS alg, T=200000: ", np.mean(avg200000_2), np.std(avg200000_2))
                print("  HMS alg, T=500000: ", np.mean(avg500000_2), np.std(avg500000_2))
                print(" NSW*:", np.mean(nsw_weights), np.std(nsw_weights))
                # reset
            n_prev = i.n
            k_prev = i.k
            avg200000_1 = []
            avg500000_1 = []
            avg200000_2 = []
            avg500000_2 = []
            nsw_weights = []
            count = 0
            wins1_200000 = 0
            wins1_500000 = 0
    # final check
    if count != 0: # reset and continue
        print("N =", n_prev, "K =", k_prev, "Count =", count)
        print("200,000: ", (wins1_200000), " better JNN")
        print("500,000: ", (wins1_500000), " better JNN")
        print(" Unweighted:")
        print("  Our alg, T=200000: ", np.mean(avg200000_1), np.std(avg200000_1))
        print("  Our alg, T=500000: ", np.mean(avg500000_1), np.std(avg500000_1))
        print("  HMS alg, T=200000: ", np.mean(avg200000_2), np.std(avg200000_2))
        print("  HMS alg, T=500000: ", np.mean(avg500000_2), np.std(avg500000_2))
        nsw_weights = np.array(nsw_weights)
        avg200000_1 = avg200000_1 / nsw_weights
        avg500000_1 = avg500000_1 / nsw_weights
        avg200000_2 = avg200000_2 / nsw_weights
        avg500000_2 = avg500000_2 / nsw_weights
        print(" Weighted:")
        print("  Our alg, T=200000: ", np.mean(avg200000_1), np.std(avg200000_1))
        print("  Our alg, T=500000: ", np.mean(avg500000_1), np.std(avg500000_1))
        print("  HMS alg, T=200000: ", np.mean(avg200000_2), np.std(avg200000_2))
        print("  HMS alg, T=500000: ", np.mean(avg500000_2), np.std(avg500000_2))
        print(" NSW*:", np.mean(nsw_weights), np.std(nsw_weights))

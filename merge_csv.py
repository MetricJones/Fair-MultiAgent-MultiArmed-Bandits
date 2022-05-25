import csv
import sys
import numpy as np
import pickle as pkl

# Turns the csv into a pickle

csv.field_size_limit(sys.maxsize)


digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']

fileLocation = "Fair-MultiAgent-MultiArmed-Bandits-main/"
filesList = []
has3List = [True, True, False, False, False]
for i in ["first", "second", "third", "fifth", "sixth"]:
    filesList.append("results_" + i + ".csv")

# substitute
# This is the current input to the file.
#  fileLocation is the folder with the files
#  filesList is the list of file names
#  has3List is a list of booleans parallel to filesList, which signals whether
#   the third algorithm (alpha != N) is in the tests for the file
fileLocation = ''
filesList = ["pair_results.csv"]
has3List = [False]


# a class to hold a single test
class TestDataItem:
    def __init__(self, n, k, itNum, mu, nsw_star, cregs1, cregs2, cregs3=None):
        self.n = n
        self.k = k
        self.itNum = itNum
        self.mu = mu
        self.cregs1 = cregs1
        self.cregs2 = cregs2
        self.cregs3 = cregs3
        self.nsw_star = nsw_star

    def lt(self, other):
        if self.n != other.n:
            return self.n < other.n
        elif self.k != other.k:
            return self.k < other.k
        elif self.itNum != other.itNum:
            return self.itNum < other.itNum
        else:
            return self.nsw_star < self.nsw_star
            # sort on iteration number
        

    def has3(self):
        return (self.cregs3 is not None)

# Helper function to shorten a list
def compress_list(l, factor=20):
    if len(l) > 25:
        return l[0::factor]

# ------ Main parse functions --------
def parse_mu(mu_str, n, k, l=10):
    checkpoint = 0
    r = np.zeros((n,k))
    for i in range(n):
        lft = mu_str.find('[', checkpoint)
        rgt = mu_str.find(']', lft)
        checkpoint = rgt
        mu_row = mu_str[lft + 1:rgt]
        mu_idx = 0
        for j in range(k):
            mu_idx = mu_row.find('.', mu_idx) - 1
            if mu_row[mu_idx] == '1':
                r[i][j] = 1
            else:
                # find the right bound
                rb = mu_idx
                while rb < len(mu_row) and mu_row[rb] in digits:
                    rb = rb + 1
                r[i][j] = float(mu_row[mu_idx : rb])
            mu_idx = mu_idx + 2
    return r


# parses a file, collecting the 
def parse_file(f, has3=False, factor=20):
    f = open(f, newline='')
    reader = f.readlines()
    rows = []
    row_i = 0
    n_rows = len(reader)
    # get first line
    row = reader[row_i]
    print(row.strip())
    row_i = row_i + 1
    row = reader[row_i]
    print(row.strip())
    row_i = row_i + 2
    # keep getting data
    c = 0
    while(True):
        # get the indices for the values
        c = c + 1
        row_idx = row_i
        row_mu = row_i + 1
        row_cregs1 = row_mu + 1
        if row_cregs1 >= n_rows:
            break
        while(reader[row_cregs1][:3] != '0.0'):
            row_cregs1 = row_cregs1 + 1
        row_cregs2 = row_cregs1 + 2
        if has3:
            row_cregs3 = row_cregs2 + 2
            row_star = row_cregs3 + 3
        else:
            row_star = row_cregs2 + 3
        row_i = row_star + 1
        # if all the indices are valid, get the values
        if row_i <= n_rows:
            nVal, kVal, itNum = reader[row_idx].split(' ')
            nVal = int(nVal)
            kVal = int(kVal)
            itNum = int(itNum)
            mu = parse_mu(''.join(reader[row_mu : row_cregs1]), nVal, kVal)
            cregs1 = reader[row_cregs1].split(' ')
            cregs1 = compress_list([float(x) for x in cregs1], factor=factor)
            cregs2 = reader[row_cregs2].split(' ')
            cregs2 = compress_list([float(x) for x in cregs2], factor=factor)
            if has3:
                cregs3 = reader[row_cregs3].split(' ')
                cregs3 = compress_list([float(x) for x in cregs3], factor=factor)
            else:
                cregs3 = None
            nsw_star = float(reader[row_star])
            row = TestDataItem(int(nVal), int(kVal), int(itNum), mu, nsw_star, cregs1, cregs2, cregs3)
        else:
            break
        rows.append(row)
    return rows
    
def read_iter():
    return None

if __name__ == "__main__":
    read_csvs_comp = []
    k = 0
    # read
    for i in filesList:
        print("Processing " + i)
        rows_comp = parse_file(fileLocation + i, has3=has3List[k], factor=20)
        read_csvs_comp.append(rows_comp)
        print("Processed " + i)
        k = k + 1
    comp_f = open('all_tests_comp.pkl', 'wb')
    # Merge
    csvs_comp = read_csvs_comp[0]
    for i in range(1, len(read_csvs_comp)):
        csvs_comp.extend(read_csvs_comp[i])
    # Sort
    print("L: ", len(csvs_comp))
    for i in range(len(csvs_comp)):
        j = i
        while j > 0 and (csvs_comp[j].lt(csvs_comp[j - 1])):
            tmp = csvs_comp[j]
            csvs_comp[j] = csvs_comp[j-1]
            csvs_comp[j-1] = tmp
            j = j - 1
    # remove duplicates
    final_comp = []
    i = 0
    j = 0
    n_prev = 0
    k_prev = 0
    it_prev = -1
    while i < len(csvs_comp):
        v = csvs_comp[i]
        if v.n == n_prev and v.k == k_prev and v.itNum == it_prev:
            # equality, remember the lower
            if v.lt(final_comp[-1]):
                final_comp[len(final_comp) - 1] = v
        else:
            # inequality, remember and update
            n_prev = v.n
            k_prev = v.k
            it_prev = v.itNum
            final_comp.append(v)
        i = i + 1
    print("M :", len(final_comp))
    pkl.dump(final_comp, comp_f)
    comp_f.close()
    print("Wrote pickle")
    # wrote pickle
    # now read pickle, for sanity
    f = open('all_tests_comp.pkl', 'rb')
    read_csvs = pkl.load(f)
    f.close()
    print("Read pickle")
    #print(pkl.loads(src))
    
    print (len(read_csvs))
    for j in len(read_csvs):
        print(j.n, j.k, j.itNum, j.nsw_star, len(j.cregs1))

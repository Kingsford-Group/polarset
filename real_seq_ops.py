'''
    Due to memory constraints all operations involving the actual sequence are here.
    This module contains code to: calculate energy surplus and deficit and calculate
    sequence energy over a random sequence.
'''
import random
from collections import deque
#  from tqdm import trange
trange = range
from collections import defaultdict
import pickle
chmap = {'A': 2, 'C': 0, 'G': 1, 'T': 3}
from anchor_distributed import working_dir
import cmath
from math import pi, floor
import multiprocessing
import matplotlib.pyplot as plt

def sequence_mer_iterator(k, seq):
    slen = len(seq)
    mod_low = 4 ** (k-1)
    cur = 0
    for i in range(k-1):
        cur = cur * 4 + chmap[seq[i]]
    for i in range(k-1, slen):
        if i >= k:
            cur -= mod_low * chmap[seq[i-k]]
        cur = cur * 4 + chmap[seq[i]]
        yield cur

def random_sequence(slen, seed = None):
    '''
    generates a random sequence.
    '''
    if seed is not None:
        random.seed(seed)
    return ''.join(random.choice('ACTG') for _ in range(slen))


class MinimizersImpl:
    '''
    base class for minimizers; this is actually a random minimizer.
    '''
    def __init__(self, w, k):
        self.w, self.k = w, k
        self.maxround = 2

    def kmer_level(self, kmer):
        '''
        Given a k-mer in integer representation return its level.
        '''
        return 0

    def stream_kmer_level(self, s):
        '''
        Given a string, return the level of its constituent k-mers.
        '''
        for km in sequence_mer_iterator(self.k, s):
            yield self.kmer_level(km)

class MykkMinimizer(MinimizersImpl):
    '''
    a parameterless class of minimizers, using the Mykkeltveit set as the
    priority set.
    the exact rules are:
        (1) if a k-mer is on the plus side of x-axis it is selected.
        (2) if a k-mer rotates from 4th quadrant to 1st, it is selected.
        (3) if a k-mer maps to the origin, it is selected if lexicographically smallest
            among its rotational equivalences.
    '''
    def __init__(self, w, k):
        super().__init__(w, k)
        self.offsets = []
        bbase = 2j * pi / k
        for i in range(k+1):
            self.offsets.append(cmath.exp(bbase * i))
        self.rot_multi = self.offsets[1]
        self.rev_rot_multi = self.offsets[k-1]

    def kmer_to_cplx(self, kmer):
        '''
        converts a k-mer to a complex number.
        @param kmer: integer representation of the kmer.
        @return: the complex number.
        '''
        ret = 0
        for i in reversed(range(self.k)):
            m = kmer % 4
            ret += m * self.offsets[i+1]
            kmer = kmer // 4
        assert kmer == 0
        return ret

    def check_rot_kmer(self, kmer):
        '''
        check if a k-mer is minimal among its rotational equivalents.
        @param kmer: integer representation of the kmer.
        @return: boolean indicating if it's rotationally minimal.
        '''
        cur = kmer
        submod = 4 ** (self.k - 1)
        for i in range(self.k):
            cd = cur % 4
            cur = cd * submod + cur // 4
            if cur > kmer:
                return False
        assert cur == kmer
        return True

    def kmer_level(self, kmer):
        '''
        Give a k-mer in integer representation, return its level.
        1: In Mykkeltveit set. 0: Not in the set.
        '''
        cc = self.kmer_to_cplx(kmer)
        if abs(cc) < 1e-6:
            return 1 if self.check_rot_kmer(kmer) else 0
        else:
            cr = cc * self.rot_multi
            #  print(kmer, cc, cr)
            if (cc.imag <= 0) and (cr.imag > 0):
                assert cc.real >= 0
                return 1
            else:
                return 0

    def _rot_sanity_check(self, kmer):
        '''
        A sanity check for rotational invariance.
        '''
        cur = kmer
        submod = 4 ** (self.k - 1)
        total_embed = 0
        count = 0
        for i in range(self.k):
            cd = cur % 4
            cur = cd * submod + cur // 4
            total_embed += self.kmer_to_cplx(cur)
            count += self.kmer_level(cur)
        print(kmer, total_embed, count)


class Miniception(MinimizersImpl):
    '''
    Implements the Miniception.
    the exact rules are:
        (1) there is a smaller minimizer (w0, k0), with w0+k0-1 = k.
        (2) for any k-mer, the smaller random minimizer is applied. if the
                first or the last k0-kmer is selected this k-mer is in the priority class.
    '''
    def __init__(self, w, k, k0):
        super().__init__(w, k)
        self.k0 = k0
        self.rand_multi = random.randrange(4 ** k0)

    def kmer_level(self, kmer):
        k, k0 = self.k, self.k0
        submod = 4 ** k0
        sub_kmers = []
        cur = kmer
        for i in range(k - k0 + 1):
            sub_kmers.append(((cur % submod) * self.rand_multi) % submod)
            cur = cur // 4
        ss = min(sub_kmers)
        if ss == sub_kmers[0]:
            return 1
        if ss == sub_kmers[-1]:
            return 1
        return 0

class LayeredAnchors(MinimizersImpl):
    '''
    loads a layered anchor set based minimizer from disk.
    '''
    def __init__(self, dump_file, seq):
        '''
            @param dump_file: link to the dump file.
            @param seq: the loaded sequence.
        '''
        with open(dump_file, 'br') as f:
            data = pickle.load(f)
        w, k, n, _, __, ___ = data["params"]
        super().__init__(w, k)
        self.w, self.n = w, n
        # due to some unintended side effects of using a suffix array
        assert n <= len(seq) - (k-1)
        assert n >= len(seq) - (k-1) - 30
        self.maxround = data["rounds"] + 2
        uniq_kmers = [0] * self.maxround
        kmers_occ = [0] * self.maxround
        self.dat = dict()
        print("generating dictionary.")
        kmer_prio = data["kmer_level"]
        it = sequence_mer_iterator(k, seq)
        _conflicts = 0
        for i in trange(n):
            km = next(it)
            pr = kmer_prio[i]
            if pr > 0:
                kmers_occ[pr] += 1
                if km in self.dat:
                    _conflicts += 1
                #  assert km not in self.dat
                self.dat[km] = pr
                uniq_kmers[pr] += 1
            elif pr == 0:
                if km in self.dat:
                    real_pr = self.dat[km]
                    if real_pr > 0:
                        kmers_occ[real_pr] += 1
        print("construction complete (w={}, k={}).".format(w, k), end='')
        for i in range(self.maxround):
            if uniq_kmers[i] > 0:
                print("level {}: {} kmers in {} locs;".format(i, uniq_kmers[i],
                                                              kmers_occ[i]),end=' ')
        print("total kmers =", n)
        if _conflicts > 0:
            print("[WARNING] Reentrant entries:", _conflicts)

    def kmer_level(self, km):
        d = self.dat.get(km, None)
        if d is None:
            return 0
        else:
            return self.maxround - d

def calc_energy_stats(seq, w, k):
    '''
    Calculates energy surplus and deficency for a given sequence and parameters w/k.
    '''
    n = len(seq) - (k-1)  # no. of k-mers
    it = sequence_mer_iterator(k, seq)
    val_count = dict()
    val_queue = deque()
    for i in range(w):
        dat = next(it)
        val_queue.append(dat)
        if dat not in val_count:
            val_count[dat] = 1
        else:
            val_count[dat] += 1
    esp, edf = 0, 0
    for i in trange(w, n):
        if i != w:
            td = val_queue.popleft()
            cc = val_count.pop(td)
            if cc > 1:
                val_count[td] = cc - 1
        dv = 1
        dat = next(it)
        val_queue.append(dat)
        if dat not in val_count:
            dv += 1
            val_count[dat] = 1
        else:
            val_count[dat] += 1
        ee = dv / len(val_count)
        esp += max(0, ee - 2 / (w+1))
        edf += max(0, 2 / (w+1) - ee)
    return esp / n, edf / n

def calc_energy(seq, mm : MinimizersImpl, anchor_sanity_check = False, ret_array = False):
    w, k = mm.w, mm.k
    n = len(seq) - (k-1)  # no. of k-mers
    km_it = sequence_mer_iterator(k, seq)
    prio_it = mm.stream_kmer_level(seq)
    val_count = dict()
    uniq_prio_count = dict()
    _total_prio = defaultdict(int)
    _dd_dist = defaultdict(int)
    highest_prio = -1
    val_queue = deque()
    prio_queue = deque()
    ret_vals = []
    for i in range(w):
        dat = next(km_it)
        prio = next(prio_it)
        _total_prio[prio] += 1
        val_queue.append(dat)
        prio_queue.append(prio)
        if dat not in val_count:
            val_count[dat] = 1
            highest_prio = max(highest_prio, prio)
            uniq_prio_count[prio] = uniq_prio_count.get(prio, 0) + 1
        else:
            val_count[dat] += 1
    ret = 0
    for i in trange(w, n):
        if i != w:
            last_km = val_queue.popleft()
            last_prio = prio_queue.popleft()
            kmc = val_count.pop(last_km)
            if kmc > 1:
                val_count[last_km] = kmc - 1
            else:  # a unique kmer is removed
                prc = uniq_prio_count.pop(last_prio)
                if prc > 1:
                    uniq_prio_count[last_prio] = prc - 1
                else:
                    highest_prio = max(uniq_prio_count.keys())
        dv = 0
        cur_km = next(km_it)
        cur_prio = next(prio_it)
        _total_prio[cur_prio] += 1
        val_queue.append(cur_km)
        prio_queue.append(cur_prio)
        if cur_km not in val_count:
            # unique k-mer; end location could count towards dv
            uniq_prio_count[cur_prio] = uniq_prio_count.get(cur_prio, 0) + 1
            highest_prio = max(highest_prio, cur_prio)
            if cur_prio == highest_prio:
                dv += 1  # if last element is highest prio and unique, it counts
        val_count[cur_km] = val_count.get(cur_km, 0) + 1
        if prio_queue[0] == highest_prio:
            dv += 1  # if first element is highest prio, it counts
        dd = uniq_prio_count[highest_prio]
        if anchor_sanity_check:
            if (highest_prio > 0) and (i < n - 30):
                if dd == 3:
                    print(prio_queue)
                    print(val_queue)
                    assert False
                else:
                    assert dd <= 2
        # debugging
        if False:
            _dv = 0
            max_prio = max(prio_queue)
            if prio_queue[0] == max_prio:
                _dv += 1
            if prio_queue[-1] == max_prio:
                if val_queue.count(val_queue[-1]) == 1:
                    _dv += 1
            _hp_vals = list(val_queue[x] for x in range(w+1) if (prio_queue[x] == max_prio))
            _dd = len(set(_hp_vals))
            _dd_dist[_dd] += 1
            assert _dv == dv
            assert _dd == dd

        ee = dv / dd
        if ret_array:
            ret_vals.append(ee)
        ret += ee
    print("[DEBUG] Prio Distribution:", _total_prio)
    print("[DEBUG] Uniq Distribution:", _dd_dist)
    if ret_array:
        return ret_vals
    else:
        return ret / n

def calc_selected_locs(seq, mm : MinimizersImpl, robust_windows = False, ret_array = False):
    w, k = mm.w, mm.k
    n = len(seq) - (k-1)
    modulus = 4 ** k
    order = []
    assert k <= 15
    print("generating random orderings")
    for i in trange(modulus):
        order.append(i)
        j = random.randrange(i+1)
        if j != i:
            order[i], order[j] = order[j], order[i]
    #  random.shuffle(order)
    print("list comprehension done")
    #  while (seed % 2) == 0:
        #  seed = random.randrange(modulus)  # ensures results are unique.
    km_it = sequence_mer_iterator(k, seq)
    prio_it = mm.stream_kmer_level(seq)
    def next_val():
        km = next(km_it)
        prio = next(prio_it)
        #  km_sh = (km * seed) % modulus
        km_sh = order[km]
        return km_sh - prio * modulus
    val_queue = deque()
    ret = 0
    ret_vals = []
    for i in range(w):
        val = next_val()
        val_queue.append(val)
    last_val = min(val_queue)
    last_dist = None
    last_count = 0
    for i in range(w):
        if val_queue[w - 1 - i] == last_val:
            if (last_dist is None) or (not robust_windows):
                last_dist = i
            last_count += 1
    for i in trange(w, n):
        if not robust_windows:
            # sanity check
            assert len(val_queue) == w
            test_val = min(val_queue)
            test_dist = None
            test_count = 0
            for j in range(w):
                if val_queue[w-1-j] == test_val:
                    test_dist = j
                    test_count += 1
            #  print((test_val, test_dist, test_count), (last_val, last_dist, last_count))
            assert (test_val, test_dist, test_count) == (last_val, last_dist, last_count)

        # new window, doesn't care about contexts
        last_dist += 1
        val = next_val()
        val_queue.append(val)
        pval = val_queue.popleft()
        new_selection = False
        if val == last_val:
            last_count += 1
        if val < last_val:  # new smallest k-mer at last window
            last_dist = 0
            last_val = val
            last_count = 1
            new_selection = True
        elif pval == last_val:  # popping a minimal k-mer
            last_count -= 1
            if last_count == 0:  # brand new minimal k-mer
                last_val = min(val_queue)
                last_dist = None
                last_count = 0
                for j in range(w):
                    if val_queue[w - j - 1] == last_val:
                        if (last_dist is None) or (not robust_windows):
                            last_dist = j
                        last_count += 1
                new_selection = True
            else:  # still the same minimal k-mer, now determine which k-mer to pick
                if last_dist == w:  # the k-mer selected is out of window
                    last_dist = None
                    for j in range(w):
                        if val_queue[w - j - 1] == last_val:
                            if (last_dist is None) or (not robust_windows):
                                last_dist = j
                    new_selection = True
                else:  # the k-mer selected is still in the window, nothing changes
                    assert last_dist < w
                    assert robust_windows
        else:  # no new smallest k-mer, nor
            pass
        ret += int(new_selection)
        if ret_array:
            ret_vals.append(int(new_selection))
    if ret_array:
        return ret_vals
    else:
        return ret / n

def proc_all_energy_stats():
    for seq_file in ["hg38_all", "chr1"]:
        with open(seq_file + ".seq") as f:
            seq = f.readline().strip()
        for w in [100, 10]:
            for k in range(15, 26):
                sp, df = calc_energy_stats(seq, w, k)
                print("Finished: {}-{}-{} DF=+{:.5f} -{:.5f}".format(seq_file, w, k, sp*(w+1), df*(w+1)))
                with open(working_dir + "estats.dat", 'a') as f:
                    print('{},{},{},{},{}'.format(seq_file, w, k, sp, df), file=f)

def proc_rand_energy_stats():
    seq_file = "random"
    seq = random_sequence(5000000)
    for w in [100, 10]:
        for k in range(15, 26):
            sp, df = calc_energy_stats(seq, w, k)
            print("Finished: {}-{}-{} DF=+{:.5f} -{:.5f}".format(seq_file, w, k, sp*(w+1), df*(w+1)))
            with open(working_dir + "estats.dat", 'a') as f:
                print('{},{},{},{},{}'.format(seq_file, w, k, sp, df), file=f)


def proc_miniception(w, kmin, kmax):
    #  seq_file = "chr1"
    seq_file = "scramble"
    with open(seq_file + ".seq") as f:
        seq = f.readline().strip()
    print("sequence loaded:", seq_file, "starting work:", w, kmin, kmax)
    for k in range(kmin, kmax + 1):
        if w == 100:
            mm = Miniception(w, k, 5)
        else:
            mm = Miniception(w, k, k0=k-w)
        d = calc_energy(seq, mm)
        print("Finished: {}-{}-{} DF={:.5f}".format(seq_file, w, k, d*(w+1)))
        #  print("Finished: {}-{}-{} DF=+{:.5f} -{:.5f}".format(seq_file, w, k, sp*(w+1), df*(w+1)))
        with open(working_dir + "miniception.dat", 'a') as f:
            print('{},{},{},{}'.format(seq_file, w, k, d), file=f)

def _proc_miniception_parallel(item):
    proc_miniception(item[0], item[1], item[2])

def miniception_all():
    works = []
    #  for i in range(15, 26):
    for i in range(21, 26):
        works.append((10, i, i))
        works.append((100, i, i))
    with multiprocessing.Pool(processes=len(works), maxtasksperchild=1) as pool:
        pool.map(_proc_miniception_parallel, works)


def proc_random_seq(policy, seq_file, w, k, random_len):
    with open(seq_file + ".seq") as f:
        seq = f.readline().strip()
    idx = "{}_{}_{}_{}".format(policy,seq_file,w,k)
    dump_file = working_dir + idx + ".dump"
    print(idx, "Loading anchor sets")
    mm = LayeredAnchors(dump_file, seq)
    print(idx, "Minimizer constructed")
    rs = random_sequence(random_len)
    print(idx, "Calculating")
    df = calc_energy(rs, mm)
    print(idx, "random density factor: {:.5f}".format(df * (w+1)))
    with open(working_dir + "random_results.dat", "a") as f:
        print("{},{},{},{},{}".format(idx, seq_file, w, k, df), file=f)

def _proc_random_seq(item):
    try:
        proc_random_seq(item[0], item[1], item[2], item[3], 5000000)
    except Exception as e:
        import sys
        import traceback as tb
        ce, ct, ctb = sys.exc_info()
        print(item[0], "Exception thrown:", repr(e))
        tb.print_exception(ce, ct, ctb)
        return 0

def random_seq_parse():
    works = []
    for w in [10, 100]:
        for k in range(15, 26):
            for policy in ["baseline", "multi_lax"]:
                works.append((policy, "hg38_all", w, k))

    with multiprocessing.Pool(processes=6, maxtasksperchild=1) as pool:
        pool.map(_proc_random_seq, works)
    #  for item in works:
        #  _proc_random_seq(item)

def chrX_centomere_test(policies, w, kvals):
    #  plt.figure(figsize=(20, 6))
    with open("chrX.seq") as f:
        seq = f.readline().strip()
        ct_seq = seq[57828561:60934693]
    for k in kvals:
        plt.clf()
        for policy in policies:
            if policy == "random":
                mm = MinimizersImpl(w, k)
            else:
                dump_file = working_dir + "{}_{}_{}_{}.dump".format(policy, "chrX", w, k)
                mm = LayeredAnchors(dump_file, seq)
            ss = calc_selected_locs(ct_seq, mm, robust_windows=True,ret_array=True)
            ss2 = calc_selected_locs(ct_seq, mm, robust_windows=False,ret_array=True)
            ces = calc_energy(ct_seq, mm, ret_array=True)
            split = 100000
            xvals = []
            yvals = []
            for i in range(len(ces) // split):
                xvals.append(i * split)
                yvals.append(sum(ces[i*split:(i+1)*split]) / split * (w+1))
            yvals2 = []
            for i in range(len(ss) // split):
                yvals2.append(sum(ss[i*split:(i+1)*split]) / split * (w+1))
            yvals3 = []
            for i in range(len(ss2) // split):
                yvals3.append(sum(ss2[i*split:(i+1)*split]) / split * (w+1))
            plt.plot(xvals, yvals, linestyle="solid", marker='o', markersize=3, label=policy)
            plt.plot(xvals, yvals2, linestyle="solid", marker='o', markersize=3, label=policy + "_robust")
            plt.plot(xvals, yvals3, linestyle="solid", marker='o', markersize=3, label=policy + "_control")
            print("done: {} for w={}, k={}".format(policy, w, k))
        plt.legend()
        plt.savefig("figures/byseq/{}_{}_{}.pdf".format("chrX", w, k))
        print("figure saved for w={}, k={}".format(w, k))
        #  df = calc_energy(ct_seq, mm)
        #  cidx = "{}_{}_{}_{}".format(policy, "chrX_ct", w, k)
        #  print(cidx, "density factor: {:.5f}".format(df * (w+1)))


if __name__ == "__main__":
    #  random_seq_parse()
    #  proc_rand_energy_stats()
    miniception_all()
    #  chrX_centomere_test(["multi_lax", "baseline-nf", "random"], 100, list(range(15, 16)))
    #  chrX_centomere_test(["random"], 100, list(range(15, 16)))

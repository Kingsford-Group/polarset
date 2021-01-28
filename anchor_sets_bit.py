# these are re-implementations of CoverageChecker and KMerChain that
# uses bitvectors (array in Python) to improve performance.

from collections import defaultdict
from anchor_sets import CoverageChecker, KMerChain
from array import array
import numpy as np
import pickle
import logging
from tqdm import trange
import argparse
#  from multiprocessing import shared_memory
working_dir = "/home/hongyuz/data/anchorsets/"
kc_preload_dir = working_dir + "preload/"
_global_sorted_idx = dict()
_global_lookup_table = dict()
_global_heights = dict()


def val_iter(n, val = 0):
    '''
    A helper function that yields N zeroes.
    @param n: number of zeroes.
    '''
    for i in range(n): yield val


class BitCoverageChecker(CoverageChecker):
    '''
    This class supercedes CoverageChecker and uses a bit vector to store set of
    current/commited locations.
    NOTE: This works for the layered anchor set only.
    NOTE: The labels are disabled.
    NOTE: The caching is disabled for now.
    '''

    @classmethod
    def empty_cells(self, len):
        return None  # this will be implemented separately.


    def __init__(self, n, w):
        '''
        The new initializer.
        '''
        super().__init__(n, w)
        assert self.w <= 100
        self._need_exbit = (self.w > 50)
        # commited stores boundary only - signed bits
        self.commited_lo = array('b', val_iter(self.num_blocks, -1))
        self.commited_hi = array('b', val_iter(self.num_blocks, -1))
        # current stores bit vector - emulated by two unsigned LLs
        # commited are also stored here for the sake of clarity..?
        self.commited = array('Q', val_iter(self.num_blocks * 2))
        self.current = array('Q', val_iter(self.num_blocks * 2))

    def _verify_timestamp(self, b):
        if self._cur_ts[b] < self._current_time:
            self._cur_ts[b] = self._current_time
            self.current[b*2] = 0
            if self._need_exbit:
                self.current[b*2+1] = 0

    def check_commited(self, x):
        '''
        Check the commited locations for whether the location is covered.
        @param x: the location to check.
        @return: (leftmost closest element, rightmost closest element) if these
            are within w units; Otherwise return self.PH_MIN, self.PH_MAX as
            placeholders
        NOTE: This is not exact at times; However it is only wrong (that is,
        the return values are not exactly the closest elements) when it is
        guaranteed that the location is covered by commited locations already
        (that is, verify returns C_COVERED)
        '''
        if x == self._com_cached_x:
            return self._com_cached_res
        b = x // self.w  # current block.
        rb = self.PH_MAX
        lb = self.PH_MIN
        if self.commited_lo[b] != -1:  # nonempty block
            for offset in [self.commited_lo[b], self.commited_hi[b]]:
                y = offset + b * self.w
                if y >= x: rb = min(rb, y)
                if y <= x: lb = max(lb, y)
        if (b > 0) and (lb == self.PH_MIN):
            if self.commited_lo[b-1] != -1:
                y = self.commited_hi[b-1] + (b-1) * self.w
                if y >= x - self.w: lb = y
        if rb == self.PH_MAX:
            if self.commited_lo[b+1] != -1:
                y = self.commited_lo[b+1] + (b+1) * self.w
                if y <= x + self.w: rb = y
        self._com_cached_x = x
        self._com_cached_res = lb, rb
        return lb, rb

    def _location_in_block_iter(self, b, for_commited = False):
        '''
        Helper function to iterate over list of locations in a block.
        @param b: index of the block.
        @param for_commited: If looking for commited locations instead.
        @return: yields list of locations within this block, with offests added back.
        '''
        base = b * self.w
        if for_commited:
            ll = [self.commited[b*2]]
            if self._need_exbit:
                ll.append(self.commited[b*2+1])
        else:
            self._verify_timestamp(b)
            ll = [self.current[b*2]]
            if self._need_exbit:
                ll.append(self.current[b*2+1])
        for v in ll[:]:
            while v > 0:
                hs = v.bit_length() - 1
                yield base + hs
                v -= 1 << hs
            base += 50  # this handles the base shift from the second location

    def get_covered_list_from_current(self, x, tags = True):
        '''
        Get the set of (location, tag) that is within w units of current location.
        NOTE: Now tag is always None.
        @param x: the location to check.
        @param tags: if tags are appended. This is for legacy compatibility.
        @return: list of (location, tag) within w units of current location.
        '''
        assert not tags
        b = x // self.w  # current block.
        ret = []
        self._verify_timestamp(b)
        for e in self._location_in_block_iter(b):
            ret.append(e)
        if b > 0:
            self._verify_timestamp(b-1)
            for e in self._location_in_block_iter(b-1):
                if e >= x - self.w: ret.append(e)
        self._verify_timestamp(b+1)
        for e in self._location_in_block_iter(b+1):
            if e <= x + self.w: ret.append(e)
        return ret

    def _add_to_counter(self, x, label = None):
        b = x // self.w
        assert self._cur_ts[b] == self._current_time
        base_val = x - b * self.w
        if base_val >= 50:
            assert base_val < 100
            vv = 1 << (base_val - 50)
            assert (self.current[b*2+1] & vv) == 0
            self.current[b*2+1] += vv
        else:
            assert base_val >= 0
            vv = 1 << base_val
            assert (self.current[b*2] & vv) == 0
            self.current[b*2] += vv

    def _rem_from_counter(self, x):
        b = x // self.w
        assert self._cur_ts[b] == self._current_time
        base_val = x - b * self.w
        if base_val >= 50:
            assert base_val < 100
            vv = 1 << (base_val - 50)
            assert (self.current[b*2+1] & vv) == vv
            self.current[b*2+1] -= vv
        else:
            assert base_val >= 0
            vv = 1 << base_val
            assert (self.current[b*2] & vv) == vv
            self.current[b*2] -= vv

    def commit_all(self):
        '''
        Commit everything in current and start over.
        '''
        for b in range(self.num_blocks):
            self._verify_timestamp(b)
            for v in self._location_in_block_iter(b):
                orig = v - b * self.w
                if self.commited_lo[b] == -1:
                    self.commited_lo[b] = orig
                    self.commited_hi[b] = orig
                else:
                    self.commited_lo[b] = min(self.commited_lo[b], orig)
                    self.commited_hi[b] = max(self.commited_hi[b], orig)
                if orig >= 50:
                    vv = 1 << (orig - 50)
                    assert (self.commited[b*2+1] & vv) == 0
                    self.commited[b*2+1] += vv
                else:
                    vv = 1 << orig
                    assert (self.commited[b*2] & vv) == 0
                    self.commited[b*2] += vv
        self.commited_ele += self.cur_ele
        self.commited_segs += self.cur_segs
        self.commited_cover += self.cur_cover
        self.start_over()

    def _get_all_locations(self, commited_only = False):
        '''
        Profile-only function.
        '''
        for i in range(self.num_blocks):
            l = list(self._location_in_block_iter(i, True))
            if len(l) > 0:
                assert self.commited_hi[i] == max(l) - i * self.w
                assert self.commited_lo[i] == min(l) - i * self.w
            if not commited_only:
                l.extend(list(self._location_in_block_iter(i, False)))
            l.sort()
            for x in l:
                yield x

def sequence_mer_list(k, seq):
    chmap = {'A': 2, 'C': 0, 'G': 1, 'T': 3}
    slen = len(seq)
    modulus = 4 ** k
    cur = 0
    ret = [0] * (slen - k + 1)
    for i in range(k-1):
        cur = cur * 4 + chmap[seq[i]]
    for i in trange(k-1, slen):
        cur = (cur * 4 + chmap[seq[i]]) % modulus
        ret[i-(k-1)] = cur
    return ret


class CompactKMerChain(KMerChain):
    '''
    Re-implementation of KMerChain class, now also with the original kmer
    iterator built in the same file.
    Update: screw that, this is literally a suffix array now.
    '''
    JUMP_CAP = 30000
    def __init__(self, buf_prefix, n, k):
        '''
        Initialization from a suffix array dump.
        The suffix array must have been loaded into shared memory.
        @param buf_prefix: prefix of buffers.
        @param n: length of data, returned by load_shared_partial_SA.
        @param k: the value of k used.
        '''
        super().__init__([])
        logging.info("Loading started")
        self.n, self.k = n, k
        self.name = buf_prefix
        assert self.k <= 30
        #  self.sh_sidx = shared_memory.SharedMemory(name=buf_prefix+"_sidx", size=4*n)
        #  self.sh_lookup = shared_memory.SharedMemory(name=buf_prefix+"_lookup", size=4*n)
        #  self.sh_heights = shared_memory.SharedMemory(name=buf_prefix+"_heights", size=n)
        #  self.sorted_idx = np.frombuffer(self.sh_sidx.buf, dtype=np.uint32)
        #  self.lookup_table = np.frombuffer(self.sh_lookup.buf, dtype=np.uint32)
        #  self.heights = np.frombuffer(self.sh_heights.buf, dtype=np.uint8)
        self.sorted_idx, self.lookup_table, self.heights = \
            _global_sorted_idx[buf_prefix], _global_lookup_table[buf_prefix], _global_heights[buf_prefix]
        logging.info("SA loaded, generating jump tables")
        self.jump_table = array('I', self.heights)
        self.jump_table[0] = 0
        last_jval = 0
        self.kmer_freq = defaultdict(int)
        for i in range(1, self.n):
            if self.heights[i - 1] >= self.k:
                last_jval += 1
            else:
                self.kmer_freq[last_jval + 1] += 1
                last_jval = 0
            self.jump_table[i] = min(self.JUMP_CAP, last_jval)
        self.kmer_freq[last_jval + 1] += 1
        logging.info("Jump tables loaded")

    def calc_freq_cutoff(self, p):
        '''
        Calculates cutoff so certain portion of k-mers are included.
        @param p: 0<p<1 the portion of k-mers to be included.
        @return: the cutoff value.
        '''
        cur = 0
        assert p < 1
        vals = sorted(list(self.kmer_freq.items()))
        for k, v in vals:
            cur += k * v
            if cur >= self.n * p:
                return k
        assert False

    @property
    def data_len(self):
        return self.n

    def iter_by_idx(self, idx, is_repr = False):
        '''
        Returns list of sequence indexes that share the same k-mer.
        @param idx: the index of the k-mer.
        @param jump_first: helper that yields the jump table value first.
                            this serves as a quick check if a k-mer is too repetitive.
                            THIS IS DEPRECATED FOR NOW
        @param is_repr: if this is already the "representative element".
                            if this is set to True, it is assumed we don't need to
                            consult the jump table.
        @return: the set of indexes that share the same k-mer (incl. input).
        '''
        sa_idx = self.lookup_table[idx]
        lb = sa_idx
        if not is_repr:
            ljump = self.jump_table[lb]
            lb = lb - ljump
            if ljump == self.JUMP_CAP:
                while (self.jump_table[lb] != 0):
                    lb -= self.jump_table[lb]
        yield self.sorted_idx[lb]
        rb = lb
        while (rb < self.n - 1) and (self.heights[rb] >= self.k):
            rb += 1
            #  print(self.sorted_idx[rb], rb, self.heights[rb], self.jump_table[rb])
            yield self.sorted_idx[rb]

    # since now we're representing k-mers by their indexes only...
    iter_by_value = iter_by_idx
    def single_value_to_loc(self, val):
        return val

    def __getitem__(self, idx):
        '''
        Instead of the actual k-mer values, this function return a "representation"
        that is, the first index in the suffix array with the same k-mer.
        @param idx: the index of the k-mer.
        @return: the "representation".
        '''
        sa_idx = self.lookup_table[idx]
        lb = sa_idx
        ljump = self.jump_table[lb]
        lb = lb - ljump
        if ljump == self.JUMP_CAP:
            while (self.jump_table[lb] != 0):
                lb -= self.jump_table[lb]
        return self.sorted_idx[lb]

    def get_real_kmers(self, idx):
        '''
        Gets the actual k-mer, in its usual form.
        Not implemented, for now.
        '''
        assert False


def coverage_checker_test_new():
    '''
    Initial testing of the CoverageChecker class.
    '''
    c = BitCoverageChecker(100, 10)
    c.add_loc(0)
    c.add_loc(5)
    c.add_loc(17)
    c.add_loc(30)
    c.add_loc(40)
    c.add_loc(46)
    c.add_loc(53)
    c._show_statistics()
    c.verify_stats()
    for x in [1, 4, 10, 23, 30, 35, 45, 60, 70]:
        print(x, c.verify(x))
    c.commit_all()
    c._show_statistics()
    for x in [1, 4, 10, 23, 30, 35, 45, 60, 70]:
        print(x, c.verify(x))
    c.add_loc(23)
    c.add_loc(64)
    c.add_loc(99)
    c.verify_stats()
    c.verify_stats(True)
    for x in [1, 4, 10, 23, 30, 35, 45, 60, 70]:
        print(x, c.verify(x))
    c._show_statistics()
    c.delete_loc(23)
    c.delete_loc(64)
    c.delete_loc(99)
    c._show_statistics()
    c.verify_stats()
    c.verify_stats(True)

def coverage_checker_test_2():
    '''
    edge case testing.
    '''
    c = BitCoverageChecker(100, 10)
    c.add_loc(0)
    c._show_statistics()
    c.add_loc(10)
    c._show_statistics()
    c.add_loc(5)
    c._show_statistics()
    c.delete_loc(10)
    c.delete_loc(5)
    c._show_statistics()
    c.verify_stats()

#  def prepare_kc(seq_file, k):
    #  '''
    #  Construct the KMerChain file and write to disk (to check if this thing will
    #  be faster by any means, and how large it really is).
    #  '''
    #  with open(seq_file + ".seq") as f:
        #  seq = f.readline().strip()
    #  logging.info("Sequence loaded")
    #  dat = sequence_mer_list(k, seq)
    #  logging.info("KMer list generated")
    #  ch = CompactKMerChain(dat)
    #  logging.info("CKC object constructed")
    #  ch.save(kc_preload_dir + "{}_{}.dat".format(seq_file, k))
    #  logging.info("Serialized to disk")
    #  return ch

def preprocess_partial_SA(seq_file, k, dump_file = None):
    '''
    Call this function to preprocess (partial) suffix arrays and do all the
    dirty work in advance.
    '''
    assert k < 31
    with open(seq_file + ".seq") as f:
        seq = f.readline().strip()
    assert len(seq) < (1 << 32)  # just enough for hg38
    logging.info("Sequence loaded")
    dat = sequence_mer_list(k, seq)
    n = len(dat)
    logging.info("KMer list generated")
    sorted_idx_ = np.argsort(dat)
    logging.info("Argsort completed")
    _ph_array = [0] * n
    sorted_idx = array('L', sorted_idx_)
    heights = array('B', _ph_array)
    lookup_table = array('L', _ph_array)
    # build lookup table now
    logging.info("Memory allocated for arrays")
    for i in trange(n):
        lookup_table[sorted_idx[i]] = i
    logging.info("Reverse lookup table calculated")
    for i in trange(n-1):
        # calculate height[x]: this is the longest common prefix of
        # suffix at sorted_idx[i] and sorted_idx[i+1].
        idx0 = sorted_idx[i]
        idx1 = sorted_idx[i+1]
        if dat[idx0] == dat[idx1]:  # k-mer agrees
            heights[i] = k
        else:  # iterate now
            for d in range(k):
                if seq[idx0+d] != seq[idx1+d]:
                    heights[i] = d
                    break
    data_tuple = (n, k, sorted_idx, lookup_table, heights)
    logging.info("Height calculated")
    if dump_file is None:
        dump_file = kc_preload_dir + seq_file + "_" + str(k) + ".dump"
    with open(dump_file, "bw") as f:
        pickle.dump(data_tuple, f, protocol=4)
    logging.info("Completed")

def load_shared_partial_SA(dump_file, buf_prefix):
    '''
    Loads the partial suffix array to memory via mp.shared_memory.
    @param dump_file: dump file generated by preprocess_partial_SA.
    @param buf_prefix: prefix of buffer file.
    @return: value of N to be feed to subsequent uses.
    '''
    with open(dump_file, "br") as f:
        n, _max_k, sorted_idx, lookup_table, heights = pickle.load(f)
    assert _max_k >= 30

    #  sh_sidx = shared_memory.SharedMemory(name=buf_prefix+"_sidx", create=True,
                                         #  size=4*n)
    #  sh_lookup = shared_memory.SharedMemory(name=buf_prefix+"_lookup", create=True,
                                           #  size=4*n)
    #  sh_heights = shared_memory.SharedMemory(name=buf_prefix+"_heights", create=True,
                                            #  size=n)
    #  np_sidx = np.frombuffer(sh_sidx.buf, dtype=np.uint32)
    #  np_lookup = np.frombuffer(sh_lookup.buf, dtype=np.uint32)
    #  np_heights = np.frombuffer(sh_heights.buf, dtype=np.uint8)
    #  assert len(np_sidx) == n
    #  assert len(np_heights) == n
    #  np_sidx[:] = sorted_idx[:]
    #  np_lookup[:] = lookup_table[:]
    #  np_heights[:] = heights[:]
    _global_heights[buf_prefix] = heights
    _global_lookup_table[buf_prefix] = lookup_table
    _global_sorted_idx[buf_prefix] = sorted_idx
    return n

def unload_shared_partial_SA(buf_prefix, n):
    '''
    Unloads the partial suffix arrays.
    @param buf_prefix: prefix of buffer file.
    '''
    pass
    #  sh_sidx = shared_memory.SharedMemory(name=buf_prefix+"_sidx", size=4*n)
    #  sh_lookup = shared_memory.SharedMemory(name=buf_prefix+"_lookup", size=4*n)
    #  sh_heights = shared_memory.SharedMemory(name=buf_prefix+"_heights", size=n)
    #  sh_sidx.close()
    #  sh_lookup.close()
    #  sh_heights.close()
    #  sh_sidx.unlink()
    #  sh_lookup.unlink()
    #  sh_heights.unlink()

def partial_SA_test(buf_prefix, k):
    logging.basicConfig(format="%(asctime)s %(message)s",datefmt="%I:%M:%S",
                    level=logging.DEBUG)
    dump_file = kc_preload_dir + buf_prefix + "_30.dump"
    n = load_shared_partial_SA(dump_file, buf_prefix)
    logging.info("SA loading finished")
    for k in range(15, 25):
        kc = CompactKMerChain(buf_prefix, n, k)
        rets = []
        for idx in range(1, 20):
            p = idx / 20
            rets.append((p, kc.calc_freq_cutoff(p)))
        print(k, rets)
    quit()
    height_dist = defaultdict(int)
    for i in trange(n-1):
        height_dist[kc.heights[i]] += 1
        if kc.heights[i] < k:
            assert kc[kc.sorted_idx[i]] != kc[kc.sorted_idx[i+1]]
        else:
            assert kc[kc.sorted_idx[i]] == kc[kc.sorted_idx[i+1]]
            x = kc.iter_by_idx(kc.sorted_idx[i])
            y = kc.iter_by_idx(kc.sorted_idx[i+1])
            for i in range(20):
                a = next(x, None)
                b = next(y, None)
                if a is None:
                    break
                assert a == b

def preprocess_main():
    '''
    this is main process for preprocessing. No need for multiprocessing.
    '''
    logging.basicConfig(format="%(asctime)s %(message)s",datefmt="%I:%M:%S",
                    level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument("fn")
    args = parser.parse_args()
    logging.info("Parameters: {}".format(args))
    preprocess_partial_SA(args.fn, 30)
    logging.info("Preprocessing finished, now doing sanity test")
    buf_prefix = args.fn
    dump_file = kc_preload_dir + args.fn + "_30.dump"
    n = load_shared_partial_SA(dump_file, buf_prefix)
    logging.info("SA loading finished, length = {}".format(n))
    partial_SA_test(buf_prefix, 18)
    #  unload_shared_partial_SA(buf_prefix, n)
    logging.info("Sanity test finished")


if __name__ == "__main__":
    #  coverage_checker_test_new()
    #  c = BitCoverageChecker(len(ch.data), 10)
    #  partial_SA_test("hg38_all", None)
    #  coverage_checker_test_2()
    #  unload_shared_partial_SA(buf_prefix, n)
    preprocess_main()
    pass

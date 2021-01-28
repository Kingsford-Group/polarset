# demo code for anchor sets: infrastructure
#  from context_sampler import sequence_mer_iterator
from collections import deque
#  from functools import lru_cache
import logging
from collections import defaultdict
import array
seq_file = "chr1.seq"
trange = range

def zero_iterator(n):
    '''
    A helper function that yields N zeroes.
    @param n: number of zeroes.
    '''
    for i in range(n): yield 0

class CoverageChecker:
    '''
    This class manages tracking of sequence coverage, and of collisions.
    It is (naively) implemented by dividing the sequence into small blocks.
    THIS WORKS FOR THE LAYERED ANCHOR SET ONLY
    IT IS NOT SUPPOSED TO WORK WITH FURTHER ITERATIONS
    '''
    def empty_cells(self, len):
        ret = []
        for i in range(len): ret.append([])
        return ret

    def __init__(self, n, w):
        '''
        Initializer.
        @param n: len of sequence.
        @param w: length of window.
        '''
        self.n, self.w = n, w
        self.num_blocks = n // w + 3
        self.commited = self.empty_cells(self.num_blocks)
        self.current = self.empty_cells(self.num_blocks)
        self._current_time = 0
        self._cur_ts = array.array('i', zero_iterator(self.num_blocks))
        self._cur_ts = [0] * self.num_blocks
        assert self.n < self.PH_MAX / 2
        self.commited_ele = 0
        self.commited_cover = 0
        self.commited_segs = 0
        self.cur_ele = 0
        self.cur_cover = 0
        self.cur_segs = 0
        self._com_cached_x = self.PH_MAX
        self._com_cached_res = None
        self._cur_cached_x = self.PH_MAX
        self._cur_cached_res = None

    PH_MIN = -100000000000
    PH_MAX =  100000000000
    C_OK = 0
    C_COVERED = 1
    C_DENIED_HARD = 2
    C_DENIED_SOFT = 3

    def _verify_timestamp(self, b):
        '''
        Helper function to make sure self.current is not storing outdated info.
        @param b: the index of the block.
        '''
        if self._cur_ts[b] < self._current_time:
            self._cur_ts[b] = self._current_time
            self.current[b] = []

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
        if len(self.commited[b]) > 0:
            for y in self.commited[b]:
                if y >= x: rb = min(rb, y)
                if y <= x: lb = max(lb, y)
        if (b > 0) and (lb == self.PH_MIN):
            if len(self.commited[b-1]) > 0:
                y = self.commited[b-1][-1]
                if y >= x - self.w: lb = y
        if rb == self.PH_MAX:
            if len(self.commited[b+1]) > 0:
                y = self.commited[b+1][0]
                if y <= x + self.w: rb = y
        self._com_cached_x = x
        self._com_cached_res = lb, rb
        return lb, rb

    def get_covered_list_from_current(self, x, tags = True):
        '''
        Get the set of (location, tag) that is within w units of current location.
        @param x: the location to check.
        @param tags: if tags are also attached to output.
        @return: list of (location, tag) within w units of current location.
        '''
        b = x // self.w  # current block.
        ret = []
        self._verify_timestamp(b)
        for e in self.current[b]:
            ret.append(e)
        if b > 0:
            self._verify_timestamp(b-1)
            for e in self.current[b-1]:
                if e[0] >= x - self.w: ret.append(e)
        self._verify_timestamp(b+1)
        for e in self.current[b+1]:
            if e[0] <= x + self.w: ret.append(e)
        if not tags:
            ret = list(c[0] for c in ret)
        return ret

    def check_current(self, x):
        '''
        Check the current locations for whether the location is covered.
        @param x: the location to check.
        @return: (leftmost closest element, rightmost closest element).
        '''
        if x == self._cur_cached_x:
            return self._cur_cached_res
        ls = self.get_covered_list_from_current(x, tags = False)
        rb = self.PH_MAX
        lb = self.PH_MIN
        for e in ls:
            if e >= x: rb = min(rb, e)
            if e <= x: lb = max(lb, e)
        self._cur_cached_x = x
        self._cur_cached_res = lb, rb
        return lb, rb

    #  def _invalidate_lru_caches(self):
        #  '''
        #  helper function to invalidate all LRU caches.
        #  '''
        #  self.check_commited.cache_clear()
        #  self.get_covered_list_from_current.cache_clear()
        #  self.check_current.cache_clear()

    def _invalidate_caches(self):
        self._com_cached_x = self.PH_MAX
        self._cur_cached_x = self.PH_MAX

    def verify(self, x):
        '''
        Check the status of a location.
        @param x: the location (0-based, start-of-kmer) to check.
        @return: either C_OK, C_DENIED or C_COVERED:
            C_COVERED: This location is covered by previous round loc.
                            (Highest precedence)
            C_OK: This location is not within w/2 bases of a selected location (cur/prev).
            C_DENIED_HARD: This location is within w/2 bases of a commited location
                            (overrides C_DENIED_SOFT)
            C_DENIED_SOFT: This location is within w/2 bases of a current location.
        '''
        # WARNING: THIS FUNCTION IS BEING DEPRECATED. USE WITH CAUTION.
        # AT THE VERY LATEST, FIX ROUNDING PROBLEM
        #  logging.warning("Using deprecated function: CoverageChecker.verify")
        #  assert False
        cur_l, cur_r = self.check_current(x)
        com_l, com_r = self.check_commited(x)
        #  print("[DEBUG] CUR {}~{} COM {}~{}".format(cur_l, cur_r, com_l, com_r))
        #  if cur_l == x:
            #  return self.C_DENIED
        if com_r - com_l <= self.w:
            return self.C_COVERED
        if (x - com_l <= self.w // 2) or (com_r - x <= self.w // 2):
            return self.C_DENIED_HARD
        if (x - cur_l <= self.w // 2) or (cur_r - x <= self.w // 2):
            return self.C_DENIED_SOFT
        return self.C_OK

    def check_all(self, x):
        '''
        Check all locations (commited and current) for whether the location is covered.
        @param x: the location.
        @return: (leftmost closest element, rightmost closest element).
        '''
        cur_l, cur_r = self.check_current(x)
        com_l, com_r = self.check_commited(x)
        return max(cur_l, com_l), min(cur_r, com_r)

    def _add_to_counter(self, x, label = None):
        '''
        Auxiliary function that actually modifies the underlying data structure.
        '''
        b = x // self.w
        self.current[b].append((x, label))

    def add_loc(self, x, label = None):
        '''
        Add a location to the current set.
        @param x: the location.
        @param label: some auxiliary information
        @return: change of energy from this move.
        '''
        com_l, com_r = self.check_commited(x)
        if com_r - com_l <= self.w:
            return 0
        cur_l, cur_r = self.check_current(x)
        lb, rb = max(cur_l, com_l), min(cur_r, com_r)
        # determine change in cover / segement / length
        dcov, dseg, dele = self.w + 1, 1, 1
        if lb != self.PH_MIN:
            dseg -= 1
            dcov -= self.w + 1 - (x - lb)
        if rb != self.PH_MAX:
            if rb - lb > self.w: dseg -= 1
            dcov = max(0, dcov - (self.w + 1 - (rb - x)))
        self._add_to_counter(x, label)
        self._invalidate_caches()
        self.cur_ele += dele
        self.cur_segs += dseg
        self.cur_cover += dcov
        return 2 * dcov - (self.w + 1) * (dele + dseg)

    def _rem_from_counter(self, x):
        '''
        Auxiliary function that actuall modifies the underlying data structure.
        '''
        b = x // self.w
        flag = False
        assert self._cur_ts[b] == self._current_time
        for ele in self.current[b]:
            if ele[0] == x:
                self.current[b].remove(ele)
                flag = True
                break
        if not flag:
            print(x, b, self.current[b])
        assert flag

    def delete_loc(self, x):
        '''
        Remove a location from the current set.
        @param x: the location.
        @return: change of energy from this move.
        '''
        com_l, com_r = self.check_commited(x)
        if com_r - com_l <= self.w:
            return 0
        #  logging.debug("DEL {}".format(x))
        self._rem_from_counter(x)
        self._invalidate_caches()
        cur_l, cur_r = self.check_current(x)
        lb, rb = max(cur_l, com_l), min(cur_r, com_r)
        # reversing changes
        dcov, dseg, dele = self.w + 1, 1, 1
        if lb != self.PH_MIN:
            dseg -= 1
            dcov -= self.w + 1 - (x - lb)
        if rb != self.PH_MAX:
            if rb - lb > self.w: dseg -= 1
            dcov = max(0, dcov - (self.w + 1 - (rb - x)))
        # dcov <= 0 only if rb - lb <= w
        self.cur_ele -= dele
        self.cur_segs -= dseg
        self.cur_cover -= dcov
        return (self.w + 1) * (dele + dseg) - 2 * dcov

    def commit_all(self):
        '''
        Commit everything in current and start over.
        '''
        for b in range(self.num_blocks):
            self._verify_timestamp(b)
            if len(self.current[b]) > 0:
                self.commited[b].extend(x[0] for x in self.current[b])
                self.commited[b].sort()
        self.commited_ele += self.cur_ele
        self.commited_segs += self.cur_segs
        self.commited_cover += self.cur_cover
        self.start_over()

    def start_over(self):
        '''
        Erase all records in current.
        '''
        self._invalidate_caches()
        self.cur_ele = 0
        self.cur_segs = 0
        self.cur_cover = 0
        self._current_time += 1

    def _show_statistics(self):
        '''
        Debugging function: show statistics of current data.
        '''
        print("[COM] Coverage = {:10d} Elements = {:10d} Segments = {:10d}".format(
            self.commited_cover, self.commited_ele, self.commited_segs))
        print("[ALL] Coverage = {:10d} Elements = {:10d} Segments = {:10d}".format(
            self.commited_cover + self.cur_cover,
            self.commited_ele + self.cur_ele, self.commited_segs + self.cur_segs))

    def total_score(self):
        '''
        Returns total savings against random minimizer, measured by # expected picks.
        Note: all scores are scaled by (w+1) (basically, density factor).
        '''
        return 2 * (self.cur_cover + self.commited_cover) - \
                (self.w + 1) * (self.cur_ele + self.cur_segs + self.commited_ele + self.commited_segs)

    def commited_score(self):
        '''
        Returns total savings against random minimizer, for the commited elements only.
        '''
        return 2 * self.commited_cover - (self.w + 1) * (self.commited_ele + self.commited_segs)

    def _get_all_locations(self, commited_only = False):
        '''
        Probably for profiling only, return list of all selected locations in
        increasing order.
        @param commited_only: If only commited elements are counted.
        @return: This is an iterator, yielding selected elements in increasing order.
        '''
        for i in range(self.num_blocks):
            l = self.commited[i][:]
            if len(self.current[i]) > 0:
                self._verify_timestamp(i)
                l.extend(x[0] for x in self.current[i])
            l.sort()
            for x in l:
                yield x

    def gap_dist(self, commited_only = False):
        '''
        Function for profiling.
        @param commited_only: If only commited elements are counted.
        @return: a dictionary of {length: count} for the distance between adjacent
                selected locations.
        '''
        ret = defaultdict(int)
        last_loc = self.PH_MIN
        for x in self._get_all_locations(commited_only):
            if last_loc != self.PH_MIN:
                ret[x - last_loc] += 1
            last_loc = x
        return ret

    def segment_dist(self, commited_only = False):
        '''
        Function for profiling.
        @param commited_only: If only commited elements are counted.
        @return: a dictionary of {length: count} for the length of unbroken segments
                as in the current set of selected locations.
        '''
        ret = defaultdict(int)
        it = self._get_all_locations(commited_only)
        x0 = next(it)
        last_loc = x0
        last_start = x0
        w = self.w
        for x in it:
            if (x - last_loc) > w:
                ret[last_loc + w - last_start] += 1
                last_start = x
            last_loc = x
        ret[last_loc + w - last_start] += 1
        return ret

    def verify_stats(self, commited_only = False):
        '''
        Function for debugging.
        @param commited_only: If only commited elements are counted.
        '''
        it = self._get_all_locations(commited_only)
        x0 = next(it, None)
        if x0 is None:
            print("[DEBUG] List is empty.")
            return
        last_loc = x0
        last_start = x0
        act_cov, act_ele, act_segs = 0, 1, 0
        w = self.w
        for x in it:
            act_ele += 1
            if (x - last_loc) > w:
                act_cov += last_loc - last_start + w + 1
                act_segs += 1
                last_start = x
            last_loc = x
        act_cov += last_loc - last_start + w + 1
        act_segs += 1
        tar_cov, tar_segs, tar_ele = self.commited_cover, self.commited_segs, self.commited_ele
        if not commited_only:
            tar_cov += self.cur_cover
            tar_segs += self.cur_segs
            tar_ele += self.cur_ele
        print("[DEBUG] Verification ({}): Cover {}->{}, Segments {}->{}, Elements {}->{}".format(
            "Commited" if commited_only else "All", tar_cov, act_cov, tar_segs, act_segs, tar_ele, act_ele))


    def uncovered_segs_dist(self, commited_only = False):
        '''
        Function for profiling.
        @param commited_only: If only commited elements are counted.
        @return: a dictionary of {length: count} for the length of unbroken segments
                that are not covered by current set of locations.
        '''
        ret = defaultdict(int)
        it = self._get_all_locations(commited_only)
        w = self.w
        last_loc = 0
        for x in it:
            if (x - last_loc) > self.w:
                ret[x - last_loc - w] += 1
            last_loc = x
        if self.n - last_loc > self.w:
            ret[self.n - last_loc - w] += 1
        return ret

    def window_hit_dist(self, window_length = None, commited_only = False):
        '''
        Function for profiling: This function iterates over every window, and
        count the number of hits within each window.
        @param window_length: length of the window. Defaults to self.w.
        @param commited_only: If only commited elements are counted.
        @return: a dictionary of {X : count}, for the number of windows that
            has exactly X selected k-mers within.
        '''
        ret = defaultdict(int)
        cl = [self.PH_MIN]
        for x in self._get_all_locations(commited_only):
            assert x > cl[-1]
            cl.append(x)
        cl.append(self.PH_MAX)
        lidx = 0
        ridx = 0
        w = self.w if window_length is None else window_length
        print("[Temp Debug] Calculating: windows hitting distribution")
        for i in trange(self.n):
            while cl[lidx] <= i - w: lidx += 1
            while cl[ridx] <= i: ridx += 1
            ret[ridx - lidx] += 1
        return ret

class KMerChain:
    '''
    A helper class that maintains a set of doubly linked lists, facilitating quick
    listing of all locations of a k-mer.
    '''
    PH = -1
    def __init__(self, data):
        self.data = data
        self.n = len(data)
        self.pre = [self.PH] * self.n
        self.nxt = [self.PH] * self.n
        self.end_loc = dict()
        self.count_by_val = defaultdict(int)
        for i in range(self.n):
            v = data[i]
            if v in self.end_loc:
                p = self.end_loc[v]
                self.nxt[p] = i
                self.pre[i] = p
            self.end_loc[v] = i
            self.count_by_val[v] += 1

    def calc_freq_cutoff(self, p):
        '''
        Calculates cutoff so certain portion of k-mers are included.
        @param p: 0<p<1 the portion of k-mers to be included.
        @return: the cutoff value.
        '''

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def data_len(self):
        return len(self.data)

    def _next_iter_by_idx(self, idx):
        c = self.nxt[idx]
        while c != self.PH:
            yield c
            c = self.nxt[c]

    def _prev_iter_by_idx(self, idx):
        c = self.pre[idx]
        while c != self.PH:
            yield c
            c = self.pre[c]

    def iter_by_idx(self, idx):
        yield idx
        for x in self._next_iter_by_idx(idx):
            yield x
        for x in self._prev_iter_by_idx(idx):
            yield x

    def iter_by_value(self, val):
        for x in self.iter_by_idx(self.end_loc[val]):
            yield x

    def single_value_to_loc(self, val):
        return self.end_loc[val]

def unique_kmer_profiling(seq, w, k):
    '''
    Initial code for unique k-mer profiling.
    '''
    assert False  # sequence_mer_iterator is now out of scope
    sequence_mer_iterator = lambda x, y: []
    seen = set()
    dupe = set()
    loc = [0] * w
    ridx = 0
    siter = sequence_mer_iterator(k, seq)
    for mer in siter:
        if mer in seen:
            dupe.add(mer)
        else:
            seen.add(mer)
    print("seen {} kmers, {} dupes".format(len(seen), len(dupe)))
    siter = sequence_mer_iterator(k, seq)
    for mer in siter:
        if mer not in dupe:
            loc[ridx % w] += 1
        ridx += 1
    print(ridx / w)
    print(loc)

def coverage_checker_test():
    '''
    Initial testing of the CoverageChecker class.
    '''
    c = CoverageChecker(100, 10)
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
    for x in [1, 4, 10, 23, 30, 35, 45, 60, 70]:
        print(x, c.verify(x))
    c._show_statistics()
    c.verify_stats()
    c.verify_stats(True)
    c.delete_loc(23)
    c.delete_loc(64)
    c.delete_loc(99)
    c._show_statistics()
    c.verify_stats()
    c.verify_stats(True)


def kmer_chain_test():
    data = [1,2,3,4,1,4,2,3,4,5,6,2,3,1,5,3,2,4,5,1,1,1,1,1]
    k = KMerChain(data)
    for x in k.iter_by_idx(1): print(x, end=" ,")
    print()
    for x in k.iter_by_value(1): print(x, end=' .')


if __name__ == "__main__":
    #  with open(seq_file) as f:
        #  seq = f.readline().strip()
    # unique_kmer_profiling(seq, 50, 16)
    from tqdm import trange
    coverage_checker_test()
    kmer_chain_test()


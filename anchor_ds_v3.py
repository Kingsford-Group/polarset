'''
Hopefully final iteration on implement CoverageChecker.
'''
from anchor_sets_bit import val_iter, CompactKMerChain
from anchor_strat_bit import CompactAnchorStrat, CoverageChecker
from array import array
import bitarray
import logging
#  from IPython import embed

class CoverageCheckerV3(CoverageChecker):
    '''
    This class supercedes CoverageChecker and by using a smaller window,
    ensures that only one value (commited / current) exists in one window.
    '''
    @classmethod
    def empty_cells(self, len):
        return None  # this is to prevent creation of long arrays.

    def __init__(self, n, w):
        '''
        Initializer.
        @param n: length of sequence.
        @param w: length of window.
        '''
        assert w <= 100
        assert (w % 2) == 0  # make things slightly easier
        super().__init__(n, w // 2)
        self.w = w
        self.hw = w // 2
        self.commited = array('b', val_iter(self.num_blocks, -1))
        self.current = array('b', val_iter(self.num_blocks, -1))

    def _verify_timestamp(self, b):
        '''
        Now returns self.current[b].
        '''
        if self._cur_ts[b] < self._current_time:
            self._cur_ts[b] = self._current_time
            self.current[b] = -1
            return -1
        else:
            return self.current[b]

    def check_loc(self, x, commited):
        '''
        An iterator that checks if a location is present.
        @param x: starting location.
        @param commited: switch for iteration over commited (True) or current (False).
        '''
        b = x // self.hw
        if commited:
            return self.commited[b] + b * self.hw == x  # always fail if unoccupied
        else:
            self._verify_timestamp(b)
            return self.current[b] + b * self.hw == x

    def forward_iter(self, x, dist_limit, commited, include_self = False):
        '''
        An iterator that iterates over current locations, starting from x.
        @param x: starting location.
        @param dist_limit: (relative) distance limit before stopping, ends inclusive.
        @param commited: switch for iteration over commited (True) or current (False).
        @param include_self: if the location itself is included in return.
        '''
        b = x // self.hw
        offset = b * self.hw
        if commited:
            tv = self.commited[b]
        else:
            tv = self._verify_timestamp(b)
        if tv != -1:
            tloc = offset + tv
            if (tloc > x) and (tloc <= x + dist_limit):
                yield tloc
            if include_self:
                if tloc == x:
                    yield tloc
        b += 1
        offset += self.hw
        while (b < self.num_blocks) and (offset <= x + dist_limit):
            if commited:
                tv = self.commited[b]
            else:
                tv = self._verify_timestamp(b)
            if tv != -1:
                if offset + tv <= x + dist_limit:
                    yield offset + tv
            b += 1
            offset += self.hw

    def backward_iter(self, x, dist_limit, commited):
        '''
        An iterator that iterates over current locations, starting from x.
        @param x: starting location.
        @param dist_limit: (relative) distance limit before stopping, ends inclusive.
        @param commited: switch for iteration over commited (True) or current (False).
        '''
        b = x // self.hw
        offset = b * self.hw
        if commited:
            tv = self.commited[b]
        else:
            tv = self._verify_timestamp(b)
        if tv != -1:
            tloc = offset + tv
            if (tloc < x) and (tloc >= x - dist_limit):
                yield tloc
        b -= 1
        offset -= self.hw
        while (b >= 0) and (offset + self.hw - 1 >= x - dist_limit):
            if commited:
                tv = self.commited[b]
            else:
                tv = self._verify_timestamp(b)
            if tv != -1:
                if offset + tv >= x - dist_limit:
                    yield offset + tv
            b -= 1
            offset -= self.hw

    def check_commited(self, x):
        '''
        Check the commited locaations and return leftmost and rightmost closest
        element within w units.
        '''
        if x == self._com_cached_x:
            return self._com_cached_res
        fw_iter = self.forward_iter(x, self.w, True)
        bw_iter = self.backward_iter(x, self.w, True)
        lb, rb = next(bw_iter, self.PH_MIN), next(fw_iter, self.PH_MAX)
        self._com_cached_x = x
        self._com_cached_res = lb, rb
        return lb, rb

    def get_covered_list_from_current(self, x, tags = True, dist_override = None):
        '''
        Get the set of (location, tag) that is within w units of current location.
        @param x: the location to check.
        @param tags: if tags are attached to output.
        @param dist_override: if we are checking within this range instead.
        @return: list of (location, tag) within w units of current location.
        '''
        assert not tags
        ret = []
        dw = self.w if dist_override is None else dist_override
        for z in self.forward_iter(x, dw, False, True):
            ret.append(z)
        for z in self.backward_iter(x, dw, False):
            ret.append(z)
        return ret

    def check_current(self, x):
        '''
        Check the current locations for whether the location is covered.
        @param x: the location to check.
        @return: (leftmost closest element, rightmost closest element).
        '''
        if x == self._cur_cached_x:
            return self._cur_cached_res
        fw_iter = self.forward_iter(x, self.w, False)
        bw_iter = self.backward_iter(x, self.w, False)
        lb, rb = next(bw_iter, self.PH_MIN), next(fw_iter, self.PH_MAX)
        self._cur_cached_x = x
        self._cur_cached_res = lb, rb
        return lb, rb

    def _add_to_counter(self, x, label = None):
        b = x // self.hw
        assert self._verify_timestamp(b) == -1
        self.current[b] = x - b * self.hw

    def _rem_from_counter(self, x):
        b = x // self.hw
        assert self._verify_timestamp(b) == x - b * self.hw
        self.current[b] = -1

    def commit_all(self):
        '''
        Commit everything in current and start over.
        '''
        for b in range(self.num_blocks):
            if self._cur_ts[b] == self._current_time:
                # need to assume covered elements don't get added
                ccom, ccur = self.commited[b], self.current[b]
                assert (ccom == -1) or (ccur == -1)
                self.commited[b] = max(ccom, ccur)
        self.commited_ele += self.cur_ele
        self.commited_segs += self.cur_segs
        self.commited_cover += self.cur_cover
        self.start_over()

    def _get_all_locations(self, commited_only = False):
        '''
        Return list of all selected locations in increasing order.
        @param commited_only: If only commited elements are counted.
        @return: Iterates over selected elements in increasing order.
        '''
        icom = self.forward_iter(0, self.n, commited=True, include_self=True)
        if commited_only:
            icur = val_iter(1, self.PH_MAX)
        else:
            icur = self.forward_iter(0, self.n, commited=False, include_self=True)
        com_next = next(icom, self.PH_MAX)
        cur_next = next(icur, self.PH_MAX)
        while min(com_next, cur_next) < self.PH_MAX:
            if com_next < cur_next:
                yield com_next
                com_next = next(icom, self.PH_MAX)
            elif com_next > cur_next:
                yield cur_next
                cur_next = next(icur, self.PH_MAX)
            else:
                assert False # this is impossible

class AnchorStrategyV3(CompactAnchorStrat):
    '''
        A new iteration of AnchorStrat that better handles the validation process.
        More specifically, better memorization of covered locations; more "precise"
        validation process that ignores covered locations; adaptive choice of
        occurrence limits using freq_threshold.
    '''
    def __init__(self, kc : CompactKMerChain, w, k, occ_limit = None, dist_limit = None):
        '''
        @param skip_cover: If covered locations should count in validation or not.
        '''
        if occ_limit is None:
            occ_limit = kc.calc_freq_cutoff(0.98)
        super().__init__(kc, w, k, occ_limit, dist_limit, cc_cls = CoverageCheckerV3)
        self.covered = bitarray.bitarray(self.n)
        self.covered.setall(0)
        self._val_ts = array('b', val_iter(self.n))
        self._val_time = 0

    def check_loc_cover(self, l):
        if self.covered[l]:
            return True
        com_l, com_r = self.covc.check_commited(l)
        if com_r - com_l <= self.w:
            self.covered[l] = True
            return True
        else:
            return False

    def get_occurence_by_loc(self, loc, is_repr = False):
        '''
        Get a list of occurrences of a specific k-mer, with covered locations
        removed.
        WARNING: In this version no verification is done any more.
        @param loc: the location of the k-mer to be indexed.
        @param is_repr: If this is already the "representative element" for KMerChain.
        @return: an iterator, yielding locations.
        '''
        for x in self.kc.iter_by_idx(loc, is_repr):
            if not self.check_loc_cover(x):
                yield x

    def filter_locs(self, start, stop = None, gap = None):
        '''
        Perform filtering through a list of locations.
        @param start, stop, gap: parameters of the range iterator.
        @return: filtered list.
        '''
        ret = []
        occ_limit = self.occ_limit
        dist_limit = self.cur_dist_limit
        self._val_time += 1
        if stop is None:
            stop = self.n
        if gap is None:
            gap = self.w
        for idx in range(start, stop, gap):
            vval = self._val_ts[idx]
            if vval == self._val_time:  # passed already
                ret.append(idx)
            elif vval != 0 - self._val_time:  # have not failed already
                dat = self.kc[idx]  # this is to filter out frequent k-mers
                if self._val_ts[dat] == 0 - self._val_time:
                    continue
                if not self.check_loc_cover(idx):
                    clocs = []
                    flag = True
                    for x in self.get_occurence_by_loc(dat, is_repr=True):
                        clocs.append(x)
                        if len(clocs) == occ_limit:
                            flag = False
                            break
                    if flag:
                        clocs.sort()
                        for ix, x in enumerate(clocs):
                            if ix > 0:
                                if x - clocs[ix-1] <= dist_limit:
                                    flag = False
                                    continue
                    else:
                        self._val_ts[dat] = 0 - self._val_time
                        continue
                    if flag:
                        ret.append(idx)
                    nval = self._val_time if flag else (0 - self._val_time)
                    for x in clocs:
                        if ((x - start) % gap) == 0:
                            self._val_ts[x] = nval
        return ret

    def add_kmer_by_loc(self, loc, tag = None, is_repr_and_valid = False):
        '''
        Adds a k-mer to the current set of selection, indexed by its location.
        @param loc: the location of the k-mer to be added.
        @param tag: the tag provided for this group.
        @param is_repr_and_valid:
        @return: change of score from this operation. If the operation is not
                    successful, 0 is returned.
        '''
        ret = 0
        if not is_repr_and_valid:
            dat = self.kc[loc]
            if self.kmer_level[dat] > 0:
                return ret
        else:
            dat = loc
        for l in self.get_occurence_by_loc(dat, is_repr = True):
            ret += self.covc.add_loc(l, tag)
        self.kmer_level[dat] = self.round_id
        return ret

    def del_kmer_by_loc(self, loc, count = False, is_repr_and_valid = False):
        '''
        Removes a k-mers from the current set of selection, indexed by its location.
        @param loc: the location of the k-mer to be removed.
        @param count: whether a count should be returned alongside change in score.
        @param is_repr: whether loc is already a representation.
        @return: change of score from this operation. If the operation is not
                    successful, 0 is returned instead.
        '''
        ret = 0
        c = 0
        if not is_repr_and_valid:
            dat = self.kc[loc]
            if self.kmer_level[dat] <= 0:
                if count:
                    return 0, None
                else:
                    return 0
        else:
            dat = loc
        for l in self.get_occurence_by_loc(dat, is_repr = True):
            ret += self.covc.delete_loc(l)
            c += 1
        self.kmer_level[dat] = 0 - self.round_id
        if count:
            return ret, c
        else:
            return ret

    def multi_random_pass(self, offset = None, strat = None):
        '''
        Some more attempts at the multiple linear pass method.
        @param offset: Ignored.
        @param strat: currently, a single value for dist tolerance, by *percentage* of w.
                    TEST: if negative, skip the singleton purge.
        @return: the values obtained from calc_current_energy and calc_random_energy.
        '''
        singleton_flag = (strat < 0)  # >0 means purge singleton and is default behavior
        strat = abs(strat)
        total_rounds = 3
        gap_values = [self.w, self.w, self.w // 5]
        if self.n < 5000000:
            reps = [10, 10, 20]
        else:
            reps = [3, 3, 2]
        #  occ_tols = [min(200, self.kc.calc_freq_cutoff(0.85)),
                    #  min(500, self.kc.calc_freq_cutoff(0.9)),
                    #  min(500, self.kc.calc_freq_cutoff(0.9))]
        occ_tols = [1 + self.kc.calc_freq_cutoff(0.85), 1 + self.kc.calc_freq_cutoff(0.9),
                    1 + self.kc.calc_freq_cutoff(0.95)]
        logging.warning("WARNING - no upper limit on k-mer occurrences")
        if strat is None:
            dist_tols = [int(self.w * 0.6)] * total_rounds
        else:
            dist_tols = [int(self.w * strat)] * total_rounds
        logging.info("[MP] Postprocessing Finished.")

        for i in range(total_rounds):
            self._single_random_pass(gap_values[i], reps[i], occ_tols[i],
                                     dist_tols[i], monotone_mode=(gap_values[i] < self.w),
                                     keep_singleton=singleton_flag)
        c0 = self.calc_current_energy()
        c1 = self.calc_random_energy()
        logging.info("Energy Information: Current DF {:.4f}, Random DF {:.4f}".format(
            c0 * (self.w + 1), c1 * (self.w+1)))
        return c0, c1

def coverage_checker_test():
    '''
    Initial testing of the CoverageChecker class.
    '''
    c = CoverageCheckerV3(100, 10)
    #  embed()
    c.add_loc(0)
    c.add_loc(6)
    c.verify_stats()
    c.add_loc(17)
    c.add_loc(30)
    c.add_loc(40)
    c.add_loc(46)
    c.add_loc(53)
    print(list(c.forward_iter(0, 100, False)))
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


if __name__ == "__main__":
    coverage_checker_test()

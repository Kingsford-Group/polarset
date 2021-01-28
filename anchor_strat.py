# demo code for anochr chains: strategy
from anchor_sets import CoverageChecker, KMerChain
from functools import wraps
from collections import defaultdict, deque
from operator import itemgetter
import argparse
import logging
import random
import pickle
#  from IPython import embed
#  seq_file = "chr1.seq"
trange = range


def random_mer_generator(w, k):
    modulus = 4 ** k
    cur_kmer = random.randrange(modulus)
    for i in range(w):
        cur_kmer = (cur_kmer * 4 + random.randrange(4)) % modulus
        yield cur_kmer


def sequence_mer_iterator(k, seq):
    chmap = {'A': 0, 'C': 1, 'T': 2, 'G': 3}
    slen = len(seq)
    modulus = 4 ** k
    cur = 0
    for i in range(k-1):
        cur = cur * 4 + chmap[seq[i]]
    for i in range(k-1, slen):
        cur = (cur * 4 + chmap[seq[i]]) % modulus
        yield cur


def validate_locs_gen(limit, dist):
    '''
    A helper function to *generate* a function that validates:
        1) The number of elements in the iterator does not exceed limit
        2) The elements are not within (dist) of each other.
    @param limit: the limit on size of return sets.
    @param dist: the maximum distance that is not allowed.
    @return: a function with the following:
        @param iter: the iterator that generate the list.
        @return: if either condition is validated return None.
            else, return a sorted list of generated values.
    '''
    @wraps(validate_locs_gen)
    def func(iter_):
        l = limit
        ret = []
        for x in iter_:
            ret.append(x)
            l -= 1
            if l == 0:
                return None
        ret.sort()
        for idx, x in enumerate(ret):
            if idx > 0:
                if (x - ret[idx-1]) <= dist:
                    return None
        return ret
    return func


def validiator_test(data):
    '''
    some testing and profiling functions.
    '''
    kc = KMerChain(data)
    logging.info("KMerChain constructed")
    validator = validate_locs_gen(10, w // 2)
    validator2 = validate_locs_gen(1000, w // 2)
    c_yes = 0
    c_no = 0
    c_yes_1 = 0
    c_no_1 = 0
    for idx in trange(len(data)):
        l1 = validator(kc.iter_by_idx(idx))
        if l1 is not None:
            c_yes += 1
        else:
            c_no += 1
        l2 = validator2(kc.iter_by_idx(idx))
        if l2 is not None:
            c_yes_1 += 1
        else:
            c_no_1 += 1
    print(c_yes, c_no)
    print(c_yes_1, c_no_1)


class AnchorStrategy:
    '''
    A wrapper class for anchor set strategy.
    Provides common convenience functions interacting with other classes.
    '''
    def __init__(self, kc : KMerChain, w, k, occ_limit = 25, dist_limit = None):
        '''
        Initializes a AnchorStrategy class.
        The actual strategy is not passed as parameters at this time, but
        rather as function calls.
        @param data: list of k-mers.
        @param w, k: parameters for the minimizer.
        @param occ_limit, dist_limit: validator parameters.
        @param kc: constructed KMerChain, just in case.
        '''
        for i in trange(2):
            pass  # trange compat test
        self.kc = kc
        self.w, self.k = w, k
        self.n = self.kc.data_len
        if dist_limit is None:
            dist_limit = self.w // 2  # NO rounding up; exactly w/2 = forbidden
        self.occ_limit, self.cur_dist_limit, self.com_dist_limit = occ_limit, dist_limit, dist_limit
        self.validator = validate_locs_gen(occ_limit, dist_limit)
        self.cur_kmers = set()
        self.all_added_kmers = set()
        self.prev_kmers = []
        self.round_id = 0
        self.covc = CoverageChecker(self.n, w)
        self._config_no = 0
        self.current_indexes = []

    def save(self, fn):
        '''
        Save current selection to a file.
        This saves the set of k-mers selected at each step, and basic parameters
        like w, k and n (for consistency check).
        Due to some technical limitations, the tags are not preserved for now.
        @param fn: name of file to store the data.
        '''
        assert self.round_id == len(self.prev_kmers)
        data = {"w": self.w, "k": self.k, "n": self.n,
                "occ_limit": self.occ_limit, "cur_dist_limit": self.cur_dist_limit,
                "com_dist_limit": self.com_dist_limit, "cur_kmers": self.cur_kmers,
                "prev_kmers": self.prev_kmers}
        with open(fn, "bw") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, fn, kc : KMerChain):
        '''
        Loads selection of k-mers from a file. Requires file name and the
        constructed KmerChain object. This restores the set of k-mers selected
        at each step, and the basic parameters of w and k.
        Due to some technical limitations, the tags are not preserved for now.
        @param fn: name of file to restore the data from.
        @param kc: the constructed KmerChain.
        @return: the restored AnchorStrategy object, which can be profiled
                and if desired, run more rounds of optimization on.
        '''
        with open(fn, "b") as f:
            data = pickle.load(f)
        assert data["n"] == kc.data_len
        ret = cls(kc, data["w"], data["k"])  # type: AnchorStrategy
        ret.change_limits(data["occ_limit"], data["cur_dist_limit"],
                          data["com_dist_limit"])
        ret.prev_kmers = data["prev_kmers"]
        ret.cur_kmers = data["cur_kmers"]
        ret.round_id = len(ret.prev_kmers)
        # Add all previous kmers as one big batch
        for ls in ret.prev_kmers:
            for x in ls:
                ret.all_added_kmers.add(x)
                for l in kc.iter_by_value(x):
                    ret.covc.add_loc(l)
        ret.covc.commit_all()
        # Add all current kmers
        for x in ret.cur_kmers:
            ret.all_added_kmers.add(x)
            for l in kc.iter_by_value(x):
                ret.covc.add_loc(l)
        return ret


    def get_occurence_by_loc(self, loc):
        '''
        Get a list of occurences of a specific k-mer and pass through validiation.
        @param loc: the location of the k-mer to be indexed.
        @return: None if it fails validation; otherwise sorted list of locations.
        '''
        return self.validator(self.kc.iter_by_idx(loc))

    def change_limits(self, new_occ_limit = None, new_cur_dist_limit = None,
                      new_com_dist_limit = None):
        '''
        Change the preset limits within.
        When dist_limit is below w/2, the anchor set density guarantee will no
        longer hold. This can be, however, useful in experiment.
        @param new_occ_limit, new_cur_dist_limit, new_com_dist_limit:
                The new set of limits. None means unchanged.
        '''
        logging.info("New Limits: Occurence {}{}, Distance CUR {}{}, COM {}{}".format(
            self.occ_limit,
            "" if new_occ_limit is None else "->{}".format(new_occ_limit),
            self.cur_dist_limit,
            "" if new_cur_dist_limit is None else "->{}".format(new_cur_dist_limit),
            self.com_dist_limit,
            "" if new_com_dist_limit is None else "->{}".format(new_com_dist_limit)))
        if new_occ_limit is not None:
            self.occ_limit = new_occ_limit
        if new_cur_dist_limit is not None:
            self.cur_dist_limit = new_cur_dist_limit
            assert self.cur_dist_limit >= 0
            assert self.cur_dist_limit < self.w
        if new_com_dist_limit is not None:
            self.com_dist_limit = new_com_dist_limit
            assert self.com_dist_limit >= 0
            assert self.com_dist_limit < self.w
        self.validator = validate_locs_gen(self.occ_limit, self.cur_dist_limit)
        self.covc._invalidate_caches()

    C_COVERED = 11
    C_DENIED_HARD = 12
    C_ACCEPTABLE = 13

    def verify_locs(self, locs, tags = True):
        '''
        Verify a set of locations. Re-implements CoverageChecker.verify to support
        better flexibility.
        @param locs: List of locations.
        @param tags: If tags are attached. (This is for legacy compatibility)
        @return: (status code, conflicts):
                C_COVERED: The location is fully covered by commited locs.
                C_DENIED_HARD: The location is in conflict with a commited location.
                                For both return codes conflicts = [].
                C_ACCEPTABLE: The location is not in conflict with a commited
                            location, but could be with a current location.
                            In this case the second return value contains
                            list of pairs of (loc, tag) with the conflicts.
        '''
        is_covered = True
        conflicts = []
        for l in locs:
            com_l, com_r = self.covc.check_commited(l)
            if com_r - com_l <= self.w:
                continue  # covered locations
            cur_l, cur_r = self.covc.check_current(l)
            is_covered = False
            if (com_r - l <= self.com_dist_limit) or (l - com_l <= self.com_dist_limit):
                return self.C_DENIED_HARD, []
            cdl = self.cur_dist_limit
            if (cur_r - l <= cdl) or (l - cur_l <= cdl):
                cur_conf = self.covc.get_covered_list_from_current(l, tags)
                for item in cur_conf:
                    y = item[0] if tags else item
                    if abs(l - y) <= cdl:
                        conflicts.append(item)
        if is_covered:
            return self.C_COVERED, []
        else:
            return self.C_ACCEPTABLE, conflicts

    def check_loc_cover(self, l):
        '''
        A simple helper function that checks if a single location is covered.
        Shorthand for self.verify_locs([l]) == self.C_COVERED.
        @param l: the location.
        @return: Boolean value indicating if this is covered.
        '''
        com_l, com_r = self.covc.check_commited(l)
        return com_r - com_l <= self.w

    def filter_locs(self, start, stop = None, gap = None):
        '''
        Given a list of locations, remove those that does not pass the validator.
        @param l: list of indexes.
        @return: filtered list.
        '''
        ret = []
        for idx in range(start, self.n if stop is None else stop,
                         self.w if gap is None else gap):
            if self.get_occurence_by_loc(idx) is not None:
                ret.append(idx)
        return ret

    def purge_selection(self, score_threshold = 0, singleton_only = True):
        '''
        Purge the set of current selected locations, to remove those whose score
        contribution is negative.
        @param score_threshold: The threshold of net score for removing an element.
                By setting this to be zero, singleton will be automatically removed.
        @param singleton_only: A special mode that only removes singletons.
                More specifically, those with net score of zero. This overrides
                score_threshold.
        @return: number of locations removed.
        '''
        assert score_threshold >= 0
        clist = list(self.cur_kmers)
        ret = 0
        for c in clist:
            # a wasteful step, but this is for internal consistency
            tloc = self.kc.single_value_to_loc(c)
            score, count = self.del_kmer_by_loc(tloc, count = True)
            if singleton_only:
                if score != 0:
                    self.add_kmer_by_loc(tloc)
                else:
                    ret += count
            else:
                if score < score_threshold:
                    self.add_kmer_by_loc(tloc)  # only add back if strictly worse
                else:
                    ret += count
        return ret

    def add_kmer_by_loc(self, loc, tag = None):
        '''
        Adds a k-mer to the current set of selection, indexed by its location.
        @param loc: the location of the k-mer to be added.
        @param tag: the tag provided for this group.
        @return: change of score from this operation. If the operation is not
                    successful, 0 is returned.
        '''
        ret = 0
        dat = self.kc[loc]
        if dat in self.all_added_kmers:
            return ret
        for l in self.kc.iter_by_idx(loc):
            ret += self.covc.add_loc(l, tag)
        self.all_added_kmers.add(dat)
        self.cur_kmers.add(dat)
        return ret

    def del_kmer_by_loc(self, loc, count = False):
        '''
        Removes a k-mers from the current set of selection, indexed by its location.
        @param loc: the location of the k-mer to be removed.
        @param count: whether a count should be returned alongside change in score.
        @return: change of score from this operation. If the operation is not
                    successful, 0 is returned instead.
        '''
        ret = 0
        dat = self.kc[loc]
        if count:
            c = 0
        if dat not in self.all_added_kmers:
            if count:
                return 0
            else:
                return ret, None
        for l in self.kc.iter_by_idx(loc):
            ret += self.covc.delete_loc(l)
            c += 1
        self.all_added_kmers.remove(dat)
        self.cur_kmers.remove(dat)
        if count:
            return ret, c
        else:
            return ret

    def discard_round(self):
        '''
        Discard all currennt k-mers. Does not increment round-id.
        @return: new round number (unchanged).
        '''
        for x in self.cur_kmers:
            self.all_added_kmers.remove(x)
        self.cur_kmers = set()
        self.covc.start_over()
        return self.round_id

    def commit_round(self):
        '''
        Commits all current k-mers and start a new round.
        @return: new round number, first round being 0.
        '''
        self.prev_kmers.append(self.cur_kmers)
        self.cur_kmers = set()
        self.covc.commit_all()
        self.round_id += 1
        return self.round_id

    def calc_total_df_saving(self, commited_only = False):
        '''
        Calculates total savings measured by density factor.
        Note: as the scores are already scaled by (w+1) we don't need to multiply
                it again.
        @param commited_only: if onnly commited elements are counted.
        @return: total savings measured by density factor.
        '''
        if commited_only:
            return self.covc.commited_score() / self.n
        else:
            return self.covc.total_score() / self.n

    def detailed_cov_stat(self, commited_only = False):
        '''
        Prints detailed coverage status for inspection.
        This is only intended to be used for diagonsis.
        @param commited_only: if only commited segments are counted.
        '''
        c_cov = self.covc.commited_cover
        c_ele = self.covc.commited_ele
        c_seg = self.covc.commited_segs
        if not commited_only:
            c_cov += self.covc.cur_cover
            c_ele += self.covc.cur_ele
            c_seg += self.covc.cur_segs
        if c_ele == 0:
            logging.info("No location covered currently")
        else:
            logging.info("Coverage = {} ({:.2f} %), segments = {} ({:.2f} ele/seg), elements = {} ({:.2f} cov/ele)".
                format(c_cov, c_cov / self.n * 100, c_seg, c_ele / c_seg, c_ele, c_cov / c_ele))

    def detailed_gap_stat(self, commited_only = False):
        '''
        Prints detailed Gap & Segment length distribution information.
        This is only for diagonsis.
        @param commited_only: if only commited segments are counted.
        '''
        d_gap = self.covc.gap_dist(commited_only)
        l_gap = list(d_gap.items())
        logging.info("Gaps: By frequency {}".format(sorted(l_gap, key=itemgetter(1), reverse=True)[:self.w]))
        logging.info("Gaps: By Length S {}".format(sorted(l_gap, key=itemgetter(0))[:self.w]))
        logging.info("Gaps: By Length L {}".format(sorted(l_gap, key=itemgetter(0), reverse=True)[:self.w]))
        l_seg = list(self.covc.segment_dist(commited_only).items())
        logging.info("Segments: By frequency {}".format(sorted(l_seg, key=itemgetter(1), reverse=True)[:self.w]))
        logging.info("Segments: By Length S {}".format(sorted(l_seg, key=itemgetter(0))[:self.w]))
        logging.info("Segments: By Length L {}".format(sorted(l_seg, key=itemgetter(0), reverse=True)[:self.w]))
        d_win = self.covc.window_hit_dist()
        logging.info("Window Coverage: {}".format(list(d_win.items())))
        d_win = self.covc.window_hit_dist(self.w + 1)
        logging.info("Context Coverage: {}".format(list(d_win.items())))

    def _kmer_prio_func(self, m):
        '''
        Helper function that returns the priority of a k-mer, according to
        current results. The higher the return value, the higher priority it is.
        More specifically, for previously commited k-mers the value is (N+1) -> 2,
        for current k-mers the value is 1, for everything else 0.
        @param m: the k-mer to be checked.
        @return: an integer indicating its priority.
        '''
        assert len(self.prev_kmers) == self.round_id
        for i in range(self.round_id):
            if m in self.prev_kmers[i]:
                return self.round_id + 1 - i
        if m in self.cur_kmers:
            return 1
        else:
            return 0

    def _naive_window_energy(self, mers):
        '''
        Helper function that calculates expected energy in current window.
        In current implementation, it treats everything in self.cur_kmers as
        priority 0, and everything else as priority 1.
        @param mers: list of k-mers in current window.
        @return: energy of current window.
        '''
        #  def priority_func(m): return (0 if m in self.cur_kmers else 1)
        priority_func = self._kmer_prio_func
        unique_last = True
        seen_mers = set()
        distinct_count = defaultdict(int)
        for idx, m in enumerate(mers):
            p = priority_func(m)
            if (idx != len(mers) - 1) and (m == mers[-1]):
                unique_last = False
            if m not in seen_mers:
                seen_mers.add(m)
                distinct_count[p] += 1
        p_key = min(distinct_count)
        p_count = distinct_count[p_key]
        divisor = 0
        if priority_func(mers[0]) == p_key:
            divisor += 1
        if (priority_func(mers[-1]) == p_key) and unique_last:
            divisor += 1
        return divisor / p_count

    def calc_current_energy(self):
        """
        Calculate total energy (that is, expected density for a rndom minimizer),
        for current configuration over current sequence.
        """
        MAX_PRIO = self.round_id + 2
        priority_func = self._kmer_prio_func
        w = self.w
        d = self.kc.data
        uniq_counts = [0] * MAX_PRIO
        high_prio = -1
        prio_vals = deque()
        for i in range(w):
            p = priority_func(d[i])
            prio_vals.append(p)
            if self.kc.pre[i] == self.kc.PH:
                if p > high_prio: high_prio = p
                uniq_counts[p] += 1
        assert self.kc.PH < 0  # required
        e = 0
        for i in trange(w, self.n):
            # old element
            j = i - w - 1  # element to remove
            if j >= 0:  # ignore special case of 1st window
                p = prio_vals.popleft()
                if (self.kc.nxt[j] == self.kc.PH) or (self.kc.nxt[j] >= i):
                    # >= i :: this is to handle the scenario where the kmer is
                    # unique and is added and deleted in the same loop.
                    # currently, the behavior is to remove then add the same
                    # k-mer.
                    uniq_counts[p] -= 1
                    if (p == high_prio) and (uniq_counts[p] == 0):  # recalculation
                        high_prio = max(i for i in range(MAX_PRIO) if uniq_counts[i] > 0)
            # new element. Do this later because we can calculate energy faster
            dv = 0  # the dividend in the formula
            p = priority_func(d[i])
            prio_vals.append(p)
            if self.kc.pre[i] < i - w:
                # PH < 0 so this also takes into account new k-mers
                # also < (i-w) not (i-w-1), to take care of the adding and
                # removing same k-mer situation (see previous comment).
                if p > high_prio: high_prio = p
                uniq_counts[p] += 1
                if p == high_prio: dv += 1  # energy for the end
            assert len(prio_vals) == (w + 1)
            # calculate energy for the start
            if (prio_vals[0] == high_prio):
                dv += 1
            assert uniq_counts[high_prio] > 0
            e += dv / uniq_counts[high_prio]
        return e / self.n

    def calc_random_energy(self, samples = 100000):
        '''
        Calculate total energy (that is, expected density for a random minimizer)
        for current configuration, over a random sequence.
        @param samples: number of distinct window samples used.
        @return: *average* energy of a window, that is expected density.
        '''
        e = 0
        for i in trange(samples):
            e += self._naive_window_energy(list(random_mer_generator(self.w + 1, self.k)))
        return e / samples

    def single_pass_random_order(self, offset = None, strat = 0):
        '''
        Implements the simpliest strategy of only adding locations modulo w = offset.
        However, the k-mers are sorted randomly. and if a k-mer is processed once,
        it will not be added again.
        This to some effect simulates the two-class priority ordering in Winnowmap.
        @param offset: where to start the chain.
        @param strat: strategy when conflict happens.
                strat = 0: remove all conflicting k-mers. (Deprecated)
                strat = 1: only do so when it results in improvement (Deprecated)
                strat = -1: do nothing (ablation study - but turns out to be extremely good?)
                strat = 2: relegate all such k-mers to level 2. (how about hard rejects?)
        @return: the values obtained from calc_current_energy and calc_random_energy.
        '''
        assert strat in {0, -1}
        if strat == 0:
            self.change_limits(new_cur_dist_limit = self.w - 1) # this emulates the old behavior?
        processed = set()
        n, w = self.n, self.w
        if offset is None:
            offset = random.randrange(w)
        indexes = list(range(offset, n, w))
        self.current_indexes = indexes
        random.shuffle(indexes)
        logging.info("[SP] Postprocessing finished")
        best_df = 0
        conf_tally = defaultdict(int)
        for i in trange(len(indexes)):
            idx = indexes[i]
            dat = self.kc[idx]
            if dat in processed:
                continue  # more frequent k-mers take lower priority
            locs = self.get_occurence_by_loc(idx)
            if locs is None:
                continue
            conflicts = []
            if strat != -1:
                status, conflicts = self.verify_locs(locs)
                assert status == self.C_ACCEPTABLE
                conf_tally[len(conflicts)] += 1
            for l, _ in conflicts:
                self.del_kmer_by_loc(l)
            self.add_kmer_by_loc(idx)
            best_df = max(best_df, self.calc_total_df_saving())
        logging.info("Tallying of Conflicts: {}".format(conf_tally))
        logging.info("Best CDF: {}".format(best_df))
        logging.info("Final CDF: {}".format(self.calc_total_df_saving()))
        #  self.detailed_cov_stat()
        #  self.detailed_gap_stat()
        # These are actual energy values, not scaled by (w+1)
        c0 = self.calc_current_energy()
        c1 = self.calc_random_energy()
        logging.info("Energy calculation: Current {:.4f} (DF {:.3f}), Random {:.4f} (DF {:.3f})".
              format(c0, c0 * (w + 1), c1, c1 * (w + 1)))
        return c0, c1

    def _single_random_pass(self, gap, rep, occ_tol, dist_tol, term_diff = 0.0001):
        '''
        simple wrapper for running the random selection algorithm.
        '''
        assert gap > dist_tol
        self._config_no += 1
        logging.info("Next configuration: Gap = {}, Limits = {}/{}, {} attempts".format(
            gap, occ_tol, dist_tol, rep))
        self.change_limits(occ_tol, dist_tol, dist_tol)
        for r in range(rep):
            offset = random.randrange(gap)
            indexes = self.filter_locs(offset, self.n, gap)
            logging.info("Attempt #{}: offset = {}, Locations {} -> {}".format(
                r, offset, len(_indexes), len(indexes)))
            _sub_index = 0
            self.detailed_cov_stat()
            self.current_indexes = indexes
            while True:
                logging.info("Round {} started. This is Config #{}, Offset #{}, Iteration #{}".format(
                    self.round_id, self._config_no, r, _sub_index))
                _sub_index += 1
                random.shuffle(indexes)
                visited_kmers = set()  # set of currently visited kmers
                cover_locs = set()
                b00, b0, b1, b2, b3 = 0, 0, 0, 0, 0
                for i in trange(len(indexes)):
                    idx = indexes[i]
                    dat = self.kc[idx]
                    if dat in self.all_added_kmers:
                        b00 += 1
                        continue
                    if (dat in visited_kmers) or (dat in self.all_added_kmers):
                        b0 += 1
                        continue
                    if self.check_loc_cover(idx):
                        cover_locs.add(idx)
                        b1 += 1
                        continue
                    visited_kmers.add(dat)
                    locs = self.get_occurence_by_loc(idx)
                    status, conflicts = self.verify_locs(locs)
                    if status == self.C_DENIED_HARD:
                        b2 += 1
                        continue
                    assert status == self.C_ACCEPTABLE
                    for l, _ in conflicts:
                        self.del_kmer_by_loc(l)
                    b3 += 1
                    self.add_kmer_by_loc(idx)
                logging.info("(Debug) Exits: {}, {}, {}, {}, {}".format(b00, b0, b1, b2, b3))
                if self.calc_total_df_saving(False) - self.calc_total_df_saving(True) < term_diff:
                    logging.info("Improvement below threshold ({} -> {})- early skipping".format(
                        self.calc_total_df_saving(True), self.calc_total_df_saving(False)))
                    logging.info("(Debug) total locs {}, visited {}, covered {}".format(
                        len(indexes), len(visited_kmers), len(cover_locs)))
                    #  self.commit_round()
                    self.discard_round()
                    break
                else:
                    logging.info("(Debug) DF changes: {} -> {}".format(
                        self.calc_total_df_saving(True), self.calc_total_df_saving(False)))
                new_indexes = []
                purged_locs = self.purge_selection()
                logging.info("Purged {} selected locations as singletons.".format(
                    purged_locs))
                for idx in indexes:
                    dat = self.kc[idx]
                    if dat not in self.cur_kmers:
                        if idx not in cover_locs:
                            new_indexes.append(idx)
                self.commit_round()
                self.detailed_cov_stat()
                logging.info("Current CDF: {:.5f}, Remaining Locations {}".format(
                    self.calc_total_df_saving(), len(new_indexes)))
                if len(new_indexes) >= len(indexes) - 100:
                    break
                indexes = new_indexes
                self.current_indexes = indexes

    def multi_random_pass(self, offset = None, strat = None):
        '''
        Some more attempts at the multiple linear pass method.
        @param offset: Ignored.
        @param strat: currently, a single value for dist tolerance, by *percentage* of w.
        @return: the values obtained from calc_current_energy and calc_random_energy.
        '''
        total_rounds = 3
        gap_values = [self.w] * 3
        reps = [5] * 3
        occ_tols = [20, 40, 60]
        if strat is None:
            dist_tols = [int(self.w * 0.6)] * 3
        else:
            dist_tols = [int(self.w * strat)] * 3
        logging.info("[MP] Postprocessing Finished.")

        for i in range(total_rounds):
            self._single_random_pass(gap_values[i], reps[i], occ_tols[i],
                                     dist_tols[i])
        c0 = self.calc_current_energy()
        c1 = self.calc_random_energy()
        logging.info("Energy Information: Current DF {:.4f}, Random DF {:.4f}".format(
            c0 * (self.w + 1), c1 * (self.w+1)))
        return c0, c1

    def single_pass_jiggling(self, offset = None, try_width = 5):
        '''
        Implements a slightly stronger strategy: move back a little bit when
        conflict happens... (Delayed)
        '''
        pass


if __name__ == "__main__":
    from tqdm import trange
    logging.basicConfig(format="%(asctime)s %(message)s",datefmt="%I:%M:%S",
                    level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument("funcname")
    parser.add_argument("-o", "--offset", type=int)
    parser.add_argument("-s", "--strat", type=int)
    parser.add_argument("-w", "--wval", type=int)
    parser.add_argument("-k", "--kval", type=int)
    args = parser.parse_args()
    logging.info("Arguments:{}".format(args))
    logging.info("[START]")
    seq_file = "chr1.seq"
    with open(seq_file) as f:
        seq = f.readline().strip()
    logging.info("Sequence loading finished")
    kmers = []
    #  w = 100 if args.wval is None else args.wval
    #  k = 16 if args.kval is None else args.kval
    w, k = args.wval, args.kval
    for x in sequence_mer_iterator(k, seq):
        kmers.append(x)
    logging.info("Sequence parsing finished")
    kc = KMerChain(kmers)
    logging.info("KMerChain constructed")
    #  validiator_test(kmers)
    s = AnchorStrategy(kc, w, k, occ_limit = 15)
    #  s.single_pass_strat(args.offset, args.strat)
    #  s.single_pass_random_order(args.offset, args.strat)
    #  s.multi_pass_random_order(args.offset, args.strat)
    getattr(s, args.funcname)(args.offset, args.strat)

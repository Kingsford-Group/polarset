'''
The new AnchorStrat class that are based on more efficient representations.
'''
from anchor_sets import CoverageChecker  # for type checks
from anchor_sets_bit import BitCoverageChecker, CompactKMerChain
import logging
import random
import argparse
import pickle
from anchor_strat import validate_locs_gen, AnchorStrategy
import array
from collections import deque
trange = range

def zero_iterator(n):
    '''
    A helper function that yields N zeroes.
    @param n: number of zeroes.
    '''
    for i in range(n): yield 0

class CompactAnchorStrat(AnchorStrategy):
    '''
    Maybe TBD later: replace self.cur_kmers, self.prev_kmers, and self.all_added_kmers
                    as one array.
    This *should* speed up things I think? Maybe?
    '''
    def __init__(self, kc : CompactKMerChain, w, k, occ_limit = 25, dist_limit = None,
                 cc_cls = BitCoverageChecker):
        '''
        Initializes a AnchorStrategy class.
        The actual strategy is not passed as parameters at this time, but
        rather as function calls.
        @param kc: constructed KMerChain, just in case.
        @param w, k: parameters for the minimizer.
        @param occ_limit, dist_limit: validator parameters.
        @param cc_cls: CoverageChecker class.
        '''
        self.kc = kc
        self.w, self.k = w, k
        self.n = self.kc.data_len
        if dist_limit is None:
            dist_limit = self.w // 2  # NO rounding up; exactly w/2 = forbidden
        self.occ_limit, self.cur_dist_limit, self.com_dist_limit = occ_limit, dist_limit, dist_limit
        self.validator = validate_locs_gen(occ_limit, dist_limit)
        self.kmer_level = array.array('b', zero_iterator(self.n))
        #  self.cur_kmers = set()
        #  self.all_added_kmers = set()
        #  self.prev_kmers = []
        self.round_id = 1  # for now we also use this as kmer_level_id
        self.covc = cc_cls(self.n, w)  # type: CoverageChecker
        self._config_no = 0

    def save(self, fn):
        '''
        Saves current selection to a file.
        This saves the parameters (w, k, tolerance values), and kmer_level.
        @param fn: name of file to store the data.
        '''
        data = {"params": [self.w, self.k, self.n, self.occ_limit, self.cur_dist_limit, self.com_dist_limit],
                "rounds": self.round_id, "kmer_level": self.kmer_level}
        with open(fn, "bw") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, fn, kc : CompactKMerChain):
        '''
        Loads a leveled anchor set from disk.
        The CoverageChecker will be restored to its state.
        '''
        with open(fn, 'br') as f:
            data = pickle.load(f)
        w, k, n, occ_limit, cur_dist_limit, com_dist_limit = data["params"]
        ret = cls(kc, w, k, occ_limit, cur_dist_limit)  # type: CompactAnchorStrat
        assert n == ret.n
        if com_dist_limit != cur_dist_limit:
            ret.change_limits(new_com_dist_limit=com_dist_limit)
        ret.round_id = data["rounds"]
        ret.kmer_level = data["kmer_level"]
        km_list = []
        for i in range(ret.round_id):
            km_list.append([])
        for i in range(n):
            if ret.kmer_level[i] > 0:
                km_list[ret.kmer_level[i] - 1].append(i)
        for kl in range(ret.round_id):
            if len(km_list[kl]) > 0:
                for i in km_list[kl]:
                    for l in kc.iter_by_idx(i):
                        ret.covc.add_loc(l)
            if kl != ret.round_id - 1:
                ret.covc.commit_all()
        return ret

    def calc_current_energy(self):
        """
        Calculate total energy (that is, expected density for a random minimizer),
        for current configuration over current sequence.
        """
        w, n = self.w, self.n
        _max_round = self.round_id + 2
        prio_converter = lambda x: 0 if x <= 0 else (_max_round - x)
        uniq_counts = [0] * _max_round
        high_prio = -1
        cur_kmers = dict()
        prio_vals = deque()
        for i in range(w):
            dat = self.kc[i]
            p = prio_converter(self.kmer_level[dat])
            prio_vals.append(p)
            if dat not in cur_kmers:
                high_prio = max(high_prio, p)
                uniq_counts[p] += 1
                cur_kmers[dat] = 0
            cur_kmers[dat] += 1
        e = 0
        for i in range(w, self.n):
            # remove outgoing k-mer; recall we operate over contexts
            j = i - w - 1
            if j >= 0:
                dat = self.kc[j]
                p = prio_vals.popleft()
                cc = cur_kmers.pop(dat)  # implicitly deletes the key
                if cc > 1:
                    cur_kmers[dat] = cc - 1  # adds the key back
                else:
                    assert cc == 1
                    # losing out a unique kmer
                    uniq_counts[p] -= 1
                    if (uniq_counts[p] == 0) and (p == high_prio):
                        for j in range(p-1, -1, -1):
                            if uniq_counts[j] > 0:
                                high_prio = j
                                break
            dv = 0  # the dividend
            dat = self.kc[i]
            p = prio_converter(self.kmer_level[dat])
            prio_vals.append(p)
            if dat not in cur_kmers:
                cur_kmers[dat] = 0
                high_prio = max(high_prio, p)
                uniq_counts[p] += 1
                if p == high_prio:  # counting the end element; needs to be unique
                    dv += 1
            cur_kmers[dat] += 1
            if prio_vals[0] == high_prio:
                dv += 1
            e += dv / uniq_counts[high_prio]
            assert len(prio_vals) == (w+1)
        return e / self.n

    def calc_random_energy(self, samples = 100000):
        logging.warning("RandomEnergy Not Implemented Yet - using estimates")
        return 2 / (self.w + 1)

    def get_occurence_by_loc(self, loc):
        '''
        Get a list of occurences of a specific k-mer and pass through validation.
        @param loc: the location of the k-mer to be indexed.
        @return: None if it fails validation, otherwise sorted list of locations.
        WARNING: In AnchorStratV3 no validation is done.
        '''
        jump_val = self.kc.jump_table[self.kc.lookup_table[loc]]
        if jump_val >= self.occ_limit:
            return None
        else:
            c = self.kc.iter_by_idx(loc)
            return self.validator(c)

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
        for l in self.kc.iter_by_idx(dat, is_repr = True):
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
        for l in self.kc.iter_by_idx(dat, is_repr = True):
            ret += self.covc.delete_loc(l)
            c += 1
        self.kmer_level[dat] = 0 - self.round_id
        if count:
            return ret, c
        else:
            return ret

    def filter_locs(self, start, stop = None, gap = None):
        '''
        Remove list of locations that are covered by commited locations, and
        list of locations that didn't pass the validator...
        @param start, gap: parameters of the iterator.
        @return: filtered list.
        '''
        #  raise Exception("Using outdated functions")
        ret = []
        for idx in range(start, self.n if stop is None else stop,
                         self.w if gap is None else gap):
            if not self.check_loc_cover(idx):
                if self.get_occurence_by_loc(idx) is not None:
                    ret.append(idx)
        return ret

    def purge_selection(self, score_threshold = 0, singleton_only = True):
        '''
        WARNING: This also advances round ID by one, essentially leaving an
                    "empty round".
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
        assert singleton_only
        self.round_id += 1
        ret = 0
        for x in self.current_indexes:
            dat = self.kc[x]
            kl = self.kmer_level[dat]
            if kl == self.round_id:
                continue  # kmer survived
            elif kl == self.round_id - 1:  # kmer to be verified here
                score, count = self.del_kmer_by_loc(dat, count = True, is_repr_and_valid=True)
                if score != 0:
                    self.add_kmer_by_loc(dat, is_repr_and_valid=True)
                else:
                    ret += count  # deleted for good
        return ret

    def discard_round(self, remove_discards = True):
        '''
        Discard all current k-mers. Does not increment round-id.
        This is discouraged for now.
        @return: new round number (unchanged).
        '''
        for x in self.current_indexes:
            dat = self.kc[x]
            kl = self.kmer_level[dat]
            if kl == self.round_id:
                self.kmer_level[dat] = 0
            elif remove_discards and (kl == 0 - self.round_id):
                self.kmer_level[dat] = 0
        self.covc.start_over()
        return self.round_id

    def commit_round(self):
        '''
        Commits all current k-mers and start a new round.
        @return: new round number, first round being 0.
        '''
        self.covc.commit_all()
        self.round_id += 1
        assert self.round_id <= 120  # 60 "real" rounds should be more than enough, w/o repeats
        return self.round_id

    def single_pass_random_order(self, strat = 0):
        '''
        Implements the simpliest strategy of only adding locations modulo w = offset.
        However, the k-mers are sorted randomly. and if a k-mer is processed once,
        it will not be added again.
        This to some effect simulates the two-class priority ordering in Winnowmap.
        @param strat: strategy when conflict happens.
                strat = 0: remove all conflicting k-mers. (Deprecated)
                strat = 1: only do so when it results in improvement (Deprecated)
                strat = -1: do nothing (ablation study - but turns out to be extremely good?)
                strat = -2: do nothing with no filtering
                strat = 2: relegate all such k-mers to level 2. (how about hard rejects?)
        @return: the values obtained from calc_current_energy and calc_random_energy.
        '''
        assert strat in {0, -1, -2}
        if strat == 0:
            self.change_limits(new_cur_dist_limit = self.w - 1) # this emulates the old behavior?
        n, w = self.n, self.w
        offset = random.randrange(w)
        # not using filter_locs as a location may be filtered many times.
        # this is fixed in AnchorStrategyV3
        # indexes = list(range(offset, n, w))
        if strat != -2:
            indexes = self.filter_locs(offset)
        else:
            indexes = list(range(offset, n, w))
        random.shuffle(indexes)
        self.current_indexes = indexes
        logging.info("[SP] Postprocessing finished: {} -> {} locations".format(
            n // w, len(indexes)))
        best_df = 0
        #  conf_tally = defaultdict(int)
        for i in trange(len(indexes)):
            idx = indexes[i]
            dat = self.kc[idx]
            if self.kmer_level[dat] != 0:  # if it's deleted or added, ignore
                continue
            locs = self.get_occurence_by_loc(idx)
            if locs is None:
                continue
            conflicts = []
            if strat == 0:
                status, conflicts = self.verify_locs(locs, tags=False)
                assert status == self.C_ACCEPTABLE
                #  conf_tally[len(conflicts)] += 1
            for l in conflicts:
                self.del_kmer_by_loc(l)
            self.add_kmer_by_loc(dat, is_repr_and_valid=True)
            best_df = max(best_df, self.calc_total_df_saving())
        #  logging.info("Tallying of Conflicts: {}".format(conf_tally))
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

    def _single_random_pass(self, gap, rep, occ_tol, dist_tol, term_diff = None, monotone_mode = False,
                            keep_singleton = False):
        '''
        simple wrapper for running the random selection algorithm.
        @param term_diff: The difference between consequent rounds of same offset to stop.
                            Set this to None to disable retrying within the same offset.
        @param keep_singleton: Whether to keep singleton k-mers. defaults to false.
        @param monotone_mode: Whether to enable monotone mode. Under monotone mode a kmer
                            is added to the anchor set only if after removal of conflicting
                            k-mers, the energy increases.
        '''
        #  assert gap > dist_tol
        self._config_no += 1
        logging.info("Next configuration: Gap = {}, Limits = {}/{}, {} attempts".format(
            gap, occ_tol, dist_tol, rep))
        self.change_limits(occ_tol, dist_tol, dist_tol)
        for r in range(rep):
            offset = random.randrange(gap)
            #  indexes = self.filter_invalidated_locs(range(offset, self.n, gap))
            indexes = self.filter_locs(offset, gap=gap)
            logging.info("Attempt #{}: offset = {}, Locations {} -> {}".format(
                r, offset, self.n // gap, len(indexes)))
            _sub_index = 0
            self.detailed_cov_stat()
            self.current_indexes = indexes
            while True:
                logging.info("Round {} started. This is Config #{}, Offset #{}, Iteration #{}".format(
                    self.round_id, self._config_no, r, _sub_index))
                _sub_index += 1
                random.shuffle(indexes)
                #  visited_kmers = set()  # set of currently visited kmers
                #  cover_locs = set()
                new_indexes = []
                b00, b0, b1, b2, b3 = 0, 0, 0, 0, 0
                for i in trange(len(indexes)):
                    idx = indexes[i]
                    dat = self.kc[idx]
                    kl = self.kmer_level[dat]
                    if kl > 0:  # already in
                        b00 += 1
                        if term_diff is not None:
                            new_indexes.append(idx)
                        continue
                    if kl == 0 - self.round_id:  # already out this round
                        b0 += 1
                        if term_diff is not None:
                            new_indexes.append(idx)
                        continue
                    if self.check_loc_cover(idx):  # only when loc is covered new_index lacks that element.
                        b1 += 1
                        continue
                    if term_diff is not None:
                        new_indexes.append(idx)
                    locs = self.get_occurence_by_loc(idx)
                    status, conflicts = self.verify_locs(locs, tags = False)
                    if status == self.C_DENIED_HARD:
                        b2 += 1
                        continue
                    assert status == self.C_ACCEPTABLE
                    delta = 0
                    for l in conflicts:
                        delta += self.del_kmer_by_loc(l)
                    b3 += 1
                    delta += self.add_kmer_by_loc(idx)
                    if (delta < 0) and (monotone_mode):
                        # revert the changes
                        self.del_kmer_by_loc(idx)
                        for l in conflicts:
                            self.add_kmer_by_loc(idx)
                logging.info("(Debug) Exits: {}, {}, {}, {}, {}".format(b00, b0, b1, b2, b3))
                if not keep_singleton:
                    purged_locs = self.purge_selection()
                    logging.info("Purged {} selected locations as singletons.".format(
                        purged_locs))
                stop_flag = (term_diff is None)
                if term_diff is not None:
                    if self.calc_total_df_saving(False) - self.calc_total_df_saving(True) < term_diff:
                        logging.info("Improvement below threshold ({} -> {})- early skipping".format(
                            self.calc_total_df_saving(True), self.calc_total_df_saving(False)))
                        stop_flag = True
                self.commit_round()
                self.detailed_cov_stat()
                logging.info("Current DF saving: {:.5f}".format(self.calc_total_df_saving()))
                #  print("[DEBUG] Coverage verification")
                #  self.covc.verify_stats()
                if stop_flag:
                    break
                indexes = []
                for idx in new_indexes:
                    dat = self.kc[idx]
                    kl = self.kmer_level[dat]
                    if kl <= 0:
                        indexes.append(idx)
                logging.info("Current CDF: {:.5f}, Remaining Locations {}".format(
                    self.calc_total_df_saving(), len(indexes)))
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
        if self.kc.name == "hg38_all":
            occ_tols = [50, 100, 150]
        else:
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


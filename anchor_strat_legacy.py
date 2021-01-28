'''
    This file contains some deprecated methods for AnchorStrat.
    TODO: Later reorganize strategies from infrastructures.
'''
from anchor_strat import AnchorStrategy
from collections import deque, defaultdict
import logging
from tqdm import trange
import random

class AnchorStratLegacy(AnchorStrategy):
    def calc_current_energy_legacy(self):
        '''
        Calculate total energy (that is, expected density for a random minimizer)
        for current configuration, over current sequence.
        @return: *average* energy of a window, that is expected density.
        '''
        assert False  # deprecated for now
        w = self.w
        d = self.kc.data
        cur_window = deque(d[:w+1])
        e = self._naive_window_energy(cur_window)
        for m in range(d[w+1:]):
            cur_window.append(m)
            cur_window.popleft()
            e += self._naive_window_energy(cur_window)
        return e / self.n

    def single_pass_strat(self, offset = None, strat = 0):
        '''
        (re-)implements the simpliest strategy: find a long non-flexible chain.
        @param offset: where to start the chain. If set to None, will be a
                        random number between 0 and w-1.
        @param strat: strategy when conflict occurs.
                strat = 0: remove all conflicting k-mers.
                strat = 1: remove all conflicting k-mers, and everything that is added before the last k-mer got removed (deprecated)
                strat = -1: do nothing (ablation study).
        '''
        # THIS IS DEPRECATED
        assert False
        best_d = 0
        n, w = self.n, self.w
        start_idx = dict()
        current_start = offset
        for idx in trange(offset, n, w):
            dat = self.kc[idx]
            if dat not in start_idx:
                start_idx[dat] = idx
            locs = self.get_occurence_by_loc(idx)
            if locs is None:
                continue
            conflicts = []
            if strat != -1:
                for l in locs:
                    status = self.covc.verify(l)
                    assert status != self.covc.C_DENIED_HARD
                    if status == self.covc.C_DENIED_SOFT:
                        conflicts.extend(self.covc.get_covered_list_from_current(l))
            #  if len(conflicts) > 0:
                #  logging.debug("{}: {}({}) locs = {}, conflicts = {}".format(
                    #  offset, idx, dat, locs, conflicts))
            if strat == 0:
                for l, tag in conflicts:
                    self.del_kmer_by_loc(l)
            elif strat == 1:
                for l, tag in conflicts:
                    self.del_kmer_by_loc(l)
                if len(conflicts) > 0:
                    rem_target = max(c[1] for c in conflicts)
                    while current_start <= rem_target:
                        self.del_kmer_by_loc(current_start)
                        current_start += w
            else:
                assert strat == -1
            self.add_kmer_by_loc(idx, start_idx[dat])
            #  logging.debug("{}:{}({}) adding locations: {}".format(
                #  offset, idx, dat, locs))
            cdf = self.calc_total_df_saving()
            if cdf > best_d:
                if (strat == 0) or (strat == -1):
                    logging.info("{}/{}: CDF = {:.5f} ending at {}".format(
                        offset, strat, cdf, idx))
                elif strat == 1:
                    if (int(cdf * 100) > int(best_d * 100)):
                        logging.info("{}/1: CDF = {:.5f} interval={}-{}".format(offset,
                        cdf, current_start, idx))
                best_d = cdf

    def multi_pass_random_order(self, offset = None, strat = 0):
        '''
        Implements a slightly changed version of the simple strategy: select
        a particular modulus, then iteratively select k-mers by rounds as long
        as these k-mers are allowed to be in.
        @param offset: where to start.
        @param strat: strategy to use...
                strat = 0: default strategy: stick with one set of k-mers, stick
                            to definitions of anchor sets
                strat = 1: based on strat = 0, switch around different offsets
                            after exhausting options for one set
                strat = 2: also try with decreasing tolerance values (more tolerant O/T)
                            in this setting, offset is invalidated.
                strat = 3: similar to strat = 2, but also breaks anchor set conditions.
        '''
        assert strat in {0, 1, 2, 3}
        n, w = self.n, self.w
        if offset is None:
            offset = random.randrange(w)
        _s1_tries = _s1_initial_tries = 10
        if strat in {2, 3}:
            tol_values = list(range(w - 1, w // 2 + 1, -5))
            if strat == 3:
                tol_values.extend(range(w // 2, w // 4, -5))
        else:
            tol_values = [None]
        for tol in tol_values:
            self.change_limits(new_occ_limit = self.occ_limit + 5,
                new_cur_dist_limit = tol, new_com_dist_limit = tol)
            old_indexes = list(range(offset, n, w))
            indexes = self.filter_invalidated_locs(old_indexes)
            logging.info("Filtering k-mer locations: {} locs -> {} locs".format(
                len(old_indexes), len(indexes)))
            while True:
                logging.info("Round {} started".format(self.round_id))
                random.shuffle(indexes)
                visited_kmers = set()
                covered_kmers = set()
                conf_len_dict = defaultdict(int)
                for i in trange(len(indexes)):
                    idx = indexes[i]
                    dat = self.kc[idx]
                    if dat in visited_kmers:
                        continue
                    if dat in self.all_added_kmers:
                        continue
                    locs = self.get_occurence_by_loc(idx)
                    visited_kmers.add(dat)
                    status, conflicts = self.verify_locs(locs)
                    if status == self.C_DENIED_HARD:
                        conf_len_dict[-2] += 1
                        continue
                    elif status == self.C_COVERED:
                        conf_len_dict[-1] += 1
                        covered_kmers.add(dat)
                    else:
                        conf_len_dict[len(conflicts)] += 1
                        for l, _ in conflicts:
                            self.del_kmer_by_loc(l)
                        self.add_kmer_by_loc(idx)
                logging.info("Statuses (-2 = denied, -1 = covered, others = # of conflicts: {}".format(
                    conf_len_dict))
                pret = self.purge_selection()
                logging.info("Purged {} selected locations as singletons.".format(pret))
                new_indexes = []
                for idx in indexes:
                    dat = self.kc[idx]
                    if (dat not in self.cur_kmers) and (dat not in covered_kmers):
                        new_indexes.append(idx)
                self.commit_round()
                self.detailed_cov_stat()
                logging.info("Current CDF: {} (from {}), Remaining Loc {}".format(
                    self.calc_total_df_saving(True), self.calc_total_df_saving(False), len(new_indexes)))
                if len(new_indexes) >= len(indexes) - 10:  # some hacky stuff here
                    if strat == 0:
                        break
                    elif strat in {1, 2, 3}:
                        _s1_tries -= 1
                        if _s1_tries == 0:
                            _s1_tries = _s1_initial_tries
                            break
                        logging.info("Switching around to another offset now")
                        offset = random.randrange(w)
                        old_indexes = list(range(offset, n, w))
                        indexes = self.filter_invalidated_locs(old_indexes)
                        logging.info("Filtering k-mer locations: {} locs -> {} locs".format(
                            len(old_indexes), len(indexes)))
                else:
                    indexes = new_indexes
            self.detailed_cov_stat()
        c0 = self.calc_current_energy()
        c1 = self.calc_random_energy()
        logging.info("Energy calculation: Current {:.4f} (DF {:.3f}), Random {:.4f} (DF {:.3f})".
              format(c0, c0 * (w + 1), c1, c1 * (w + 1)))

if __name__ == "__main__":
    print("This is AnchorStrategy Legacy Code Deposit.")

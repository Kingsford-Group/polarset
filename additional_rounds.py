'''
Quick patchwork for running more rounds.
'''

import logging
from anchor_ds_v3 import AnchorStrategyV3
from anchor_sets_bit import load_shared_partial_SA, CompactKMerChain
from anchor_distributed import working_dir
import argparse
incl_log = working_dir + "incremental.dat"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seqfile")
    parser.add_argument("-k", "--kval", type=int)
    parser.add_argument("-w", "--wval", type=int, required=True)
    args = parser.parse_args()
    seq = args.seqfile
    w = args.wval
    k = args.kval
    policy = "multi_lax"
    tag = "{}_{}_{}_{}".format(policy, seq, w, k)
    logging.basicConfig(format="%(asctime)s [{}]  %(message)s".format(tag),
                        datefmt="%d %H:%M:%S",level=logging.DEBUG)
    logging.info("started loading:".format(seq))
    seq_n = load_shared_partial_SA(working_dir + "preload/{}_30.dump".format(seq), seq)
    logging.info("SA loaded")
    kc = CompactKMerChain(seq, seq_n, k)
    logging.info("KC loaded")
    s = AnchorStrategyV3.load(working_dir + tag + ".dump", kc)
    logging.info("AnchorStrat loaded")
    current_es = s.calc_current_energy()
    with open(incl_log, 'a') as f:
        print(tag + "+0", seq, w, k, current_es, 0, sep=',', file=f)
    logging.info("Initial DF Saving: {:.5f}, Calculated DF: {:.5f}".format(
        s.calc_total_df_saving(), current_es * (w+1)))
    s.commit_round()
    thres = kc.calc_freq_cutoff(0.99)
    _round_id = 0
    while True:
        _round_id += 1
        old_df = s.calc_total_df_saving()
        s._single_random_pass(w // 5, 3, thres, None, monotone_mode=True)
        logging.info("Calculating Actual Energy")
        new_df = s.calc_total_df_saving()
        new_es = s.calc_current_energy()
        logging.info("Incremental Round {} - DF Saving: {:.5f} (+{:.5f}), Calculated DF: {:.5f}".format(
            _round_id, new_df, new_df - old_df, new_es * (w+1)))
        with open(incl_log, 'a') as f:
            print(tag + "+" + str(_round_id), seq, w, k, new_es, 0, sep=',', file=f)
        if (new_df - old_df) < 0.01:
            break
    logging.info("finished - triggered finish condition")
    s.save(working_dir + "{}_incl.dump".format(tag))
    logging.info("finished - triggered finish condition")

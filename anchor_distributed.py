'''
    Boilerplate codes to run several experiments in parallel.
'''
from anchor_sets import KMerChain
from anchor_strat import AnchorStrategy, sequence_mer_iterator
from anchor_sets_bit import CompactKMerChain, load_shared_partial_SA, BitCoverageChecker
from anchor_strat_bit import CompactAnchorStrat
from anchor_ds_v3 import CoverageCheckerV3, AnchorStrategyV3
import random
import argparse
from multiprocessing import Pool, Manager
import os.path
import datetime
import logging
import time
import gc
working_dir = None
assert working_dir is not None
master_log = working_dir + "master_log"
master_data = working_dir + "master_dat"
mgr = Manager()
mlock = mgr.Lock()
task_list = []

_v3_testing = True

# simple timer
def ctime():
    now = datetime.datetime.now()
    return now.strftime("%H:%M:%S")

trange = range

def worker_func(item):
    '''
    This is the main function that workers are supposed to call.
    The input to this function is a 6-element tuple:
    @param idx: the identifier of this worker. used in storing log files and
                dump files.
    @param seq_file: the name of the sequence file.
    @param w, k: the parameters for w and k.
    @param funcname: the name of the function to call within AnchorStrat class.
    @param strat_val: the value to be passed to the function call.
    @return: nothing.
    '''
    for i in trange(2):
        pass  # trange compat test
    idx, seq_file, w, k, funcname, strat_val = item
    print(ctime(), "Work started:", idx)
    logfile = working_dir + idx + ".log"
    logging.basicConfig(format="%(asctime)s %(message)s",datefmt="%I:%M:%S",
                    level=logging.DEBUG, filename=logfile, filemode='w', force=True)
    with open(seq_file + ".seq") as f:
        seq = f.readline().strip()
    logging.info("Sequence loading finished")
    kmers = []
    for x in sequence_mer_iterator(k, seq):
        kmers.append(x)
    logging.info("Sequence parsing finished")
    kc = KMerChain(kmers)
    logging.info("KMerChain constructed")
    s = AnchorStrategy(kc, w, k, occ_limit = 15)
    c0, c1 = getattr(s, funcname)(strat = strat_val)
    s.save(working_dir + idx + ".dump")
    print(ctime(), "Work finished:", idx)
    with open(master_log, "a") as f:
        print("{} {} reported densities: {:.6f} Current {:.6f} Random".format(
            ctime(), idx, c0, c1), file=f)

def new_worker_func(item):
    '''
    The same function as above, but using the compact version of these data
    structures.
    '''
    idx, seq_file, n, w, k, funcname, strat_val = item
    print(ctime(), "Work started:", idx)
    stime = time.perf_counter()
    logfile = working_dir + idx + ".log"
    logging.basicConfig(format="%(asctime)s [{}] %(message)s".format(idx),
                        datefmt="%d %H:%M:%S",level=logging.DEBUG, filename=logfile, filemode='w')
                        #  ,force=True)  # no longer required as each process
                        #  works on 1 task only now
    logging.info("Sequence loading finished")
    # don't load the sequence; do the preloading instead
    try:
        kc = CompactKMerChain(seq_file, n, k)
        logging.info("CompactKMerChain Loaded (Preprocessed Suffix Array)")
        if _v3_testing and (not idx.startswith("baseline")):
            s = AnchorStrategyV3(kc, w, k)
            #  s = CompactAnchorStrat(kc, w, k, occ_limit = 15, cc_cls=CoverageCheckerV3)
        else:
            occ_limit = kc.calc_freq_cutoff(0.98) + 1
            s = CompactAnchorStrat(kc, w, k, occ_limit)
        logging.info("CompactAnchorStrat Constructed")
        c0, c1 = getattr(s, funcname)(strat = strat_val)
        s.save(working_dir + idx + ".dump")
    except Exception as e:
        import sys
        import traceback as tb
        ce, ct, ctb = sys.exc_info()
        logging.error("Exception Thrown: {}".format(repr(e)))
        t = time.perf_counter() - stime
        print(ctime(), "Exception thrown from", idx, "time spent: {:.2f} sec".format(t))
        tb.print_exception(ce, ct, ctb)
        with open(master_log, "a") as f:
            print("{} {} exits with exception: {}".format(ctime(), idx, repr(e)))
            tb.print_exception(ce, ct, ctb, file=f)
    else:
        t = time.perf_counter() - stime
        print(ctime(), "Work finished:", idx, "time spent: {:.2f} sec".format(t))
        with open(master_log, "a") as f:
            print("{} {} reported densities: {:.6f} (DF: {:.4f}) in {:.2f} sec".format(
                ctime(), idx, c0, c0 * (w + 1), t), file=f)
        with open(master_data, "a") as f:
            print(','.join(list(str(c) for c in [idx, seq_file, w, k, c0, c1])), file=f)
    #  global mlock, task_list
    with mlock:
        task_list.remove(idx)
        print(ctime(), "{} jobs remaining. {}".format(len(task_list),
                    "[List too long]" if (len(task_list) >= 10) else task_list))

def work_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seqfile")
    parser.add_argument("--kmin", type=int)
    parser.add_argument("--kmax", type=int)
    parser.add_argument("-w", "--wval", type=int, required=True)
    parser.add_argument("-p", "--processes", type=int, required=True)
    args = parser.parse_args()
    print(args)
    if args.seqfile is None:
        seqs = ["hg38_all"]
    else:
        seqs = [args.seqfile]
    if args.kmin is None:
        kvals = [15, 16, 17, 18, 19]
    else:
        kvals = list(range(args.kmin, args.kmax + 1))
    wvals = [args.wval]
    fs_pairs = {
                "multi_lax": ("multi_random_pass", 0.6)
                ,
                "baseline": ("single_pass_random_order", -1)
                #  ,"baseline-nf": ("single_pass_random_order", -2)
                #  ,"simple": ("single_pass_random_order", 0)
                #  ,"multi_lax_test": ("multi_random_pass", -0.6)
                #  ,"multi_strict": ("multi_random_pass", 0.9)
                }
    worklist = []  # type: List
    for i in trange(2):
        pass  # trange compat test
    seq_dlen = dict()
    for seq in seqs:
        print(ctime(), "Started loading:", seq)
        preload_dir = working_dir + "preload/{}_30.dump".format(seq)
        seq_n = load_shared_partial_SA(preload_dir, seq)
        seq_dlen[seq] = seq_n
        print(ctime(), "Finished loading:", seq, "data length:", seq_n)
    skipped_work = []
    seen_work = set()
    if os.path.exists(master_data):
        with open(master_data) as f:
            for line in f:
                item = line.split(',')[0]
                seen_work.add(item)
    for name, fsitem in fs_pairs.items():
        for s in seqs:
            for k in kvals:
                for w in wvals:
                    fname, strat = fsitem
                    idx = "{}_{}_{}_{}".format(name, s, w, k)
                    if (idx in seen_work) and (seq_dlen[s] > 5000000):
                        skipped_work.append(idx)
                    else:
                        worklist.append((idx, s, seq_dlen[s], w, k, fname, strat))
    print("list of works:", worklist)
    print("skipped works:", skipped_work)
    gc.collect()
    # Load partial SAs to shared memory.
    # now don't shuffle, these are in the order we desire (baseline ->
    # current)
    #  random.shuffle(worklist)
    _tl = list(x[0] for x in worklist)
    task_list = mgr.list(_tl)
    with open(master_log, "a") as f:
        print(ctime(), "works started. List of works:", worklist, file=f)
    if args.processes is None:
        p = 10
    else:
        p = args.processes
    with Pool(processes=p, maxtasksperchild=1) as pool:
        pool.map(new_worker_func, worklist)
    #  for item in worklist:
        #  new_worker_func(item)


if __name__ == "__main__":
    work_main()

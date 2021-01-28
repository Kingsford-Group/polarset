from anchor_distributed import master_data, working_dir
import matplotlib.pyplot as plt

names =  {"baseline": "Fixed Interval Sampling",
         # "baseline-nf": "Baseline",
         #  "simple": "One-Round",
         #  "multi_strict": "Layered Strict",
          "miniception": "Miniception",
         "multi_lax": "Layered Polar Sets",
          "multi_lax_incl": "(Test) Extra Rounds"}

target_seq = "hg38_all"
kvals = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

all_dat = dict()
edat = dict()



def plot_res_single_w(seq, wval, kvals, names, skip_energy = False,
                      subplot = False):
    '''
        Create plot for single value of w and varying k.
        Pulls data from global variables all_dat and edat.
        @param seq: name of sequence file.
        @param wval: the value of w.
        @param kvals: values of k, in increasing order.
        @param names: dictionary of {item_name : display_name}.
        @param skip_energy: if using 2/(w+1) as random density.
        @param subplot: if this is a subplot.
    '''
    if not subplot:
        plt.clf()
    random_df = []
    if not skip_energy:
        for k in kvals:
            sp, df = edat[(seq, wval, k)]
            random_df.append(2 + (sp - df) * (wval + 1))
    else:
        random_df = [2] * len(kvals)
    plt.plot(kvals, random_df, label="Random Minimizers", linestyle='solid', marker='o', markersize=4)
    plt.plot(kvals, [1 + 1 / wval] * len(kvals), '--', label="Lower Bound")
    plt.xticks(kvals)
    for sname, slabel in names.items():
        xs = []
        ys = []
        for k in kvals:
            iname = (sname, seq, wval, k)
            if iname in all_dat:
                xs.append(k)
                ys.append(all_dat[iname] * (wval + 1))
        if len(xs) > 0:
            plt.plot(xs, ys, linestyle='solid', marker='o', label=slabel, markersize=4)
        #  plt.plot(kvals, list(x * (wval + 1) for x in dfs), label=slabel)
    plt.xlabel("Value of k (w={})".format(wval))
    plt.ylabel("Density factor")
    if not subplot:
        plt.legend()
        plt.savefig("figures/{}_{}.pdf".format(seq, wval))

def plot_energy_single_w(seq, wval, kvals, subplot = False):
    '''
        Create sequence energy plots, for a single value of w and varying k.
        Pulls data from global variables edat.
        @param seq: name of sequence file.
        @param wval: value of w.
        @param kvals: values of k, in increasing order.
        @param subplot: if this is part of a subplot.
    '''
    if not subplot:
        plt.clf()
    sps = []
    dfs = []
    for k in kvals:
        sp, df = edat[(seq, wval, k)]
        sps.append(sp * (wval + 1))
        dfs.append(0 - df * (wval + 1))
    plt.plot(kvals, [0] * len(kvals), 'k', label="Zero")
    plt.plot(kvals, sps, label="Energy Surplus", linestyle="solid",
             marker='o', markersize=4)
    plt.plot(kvals, dfs, label="Energy Deficit", linestyle="solid",
             marker='o', markersize=4)
    plt.xlabel("Value of k (w={})".format(wval))
    plt.xticks(kvals)
    plt.ylabel("Density factor")
    plt.ylim((-0.01, 0.01))
    if not subplot:
        plt.legend()
        plt.savefig("figures/{}_{}_energy.pdf".format(seq, wval))

def parallel_plot(plot_func, p0, p1, fn, bbox, rect, leg_cols=3):
    '''
    Generate parallel plots as seen in the manuscript.
    @param plot_func: the plotting function to be used.
    @param p0, p1: list of parameters feed to plot_func.
    @param fn: output file name.
    @param bbox, rect: parameters for plt.figlegend and plt.tight_layout to control figure size.
    @param leg_cols: parameter for plt.figlegend(). Defaults to 3.
    '''
    import matplotlib
    matplotlib.rcParams.update({"font.size":6})
    plt.clf()
    plt.figure(figsize=(4, 3))
    plt.subplot(121)
    plot_func(*p0, subplot = True)
    plt.figlegend(loc="lower center", ncol=leg_cols, bbox_to_anchor=bbox)
    plt.subplot(122)
    plot_func(*p1, subplot = True)
    plt.tight_layout(rect=rect)
    plt.savefig("figures/{}.pdf".format(fn))

def parse_old_logs(idx, w, n):
    '''
    Reads the old logs to recover the density factor estimates.
    @param idx: index of the log file.
    @param w: the value of w used.
    @param n: the number of k-mers in the sequence.
    @return: list of density factors.
    '''
    ret = []
    with open(working_dir + idx + ".log") as f:
        for line in f:
            if "Coverage" in line:  # int rounds
                l = line.split()
                cov = int(l[l.index("Coverage") + 2])
                seg = int(l[l.index("segments") + 2])
                ele = int(l[l.index("elements") + 2])
                ret.append(2 - (cov * 2  - (seg + ele) *(w+1)) / n )
            elif "Energy Information" in line:  # final round
                l = line.split()
                ds = float(l[l.index("Current") + 2][:-1])
                ret.append(ds)
    if len(ret) == 14:
        ret = list(ret[0:14:2])
    return ret

def parse_trajectory(policy, seq, wval, kvals, subplot = False):
    assert seq == "hg38_all"
    hg38_len = 3049315783
    seq_len = hg38_len
    if not subplot:
        plt.clf()
    xs = list(range(8))[1:]
    plt.plot(xs, [2] * len(xs), 'k', label="Start")
    plt.plot(xs, [1 + 1 / wval] * len(xs), 'k--', label="Lower Bound")
    for k in kvals:
        idx = "{}_{}_{}_{}".format(policy, seq, wval, k)
        dat = parse_old_logs(idx, wval, seq_len - k + 1)
        #  dat = [2] + dat
        if k == 25:
            plt.plot(xs, dat, "navy", label="k=" + str(k), linestyle='solid', linewidth=1.2, marker='o', markersize=3)
        else:
            plt.plot(xs, dat, label="k=" + str(k), linestyle='solid', linewidth=1.2, marker='o', markersize=3)
    plt.xlabel("Round No. (hg38, w={})".format(wval))
    plt.ylabel("Estimated Density Factor")
    plt.xticks(xs)
    if not subplot:
        plt.legend()
        plt.savefig("figures/prog/{}_{}_{}_prog.pdf".format(policy, seq, wval))

def parse_long_trajectory():
    '''
    a temporary function for extra long trajectories
    '''
    policy="multi_lax"
    seq = "hg38_all"
    seq_len = 3049315783
    wval = 100
    plt.clf()
    for k in [15, 16, 17, 18]:
        idx = "{}_{}_{}_{}".format(policy, seq, wval, k)
        dat = parse_old_logs(idx, wval, seq_len - k + 1)
        if k == 15:
            dat = [2] + dat + list(2-x for x in [0.23907, 0.27626, 0.29981, 0.32087, 0.33445, 0.35128,
                               0.36209, 0.37246, 0.38132, 0.38766, 0.39340, 0.39418,
                               0.39801, 0.39995, 0.40379])
        elif k == 16:
            dat = [2] + dat + list(2-x for x in [0.46027, 0.47836, 0.48190, 0.48848,
                                                 0.49485, 0.50308, 0.50823, 0.51192,
                                                 0.51384, 0.51405, 0.51507, 0.51567,
                                                 0.51567, 0.51567, 0.51567])
        elif k == 17:
            dat = [2] + dat + list(2-x for x in [0.56462, 0.56823, 0.57139, 0.57190,
                                                 0.57406, 0.57543] + [0.57543] * 9)
        elif k == 18:
            dat = [2] + dat + list(2-x  for x in [0.62639, 0.63481, 0.63628, 0.64110,
                                                  0.64207, 0.64625] + [0.64225] * 9)
        plt.plot(dat, label="k="+str(k))
    plt.axvline(x=5, color='k', linestyle='-.')
    plt.axvline(x=7, color='k')
    plt.xlabel("Round No. (hg38, w={})".format(wval))
    plt.ylabel("Estimated Density Factor")
    plt.legend()
    plt.savefig("figures/prog/tmp_long_prog.pdf".format(policy, seq, wval))


if __name__ == "__main__":
    #  parse_long_trajectory()
    #  quit()
    #  parse_trajectory("multi_lax", "hg38_all", 10, range(15, 26))
    #  parse_trajectory("multi_lax", "hg38_all", 100, range(15, 26))
    parallel_plot(parse_trajectory, ("multi_lax", "hg38_all", 10, kvals),
                  ("multi_lax", "hg38_all", 100, kvals), "hg38_traj",
                  bbox=(0.52, 0), rect=[0, 0.15, 1, 1], leg_cols=5)
    quit()

    with open(master_data) as f:
        for l in f:
            items = l.split(sep=',')
            iname, seq, w, k, c0, c1 = items
            w, k, c0 = int(w), int(k), float(c0)
            s = None
            for kw in names:
                if iname.startswith(kw):
                    s = kw
            all_dat[(s, seq, w, k)] = c0

    with open(working_dir + "random_results.dat") as f:
        for l in f:
            items = l.split(sep=',')
            iname, seq, w, k, c0 = items
            w, k, c0 = int(w), int(k), float(c0)
            s = None
            for kw in names:
                if iname.startswith(kw):
                    s = kw
            all_dat[(s, "random", w, k)] = c0


    with open(working_dir + "estats.dat") as f:
        for l in f:
            items = l.split(sep=',')
            seq, w, k, sp, df = items
            w, k, sp, df = int(w), int(k), float(sp), float(df)
            edat[(seq, w, k)] = (sp, df)

    with open(working_dir + "miniception.dat") as f:
        for l in f:
            items = l.split(sep=',')
            seq, w, k, d = items
            w, k, d = int(w), int(k), float(d)
            all_dat[("miniception", seq, w, k)] = d

    #  with open(working_dir + "incremental.dat") as f:
        #  for l in f:
            #  items = l.split(sep=',')
            #  iname, seq, w, k, c0, c1 = items
            #  if iname.endswith("+0"):
                #  continue
            #  w, k, c0 = int(w), int(k), float(c0)
            #  s = None
            #  for kw in names:
                #  if iname.startswith(kw):
                    #  s = kw + "_incl"
            #  all_dat[(s, seq, w, k)] = c0

    #  all_dat[("multi_lax_incl", "hg38_all", 100, 15)] = 0.015825
    #  all_dat[("multi_lax_incl", "hg38_all", 100, 16)] = 0.014712


    #  del names["miniception"]
    #  parallel_plot(plot_res_single_w, ("hg38_all", 10, kvals, names), ("hg38_all", 100, kvals, names),
                  #  "hg38_all_res", bbox=(0.52, 0), rect=[0, 0.1, 1, 1])
    #  plot_res_single_w("chrX", 100, kvals, names, skip_energy=True)
    #  plot_res_single_w("chrX", 10, kvals, names, skip_energy=True)
    #  quit()

    #  _scramble_kvals = list(range(10, 21))
    parallel_plot(plot_res_single_w, ("scramble", 10, kvals, names, True), ("scramble", 100, kvals, names, True),
                  "scramble", bbox=(0.52, 0), rect=[0, 0.1, 1, 1])
    quit()

    parallel_plot(plot_energy_single_w, ("hg38_all", 10, kvals), ("hg38_all", 100, kvals),
                  "hg38_estat", bbox=(0.55, 0), rect=[0, 0.05, 1, 1])
    parallel_plot(plot_res_single_w, ("hg38_all", 10, kvals, names), ("hg38_all", 100, kvals, names),
                  "hg38_all_res", bbox=(0.52, 0), rect=[0, 0.1, 1, 1])
    parallel_plot(plot_res_single_w, ("chr1", 10, kvals, names), ("chr1", 100, kvals, names),
                  "chr1", bbox=(0.52, 0), rect=[0, 0.1, 1, 1])
    #  del names['miniception']
    parallel_plot(plot_res_single_w, ("random", 10, kvals, names), ("random", 100, kvals, names),
                  "random_res", bbox=(0.52, 0), rect=[0, 0.1, 1, 1])
    #  plot_energy_single_w("hg38_all", 10, kvals)
    #  plot_energy_single_w("hg38_all", 100, kvals)
    #  plot_res_single_w("hg38_all", 10, kvals, names)
    #  plot_res_single_w("hg38_all", 100, kvals, names)
    #  plot_energy_single_w("chr1", 10, kvals)
    #  plot_energy_single_w("chr1", 100, kvals)
    #  plot_res_single_w("chr1", 10, kvals, names)
    #  plot_res_single_w("chr1", 100, kvals, names)
    #  plot_res_single_w("chr1", 100, list(range(7, 15)), names, skip_energy=True)

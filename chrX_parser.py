sfile = "/mnt/disk04/gus/t2t_human/chm13.chrX_v0.7.fasta"

def parse_file():
    lines = []
    with open(sfile) as f:
        for line in f:
            l = line.strip()
            if not l.startswith(">"):
                lines.append(l)
    seq = "".join(lines)
    ct_seq = seq[57828561:60934693]
    with open("chrX.seq", "w") as f:
        print(seq, file=f)
    with open("chrXC.seq", "w") as f:
        print(ct_seq, file=f)

def validate():
    with open("chrX.seq") as f:
        s = f.readline().strip()
        print("Length:", len(s))
        for c in s:
            assert c in 'ACGT'

if __name__ == "__main__":
    parse_file()
    print("parsing finished")
    #  validate()
    #  print("validation finished")

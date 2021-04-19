#!/usr/bin/env python

def fix_tab(x, nt=8):
    while '\t' in x:
        iid = x.index('\t')
        ntab = nt if iid % nt == 0 else nt - iid % nt
        x = x[:iid] + (" " * ntab) + x[iid + 1:]
    return x

import sys
import os

for f in sys.argv[1:]:
    if os.path.isfile(f):
        ll = open(f, 'r').readlines()
        open(f, 'w').writelines([fix_tab(l) for l in ll])

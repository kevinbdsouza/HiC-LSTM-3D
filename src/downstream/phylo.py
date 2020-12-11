import pandas as pd
import re
import numpy as np
import os
from common import data_utils


class PhyloP:
    def __init__(self, cfg, phylo_path, chr):
        self.cfg = cfg
        self.chr = chr
        self.bigwig_path = os.path.join(phylo_path, "hg19.100way.phyloP100way.bw")
        self.bedgraph_path = os.path.join(phylo_path, str(chr), "phyloP100way.chr" + str(chr) + ".bedgraph")
        self.npz_path = os.path.join(phylo_path, str(chr), "chr" + str(chr) + ".phyloP100way.npz")
        self.bwtobd = "/data2/latent/bigWigToBedGraph"

    def get_phylo_data(self):
        os.system(
            "/data2/latent/bigWigToBedGraph {} {} -chrom=chr{}".format(self.bigwig_path, self.bedgraph_path, self.chr))

        data = data_utils.bedgraph_to_dense(self.bedgraph_path, verbose=True)
        data = data_utils.decimate_vector(data, k=10000)
        np.savez(self.npz_path, data)

        os.system("rm {}".format(self.bedgraph_path))
        return

    def filter_phylo_data(self):
        phylo_data = None

        return phylo_data


if __name__ == '__main__':
    phylo_path = "/opt/data/latent/data/downstream/PE-interactions"

    chromosome = 'chr21'
    cell_name = 'E003'
    cfg = None

    pe_ob = PhyloP(cfg)
    pe_ob.get_phylo_data(phylo_path)
    print("done")

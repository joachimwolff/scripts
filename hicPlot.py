#!/usr/bin/env python
import plotly

import plotly.plotly as py
import plotly.graph_objs as go
import argparse
import numpy as np
import hicexplorer.HiCMatrix as HiCMatrix
import hicexplorer.hicPlotMatrix as HiCPlotMatrix

def parse_arguments(args=None):
    parser=argparse.ArgumentParser(description='')
    parser.add_argument('--matrix', '-m',
                        help='Path of the Hi-C matrix to plot',
                        required=True)
    parser.add_argument('--region',
                        help='Plot only this region. The format is '
                        'chr:start-end The plotted region contains '
                        'the main diagonal and is symmetric unless '
                        ' --region2 is given'
                        )
    return parser

def plot_heatmap(pMatrix, pFilename):
    data = [
        go.Heatmap(
            z=pMatrix, x=None, y=None
                )
            ]
    plotly.offline.plot(data, filename=pFilename, auto_open=False)


def main(args=None):
    args = parse_arguments().parse_args(args)
    ma = HiCMatrix.hiCMatrix(args.matrix)
    # matrix = np.asanyarray(ma.getMatrix().astype(float))
    # print matrix
    if args.region:
        chrom, region_start, region_end = HiCPlotMatrix.translate_region(args.region)
        if chrom not in ma.interval_trees.keys():
            chrom = change_chrom_names(chrom)
            if chrom not in ma.interval_trees.keys():
                exit("Chromosome name {} in --region not in matrix".format(change_chrom_names(chrom)))

        args.region = [chrom, region_start, region_end]
        idx1, start_pos1 = zip(*[(idx, x[1]) for idx, x in enumerate(ma.cut_intervals) if x[0] == chrom and
                x[1] >= region_start and x[2] < region_end])
        
        idx2 = idx1
        chrom2 = chrom
        start_pos2 = start_pos1
        # select only relevant part
        matrix = np.asarray(ma.matrix[idx1, :][:, idx2].todense().astype(float))
    # matrix = np.asarray(ma.matrix[idx1, :][:, idx2].todense().astype(float))
    matrix = np.nan_to_num(matrix)
    matrix = np.log(matrix)
    plot_heatmap(matrix, "hicPlot1.html")


if __name__ == "__main__":
    main()


#!/usr/bin/env python
import plotly
from itertools import groupby

from operator import itemgetter
import plotly.plotly as py
import plotly.graph_objs as go
import argparse
import numpy as np
import hicexplorer.HiCMatrix as HiCMatrix
import hicexplorer.hicPlotMatrix as HiCPlotMatrix
import matplotlib.pyplot as plt
from scipy import ndimage
# from sklearn.cluster import spectral_clustering
from sklearn.cluster import AffinityPropagation
from pylab import *
from itertools import cycle
from sklearn.feature_extraction import image
import math
def parse_arguments(args=None):
    parser=argparse.ArgumentParser(description='')
    parser.add_argument('--matrix', '-m',
                        help='Path of the Hi-C matrix to plot',
                        required=False)
    parser.add_argument('--region',
                        help='Plot only this region. The format is '
                        'chr:start-end The plotted region contains '
                        'the main diagonal and is symmetric unless '
                        ' --region2 is given'
                        )
    return parser

def change_chrom_names(chrom):
    """
    Changes UCSC chromosome names to ensembl chromosome names
    and vice versa.
    """
    # TODO: mapping from chromosome names like mithocondria is missing
    if chrom.startswith('chr'):
        # remove the chr part from chromosome name
        chrom = chrom[3:]
    else:
        # prefix with 'chr' the chromosome name
        chrom = 'chr' + chrom

    return chrom
def plot_heatmap(pMatrix, pFilename, pEdgeMatrix):
    data = [
        go.Heatmap(
            z=pMatrix, x=None, y=None
                )
            ]
    plotly.offline.plot(data, filename=pFilename, auto_open=False)
    # plt.imshow(pMatrix, cmap='hot', interpolation='nearest')
    # plt.show()
    # data = pEdgeMatrix
    # plotly.offline.plot(data, filename=pFilename+"Edge.html", auto_open=False)

def gradient(pMatrix):
    matrix = ndimage.gaussian_filter(pMatrix, 8)
    Sx = ndimage.sobel(pMatrix, axis=0, mode="constant")
    Sy = ndimage.sobel(pMatrix, axis=1, mode="constant")
    return Sx, Sy

def edges(pMatrix):
    Sx, Sy = gradient(pMatrix)
    return np.hypot(Sx, Sy)
def extract_diagonal(pMatrix, pWidth):
    diagonal = np.empty([len(pMatrix), pWidth])
    # print diagonal
    
    for i in xrange(len(pMatrix)):
        for j, k in zip(xrange(pWidth, 0, -1), xrange(0, pWidth, 1)):
            # print j, "::", k
            if i - j < 0:
                diagonal[i, k] = 0
            else:
                diagonal[i, k] = pMatrix[i, i - j]
        diagonal[i, pWidth - 1] = 0
    return diagonal.T

def direction_edges(pGradientX, pGradientY):
    return atan2(pGradientX, pGradientY)

def corner(pMatrix):
    Sx = np.gradient(pMatrix, axis = 0)
    # Sx = np.gradient(Sx, axis = 0)
    
    Sy = np.gradient(pMatrix, axis = 1)
    # Syy = np.gradient(Syy, axis = 1)
    
    return Sx + Sy
    # np.gradient(pMatrix)
    

def main(args=None):
    args = parse_arguments().parse_args(args)
    ma = HiCMatrix.hiCMatrix(args.matrix)

    # list_numbers = np.array([12,12,12,12,12,24,24,24,24], dtype=np.float)
    # mask = np.array([-1, 0, 1], dtype=np.float)
    
    # print "f'(x) = f(x+1) - f(x)"
    # print list_numbers
    # print np.gradient(list_numbers)
    # print np.convolve(list_numbers, mask)    
    matrix = np.asanyarray(ma.getMatrix().astype(float))
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
    # matrix = np.log(matrix)
    # matrix_norm = matrix * (255.0/matrix.max())
    # high_values_indices = matrix_norm >  150  # Where values are low
    # matrix_norm[high_values_indices] = 255
    # print edges_matrix
    # edges_matrix = edges(matrix)
    
    matrix_log = np.log(matrix)
    # matrix = ndimage.gaussian_filter(matrix_log, 1)
    
    # direction_edges()
    # plot_heatmap(matrix_log, "hicPlot.html", edges_matrix)
    # plot_heatmap(edges_matrix, "hicPlot_edges.html", edges_matrix)
    
    # matrix_norm = matrix * (255.0/matrix.max())
    # matrix_norm = np.log(matrix_norm)
    
    # plot_heatmap(matrix_norm, "hicPlot_normalized.html", edges_matrix)
    
    # matrix_norm_log = matrix_log * (255.0/matrix_log.max())
    # # matrix_norm_log = np.log(matrix_norm_log)
    # plot_heatmap(matrix_norm_log, "hicPlot_normalized_log.html", edges_matrix)
    # edges_matrix = edges(matrix_norm_log)
    # plt.matshow(edges_matrix)
    # matrix_log_edges = matrix_log * (255.0/matrix_log.max())
    # matrix_log_edges = 255 - matrix_log_edges
    edges_matrix = edges(matrix_log)
    edges_matrix = 255 - edges_matrix
    # edges_matrix = 0.01 * edges_matrix
    # plt.figure(1)
    # plt.imshow(edges_matrix, interpolation='none')
    # plt.show()

    # from scipy import misc
    # f = misc.face()
    # misc.imsave('face.png', f) # uses the Image module (PIL)
    # low_values_indices = edges_matrix < 100  # Where values are low
    # edges_matrix[low_values_indices] = 0

    # high_values_indices = edges_matrix > 100  # Where values are low
    # edges_matrix[high_values_indices] = 255

    # edges_inv = 255 - edges_matrix
    # print edges_matrix
    high_values_indices = edges_matrix >  150  # Where values are low
    edges_matrix[high_values_indices] = 255

    high_values_indices = edges_matrix <  150  # Where values are low
    edges_matrix[high_values_indices] = 40

    high_values_indices = edges_matrix <  120  # Where values are low
    edges_matrix[high_values_indices] = 20


    high_values_indices = edges_matrix <  80  # Where values are low
    edges_matrix[high_values_indices] = 0

    ###### find minimas
    edges_line = extract_diagonal(edges_matrix, 50)
    # edges_line = np.nan_to_num(edges_line)
    
    # print edges_line
    # print edges_matrix
    corners =  corner(edges_line)
    print corners
    from copy import deepcopy
    corners_plot = deepcopy(corners)
    # corner_indices = corners == 0
    # none_corner_indices = corners != 0
    corners = np.nan_to_num(corners)
    # corners[corner_indices] = 0
    # corners[none_corner_indices] = 255
    # print corners_plot
    corner_values = np.zeros(len(corners[0]))
    for i in xrange(len(corners)):
        for j in xrange(len(corners[i])):
            corner_values[j] += corners[i][j]
    corner_values /= len(corner_values)        
    cluster_ranges = []
    for i in xrange(len(corners[-5])):
        if corners[-5][i] == 0:
            cluster_ranges.append(i)
    cluster_borders = [[]]
    for k, g in groupby(enumerate(cluster_ranges), lambda (i, x): i-x):
        tmp_list = map(itemgetter(1), g)
        if len(tmp_list) > 2:
            cluster_borders[0].append([tmp_list[1],tmp_list[-1]]) 
        # print [foo[1], foo[-1]]

    print cluster_borders

    # hierachical_clusters = []
    preferred_partner = {0:1, len(cluster_borders[0])-1 : len(cluster_borders[0])-2}
    # diff_left = cluster_borders[1][0] - cluster_borders[0][1] 
    j = 0
    while len(cluster_borders[j]) >= 2:
        for i in xrange(1, len(cluster_borders[j]) - 1, 1):
            diff_left = cluster_borders[j][i][0] - cluster_borders[j][i-1][1]
            diff_right = cluster_borders[j][i+1][0] - cluster_borders[j][i][1]
            if diff_left < diff_right:
                preferred_partner[i] = i - 1
            else:
                preferred_partner[i] = i + 1
        
        cluster_borders.append([])
        # print preferred_partner
        if len(preferred_partner) > 0:
            for i in xrange(0, len(cluster_borders[j])):
                # print i 
                # print preferred_partner[i]
                # print preferred_partner[preferred_partner[i]]
                # print "\n\n"
                if i == preferred_partner[preferred_partner[i]]:
                    cluster_borders[j+1].append([cluster_borders[j][i][0], cluster_borders[j][preferred_partner[i]][-1]])
                elif preferred_partner[preferred_partner[i]] != -1:
                    cluster_borders[j+1].append(cluster_borders[j][i])
                preferred_partner[i] = -1
        # print cluster_borders
        preferred_partner.clear()
        # if ()
        # print "Iteration: ", j
        # print "Cluster: ", cluster_borders[j+1]
        preferred_partner = {0:1, len(cluster_borders[j+1])-1 : len(cluster_borders[j+1])-2}
        j += 1

    print cluster_borders



    print corners
   

    fig, ax = plt.subplots()
    plt.title(args.region, fontsize=14, fontweight='bold')
    ax.imshow(matrix_log, cmap='hot', interpolation='none', )
    # edges_matrix = np.ma.masked_where(edges_matrix > 150, edges_matrix)    
    # ax.set_facecolor("red")
    # edges_matrix_rgb = to_rgb4(edges_matrix)
    ax.matshow(edges_matrix)
    
    plt.savefig("foo.png", dpi=600)
    
    plt.cla()
    ax.imshow(matrix_log , cmap='hot')
    plt.savefig("matrix_log.png", dpi=600)

    plt.cla()
    ax.matshow(edges_matrix , cmap='gray')
    plt.savefig("edges.png", dpi=600)

    plt.cla()
    ax.matshow(matrix , cmap='hot')
    plt.savefig("matrix_gauss.png", dpi=600)

    
    plt.cla()
    ax.matshow(edges_line , cmap='gray')
    plt.savefig("edges_line.png", dpi=600)

    
    plt.cla()
    fig, (ax1, ax2)  = plt.subplots(2, 1)
    x = range(0, len(corners_plot[-5]))
    # x = [0] * len(corners_plot[-1])
    # print x
    ax1.matshow(corners , cmap='gray')
    ax2.fill_between(x, 0, corner_values)
    plt.savefig("corners.png", dpi=600)

    from scipy.io import mmwrite
    mmwrite("corners", corners)
    mmwrite("edges", edges_line)
    np.save("corners_second_gradient", corners_plot[-5])
    
    print len(corners_plot[-5])

    from scipy.cluster import hierarchy
    # import matplotlib.pyplot as plt
   
    Z = hierarchy.linkage(cluster_borders[0], 'single')
    plt.figure()
    dn = hierarchy.dendrogram(Z, count_sort=True)
    print "dn: ", dn
    plt.savefig("Dendogram.png", dpi=600)
def to_rgb4(im):
    # we use weave to do the assignment in C code
    # this only gets compiled on the first call
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret
if __name__ == "__main__":
    main()


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
                diagonal[i, k] = 255
            else:
                diagonal[i, k] = pMatrix[i, i - j]
        diagonal[i, pWidth - 1] = 0
    return diagonal.T

def direction_edges(pGradientX, pGradientY):
    return atan2(pGradientX, pGradientY)

def corner(pMatrix):
    Sxx = np.gradient(pMatrix, axis = 0)
    Sxx = np.gradient(Sxx, axis = 0)
    
    Syy = np.gradient(pMatrix, axis = 1)
    Syy = np.gradient(Syy, axis = 1)
    
    return Sxx + Syy
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

    # corner_indices = corners == 0
    none_corner_indices = corners != 0
    
    # corners[corner_indices] = 0
    corners[none_corner_indices] = 255

    cluster_ranges = []
    for i in xrange(len(corners[-5])):
        if corners[-5][i] == 0:
            cluster_ranges.append(i)
    cluster_borders = []
    for k, g in groupby(enumerate(cluster_ranges), lambda (i, x): i-x):
        tmp_list = map(itemgetter(1), g)
        if len(tmp_list) > 2:
            cluster_borders.append([tmp_list[1], tmp_list[-1]]) 
        # print [foo[1], foo[-1]]

    print cluster_borders

    hierachical_clusters = []
    for i in xrange(1, len(cluster_borders), 1):
         diff_left = cluster_borders[i][0] - cluster_borders[i-1][1]
         diff_right = cluster_borders[i+1][0] - cluster_borders[i][1]
         
 
    # print cluster_ranges
    # high_values_indices = corners ==  0  # Where values are low
    # print high_values_indices
    # corners[high_values_indices] = 255

    # for i in xrange(0, len(corners)):
    #     for j in xrange(0, len(corners[i])):
    #         if 
    print corners
    # print corners
    # high_values_indices = corners <  0  # Where values are low
    # corners[high_values_indices] = 255

    # edges_graph = image.img_to_graph(edges_inv)
    # edges_graph = edges_graph.tocsr()
    # print edges_graph
    # y_predict = AffinityPropagation(preference=-50).fit_predict(edges_graph)
   
    # colors = np.array(['b','g','r','c','m','y','k']* ((n_clusters_ / 7) + 1))


    # colors = colors.split("")
    # print colors
    # for 
    # for k, col in zip(xrange(n_clusters_), colors):
    #     class_members = labels == k
        # print "class_members: ", class_members
        # cluster_center = X[cluster_centers_indices[k]]
        # plt.plot(X[class_members, 0], X[class_members, 1], col)
    # for i in xrange(len(X)):
    #     for j in xrange(len(X[0])):
        # y_pred = af.predict(i, j)
    # plt.scatter(image_array, y_pred, c=y_pred, cmap='grey')
        # plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
        #         markeredgecolor='k', markersize=14)
        # for x in X[class_members]:
        #     # print [cluster_center[0], x[0]]
        #     # print [cluster_center[1], x[1]]
        #     plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

    # plt.title('Estimated number of clusters: %d' % n_clusters_)
    # plt.savefig("foo.png", dpi=1200)


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
    ax.matshow(corners , cmap='gray')
    plt.savefig("corners.png", dpi=600)
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


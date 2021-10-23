"""
SISTER
Space-based Imaging Spectroscopy and Thermal PathfindER
Author: Adam Chlus
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import numpy as np
import hytools_lite as htl
from hytools_lite.io.envi import WriteENVI
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from skimage.util import view_as_blocks
from numba import jit

def progbar(curr, total, full_progbar = 100):
    '''Display progress bar.
    Gist from:
    https://gist.github.com/marzukr/3ca9e0a1b5881597ce0bcb7fb0adc549
    Args:
        curr (int, float): Current task level.
        total (int, float): Task level at completion.
        full_progbar (TYPE): Defaults to 100.
    Returns:
        None.
    '''
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
    print('\r', '#'*filled_progbar + '-'*(full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')

def subsample(hy_obj,sample_size):
    sub_samples = np.zeros((hy_obj.lines,hy_obj.columns)).astype(bool)
    idx = np.array(np.where(hy_obj.mask['sample'])).T
    idxRand= idx[np.random.choice(range(len(idx)),int(len(idx)*sample_size), replace = False)].T
    sub_samples[idxRand[0],idxRand[1]] = True
    hy_obj.mask['samples'] = sub_samples

    X = []

    hy_obj.create_bad_bands([[300,400],[1300,1450],[1780,2000],[2450,2600]])
    for band_num,band in enumerate(hy_obj.bad_bands):
        if ~band:
            X.append(hy_obj.get_band(band_num,mask='samples'))
    return  np.array(X).T

@jit(nopython=True)
def calc_bray_curtis_blocks(image_blocks,nclusters):

    windows_pixels = image_blocks.shape[1]*image_blocks.shape[2]
    bray_curtis = np.full((image_blocks.shape[0],image_blocks.shape[0]),-4.0)

    i=0
    for nbhd1 in image_blocks:
        nbhd1 = nbhd1.flatten()
        cover1 = [0 for x in range(nclusters)]
        nbhd1_size = 0

        for a in nbhd1:
            if a !=nclusters:
                cover1[a]+=1
                nbhd1_size+=1
        if nbhd1_size/windows_pixels < .75:
            i+=1
            continue
        j=i
        for nbhd2 in image_blocks[i:]:
            nbhd2 = nbhd2.flatten()
            cover2 = [0 for x in range(nclusters)]
            nbhd2_size = 0

            for a in nbhd2:
                if a !=nclusters:
                    cover2[a]+=1
                    nbhd2_size+=1

            if nbhd2_size/windows_pixels < .75:
                j+=1
                continue

            numerator = 0
            for c in range(nclusters):
                numerator += np.abs(cover1[c]-cover2[c])

            bray_curtis[i,j] = numerator/(nbhd1_size+nbhd2_size)
            bray_curtis[j,i] = bray_curtis[i,j]
            j+=1
        i+=1
    return bray_curtis

@jit(nopython=True)
def calc_alpha(classes,window,nclusters):
    shannon = np.zeros(classes.shape)
    simpson = np.zeros(classes.shape)
    windows_pixels = window**2

    lines = classes.shape[0]
    columns = classes.shape[1]

    half_window = int(window/2)

    for line in range(half_window,lines-half_window):
        for col in range(half_window,columns-half_window):
            nbhd = classes[line-half_window:line+half_window,col-half_window:col+half_window].flatten()
            cover = [0 for x in range(nclusters)]
            nbhd_size = 0

            for a in nbhd:
                if a !=nclusters:
                    cover[a]+=1
                    nbhd_size+=1

            if nbhd_size/windows_pixels < .75:
                continue

            shn,smp = 0,0

            for c in range(nclusters):
                if cover[c] !=0:
                    p = cover[c]/nbhd_size
                    shn += p * np.log(p)
                    smp += p**2
            shannon[line,col] = -shn
            simpson[line,col] = 1/smp
    return shannon,simpson

def main():
    '''
    Spectral diversity metrics

    Adapted from:

        Féret, J. B., & de Boissieu, F. (2020).
        biodivMapR: An r package for α‐and β‐diversity mapping
        using remotely sensed images.
        Methods in Ecology and Evolution, 11(1), 64-70.
        https://doi.org/10.1111/2041-210X.13310

    '''

    desc = "Generate spectral diversity metrics"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('rfl_file', type=str,
                        help='Input reflectance image')
    parser.add_argument('out_dir', type=str,
                        help='Output directory')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--nclusters', type=int, default=25,
                        help='Number of k-means cluster')
    parser.add_argument('--window', type=int, default=10,
                        help='Window size for calculating diversity metrics')
    parser.add_argument('--pca', action='store_true',
                        help='Export PCA image')
    parser.add_argument('--species', action='store_true',
                        help='Export spectral species map')
    parser.add_argument('--ncpus', type=int, default=1,
                        help='Number of CPUs for MDS')

    args = parser.parse_args()

    rfl = htl.HyTools()
    rfl.read_file(args.rfl_file,'envi')

    rfl.mask['sample'] = rfl.mask['no_data']

    # Sample data
    X  = subsample(rfl,.1)

    # Center, scale and fit PCA transform
    x_mean = X.mean(axis=0)[np.newaxis,:]
    X -=x_mean
    x_std = X.std(axis=0,ddof=1)[np.newaxis,:]
    X /=x_std
    X = X[~np.isnan(X.sum(axis=1)) & ~np.isinf(X.sum(axis=1)),:]

    #Perform initial PCA fite
    pca = PCA(n_components=40)
    pca.fit(X)

    #Refit
    comps = np.argwhere(np.cumsum(pca.explained_variance_ratio_) > .99)[0][0] + 1
    pca = PCA(n_components=comps)
    pca.fit(X)

    #Generate PCA transformed image
    pca_transform = np.zeros((rfl.lines,rfl.columns,comps))
    iterator = rfl.iterate(by = 'chunk',chunk_size = (500,500))

    while not iterator.complete:
        chunk = iterator.read_next()
        X_chunk = chunk[:,:,~rfl.bad_bands].astype(np.float32)
        X_chunk = X_chunk.reshape((X_chunk.shape[0]*X_chunk.shape[1],X_chunk.shape[2]))
        X_chunk -=x_mean
        X_chunk /=x_std
        X_chunk[np.isnan(X_chunk) | np.isinf(X_chunk)] = 0
        pca_chunk=  pca.transform(X_chunk)
        pca_chunk = pca_chunk.reshape((chunk.shape[0],chunk.shape[1],comps))
        pca_chunk[chunk[:,:,0] == rfl.no_data] =0
        pca_transform[iterator.current_line:iterator.current_line+pca_chunk.shape[0],
                      iterator.current_column:iterator.current_column+pca_chunk.shape[1]] = pca_chunk

    #Export PCA image
    if args.pca:
        header = rfl.get_header()
        header['bands']= comps
        header['band names']= ['pca_%02d' % band for band in range(comps)]
        header['wavelength']= []
        header['fwhm']= []
        out_file = args.out_dir + rfl.base_name + '_pca'
        writer = WriteENVI(out_file,header)
        for band in range(comps):
            writer.write_band(pca_transform[:,:,band],band)

    #Cluster PCA data
    pca_sample= pca.transform(X.astype(float))
    clusters = KMeans(n_clusters=args.nclusters)
    clusters.fit(pca_sample)
    classes = clusters.predict(pca_transform.reshape(rfl.lines*rfl.columns,comps)).reshape(rfl.lines,rfl.columns)
    classes[~rfl.mask['no_data']] = args.nclusters

    if args.species:
        header = rfl.get_header()
        header['bands']= 1
        header['band names']= ['species']
        header['wavelength']= []
        header['fwhm']= []
        out_file = args.out_dir + rfl.base_name + '_species'
        writer = WriteENVI(out_file,header)
        writer.write_band(classes,0)

    if args.verbose:
        print("Calculating alpha diversity metrics.....")

    shannon,simpson = calc_alpha(classes,args.window,args.nclusters)
    shannon[~rfl.mask['no_data']] = -9999
    simpson[~rfl.mask['no_data']] = -9999

    #Export alpha diversity indices
    out_file = args.out_dir + rfl.base_name + '_alpha_diversity'
    header = rfl.get_header()
    header['bands']= 2
    header['band names']= ['shannon','simpson']
    writer = WriteENVI(out_file,header)
    writer.write_band(shannon[:,:],0)
    writer.write_band(simpson[:,:],1)

    #Run beta diversity questions
    column_end =args.window*(rfl.columns//args.window)
    line_end = args.window*(rfl.lines//args.window)

    image_blocks = view_as_blocks(classes[:line_end,:column_end],(args.window,args.window))
    image_blocks = image_blocks.reshape(image_blocks.shape[0]*image_blocks.shape[1],
                                        args.window,args.window)
    if args.verbose:
        print("Calculating dissimilarity matrix.......")

    bray_curtis = calc_bray_curtis_blocks(image_blocks,args.nclusters)

    #Filter dissimilarity matrix
    bc_filt = bray_curtis[:,bray_curtis.sum(axis=0) != bray_curtis.shape[1]*-4]
    bc_filt = bc_filt[bc_filt.sum(axis=1) != bc_filt.shape[1]*-4,:]

    if args.verbose:
        print("Performing multidimensional scaling.....")

    nmds = MDS(n_components=3,n_jobs=args.ncpus)
    nmds.dissimilarity = 'precomputed'
    coords = nmds.fit_transform(bc_filt)

    full_coords = np.zeros((bray_curtis.shape[0],3))
    full_coords[bray_curtis.sum(axis=0) != bray_curtis.shape[0]*-4] = coords

    image_blocks = view_as_blocks(classes[:line_end,:column_end],(args.window,args.window))
    beta = full_coords.reshape((image_blocks.shape[0],image_blocks.shape[1],3))
    beta[image_blocks.mean(axis = (2,3))<0] = np.nan

    header = rfl.get_header()
    header['bands']= 3
    header['band names']= ['species']
    header['wavelength']= []
    header['lines']= beta.shape[0]
    header['samples']= beta.shape[1]
    pixel_x,pixel_y =header['map info'][5:7]
    header['map info'][5:7] = [str(float(pixel_x)*args.window),
                               str(float(pixel_y)*args.window)]
    header['data ignore value'] = 0

    out_file = args.out_dir + rfl.base_name + '_beta_diversity'
    writer = WriteENVI(out_file,header)
    for band in range(3):
        writer.write_band(beta[:,:,band],band)

if __name__== "__main__":
    main()

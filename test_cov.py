from cosmoprimo.fiducial import DESI
import sys
sys.path.append('/home/r/rbond/jiaqu/Thumbstack_DESI/')
import universe
# reload(universe)
from universe import *

import mass_conversion
# reload(mass_conversion)
from mass_conversion import *

import catalog
# reload(catalog)
from catalog import *

import thumbstack
# reload(thumbstack)
from thumbstack import *

import cmb
# reload(cmb)
from cmb import *


def map_function(pathMap):
    result = enmap.read_fits(pathMap)
    # if len(result.shape)>2:
    #     result = result[1]
    return result

def mask_function(pathMask):
    result = enmap.read_fits(pathMask)
    # if len(result.shape)>2:
    #     result = result[1]
    return result

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--random", action='store_true',help='used random catalog')
parser.add_argument("--savename",type=str,default=None)
parser.add_argument("--catalogue",type=str,default="full_catalog_Y3_test_swap.txt")
parser.add_argument("--output-dir", type=str,  default=None,help='Output directory.')

args = parser.parse_args()


cmbMap = enmap.read_fits("/gpfs/fs0/project/r/rbond/msyriac/ilc_dr6v3/20230606/hilc_fullRes_TT_17000.fits")
cmbMask = enmap.read_fits("/gpfs/fs0/project/r/rbond/msyriac/ilc_dr6v3/20230606/wide_mask_GAL070_apod_1.50_deg_wExtended.fits")

mask_5sigma_1 = 1.*(cmbMap<5*np.std(cmbMap))
mask_5sigma_2 = 1.*(cmbMap>-5*np.std(cmbMap))
cmbMask_2 = cmbMask*mask_5sigma_1*mask_5sigma_2

def run_analysis(nproc = 1, test=False,
                 catpath = "/home/r/rbond/jiaqu/sims/DESI/catalogue/"):
    outpath = '/home/r/rbond/jiaqu/sims/DESI/'
    tsname=args.savename
    log_fn = outpath+'%s_args.log'
    u = DESI()
    massConv = MassConversionKravtsov14()

    if test:
        nObj = 1 #ensures unequal pos/neg weights
        do_bootstrap = False
        do_stacked_map = True
        do_vshuffle = False
    else:
        print("no test")
        nObj = None
        do_bootstrap = True
        do_stacked_map = False
        do_vshuffle = False
    catname = args.catalogue

    galcat = Catalog(u,
                     massConv,
                     # catType='radec',                     
                     # catType='stellar_mass',
                     catType='no_mass',
                     name=catname,
                     nObj=nObj,
                     pathInCatalog=catpath+catname,
                     workDir=catpath,
                     rV=0.65)
    
    # galcat.readRADecCatalog()
    # galcat.readStellarMassCatalog()
    galcat.readNoMassCatalog()
    
    ts = ThumbStack(
    u,
    galcat,
    cmbMap,
    # cmbMask,
    cmbMask_2,
    cmbHit=None,
    name=tsname,
    save=False,
    nProc=nproc,
    filterTypes='diskring',
    estimatorTypes=['ksz_uniformweight'],
    doVShuffle=do_vshuffle,
    doBootstrap=do_bootstrap,
    workDir=outpath,
    runEndToEnd=True,
    test=test,
    doStackedMap=do_stacked_map,
    block_bootstrap=True,nSamples=50000
)


run_analysis()
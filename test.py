from cosmoprimo.fiducial import DESI
import sys
sys.path.append('/home/jiaqu/Thumbstack_DESI/')
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
import numpy as np
from pixell import enmap, curvedsky as cs
print("import done")

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
parser.add_argument("--output-dir", type=str,  default=None,help='Output directory.')
parser.add_argument("--savename",type=str,default=None)
parser.add_argument("--catalogue",type=str,default="full_catalog_Y1_berni_mask.txt")
parser.add_argument("--cmb",type=str,default="/project/rrg-rbond-ac/msyriac/ilc_dr6v3/20230606/hilc_fullRes_TT_17000.fits")
parser.add_argument("--mask",type=str,default="/project/rrg-rbond-ac/msyriac/ilc_dr6v3/20230606/wide_mask_GAL070_apod_1.50_deg_wExtended.fits")
parser.add_argument("--apply-cmb-mask2", action='store_false', 
                    help='Apply additional CMB mask 2 for >5sigma outlier removal')
parser.add_argument("--save-filtered-catalog", action='store_true',
                    help='Save the filtered catalog as a CSV file after applying all masks')

args = parser.parse_args()

savename = args.savename
cmbMap = enmap.read_fits(args.cmb)
cmbMask = enmap.read_fits(args.mask)

mask_5sigma_1 = 1.*(cmbMap<5*np.std(cmbMap))
mask_5sigma_2 = 1.*(cmbMap>-5*np.std(cmbMap))
cmbMask_2 = cmbMask*mask_5sigma_1*mask_5sigma_2
print("read cmb done")

def run_analysis(nproc = 1, test=False,
                 catpath = "/scratch/jiaqu/desi/catalogue/"):
    outpath = '/scratch/jiaqu/desi/'
    tsname=savename
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
        do_bootstrap = False
        do_stacked_map = False
        do_vshuffle = args.random
    catname = args.catalogue

    galcat = Catalog(u,
                     massConv,
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
    cmbMask,
    cmbHit=None,
    cmbMap2=None,
    cmbMask2=cmbMask_2,  # Pass the computed cmbMask_2
    name=tsname,
    save=True,
    nProc=nproc,
    filterTypes='diskring',
    estimatorTypes=['ksz_uniformweight'],
    doVShuffle=do_vshuffle,
    doBootstrap=do_bootstrap,
    workDir=outpath,
    runEndToEnd=True,
    test=test,
    doStackedMap=do_stacked_map,
    applyCmbMask2=args.apply_cmb_mask2 
)

    # Save the filtered catalog if requested
    if args.save_filtered_catalog:
        filtered_catalog_path = ts.saveFilteredCatalog(
            filterType='diskring',  # Use the same filter type as the analysis
            overlap=False,  # Set to True if you want to require overlap with CMB map
            psMask=True,    # Remove point sources
            outlierReject=True,  # Remove outliers
            applyCmbMask2=args.apply_cmb_mask2, # Use same CMB mask 2 setting as main analysis
            outputPath="/scratch/jiaqu/desi/output/" + savename + "/filtered_catalog.csv"  # Specify output path
        )
        print(f"Filtered catalog saved to: {filtered_catalog_path}")


run_analysis()
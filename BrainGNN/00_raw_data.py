from nilearn import datasets
import argparse
from imports import preprocess_data as Reader
import os
import shutil
import sys
breakpoint()

# Input data variables
#code_folder = os.getcwd()
root_folder = '/media/SSD2/MindGraphs/data_func/'
data_folder = os.path.join(root_folder, 'ABIDE_pcp/cpac/filt_noglobal/')
#data_folder = os.path.join("media/SSD3", 'ABIDE_pcp/cpac/filt_noglobal/')
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
shutil.copyfile(os.path.join(root_folder,'subject_ID.txt'), os.path.join(data_folder, 'subject_IDs.txt'))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description='Download ABIDE data and compute functional connectivity matrices')
    parser.add_argument('--pipeline', default='cpac', type=str,
                        help='Pipeline to preprocess ABIDE data. Available options are ccs, cpac, dparsf and niak.'
                             ' default: cpac.')
    parser.add_argument('--atlas', default='cc200',
                        help='Brain parcellation atlas. Options: ho, cc200 and cc400, default: cc200.')
    parser.add_argument('--download', default=True, type=str2bool,
                        help='Dowload data or just compute functional connectivity. default: True')
    args = parser.parse_args()
    print(args)

    params = dict()

    pipeline = args.pipeline
    atlas = args.atlas
    download = args.download

    # Files to fetch

    files = ['rois_' + atlas]

    filemapping = {'func_preproc': 'func_preproc.nii.gz',
                   files[0]: files[0] + '.1D'}


    # Download database files
    if download == True:
        abide = datasets.fetch_abide_pcp(data_dir=root_folder, pipeline=pipeline,
                                         band_pass_filtering=True, global_signal_regression=False, derivatives=["func_preproc"],
                                         quality_checked=False, )
        
if __name__ == '__main__':
    main()
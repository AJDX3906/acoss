# -*- coding: utf-8 -*-
"""
Batch audio feature extractor for acoss
"""
import argparse
import time
import glob
import os
import deepdish as dd
from joblib import Parallel, delayed
from progress.bar import Bar
from shutil import rmtree

from .utils import log, read_txt_file, savelist_to_file, create_audio_path_batches
from .features import AudioFeatures

__all__ = ['PROFILE', 
            'compute_features', 
            'compute_features_from_list_file', 
            'batch_feature_extractor']

PROFILE = {
           'sample_rate': 44100,
           'input_audio_format': '.mp3',
           'downsample_audio': False,
           'downsample_factor': 2,
           'endtime': None,
           'features': ['hpcp',
                        'key_extractor',
                        'madmom_features',
                        'mfcc_htk']
        }


_LOG_FILE_PATH = "acoss.extractor.log"
_LOG_FILE = log(_LOG_FILE_PATH)
_ERRORS = list()


def compute_features(audio_path, params=PROFILE):
    """
    Compute a list of audio features for a given audio file as per the extractor profile.

    NOTE: Audio files should be structured in a way that each cover song clique has a folder with it's tracks inside to
          have the correct cover label in the resulted feature dictionary.

          eg: ./audio_dir/
                    /cover_clique_label/ (folder name)
                        /audio_file.mp3 (or any other format)

    :param audio_path: path to audio file
    :param params: dictionary of parameters for the extractor (refer 'extractor.PROFILE' for default params)

    :return: a python dictionary with all the requested features computed as key, value pairs.
    """
    feature = AudioFeatures(audio_file=audio_path, sample_rate=params['sample_rate'])
    if feature.audio_vector.shape[0] == 0:
        raise IOError("Empty or invalid audio recording file -%s-" % audio_path)

    if params['endtime']:
        feature.audio_vector = feature.audio_slicer(endTime=params['endtime'])
    if params['downsample_audio']:
        feature.audio_vector = feature.resample_audio(params['sample_rate'] / params['downsample_factor'])

    out_dict = dict()
    # now we compute all the listed features in the profile dict and store the results to a output dictionary
    for method in params['features']:
        out_dict[method] = getattr(feature, method)()

    track_id = os.path.basename(audio_path).replace(params['input_audio_format'], '')
    out_dict['track_id'] = track_id

    label = audio_path.split('/')[-2]
    out_dict['label'] = label

    return out_dict


def compute_features_from_list_file(input_txt_file, feature_dir, params=PROFILE):
    """
    Compute specified audio features for a list of audio file paths and store to disk as .h5 file
    from a given input text file.
    It is a wrapper around 'compute_features'.

    :param input_txt_file: a text file with a list of audio file paths
    :param feature_dir: a path
    :param params: dictionary of parameters for the extractor (refer 'extractor.PROFILE' for default params)

    :return: None
    """

    start_time = time.monotonic()
    _LOG_FILE.info("Extracting features for %s " % input_txt_file)
    data = read_txt_file(input_txt_file)
    data = [path for path in data if os.path.exists(path)]
    if len(data) < 1:
        _LOG_FILE.debug("Empty collection txt file -%s- !" % input_txt_file)
        raise IOError("Empty collection txt file -%s- !" % input_txt_file)
    
    progress_bar = Bar('acoss.extractor.compute_features_from_list_file', 
                        max=len(data), 
                        suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
    for song in data:
        try:
            feature_dict = compute_features(audio_path=song, params=params)
            work_id = song.split('/')[-2]
            work_dir = "%s%s/" % (feature_dir, work_id)
            if not os.path.exists(work_dir):
                os.makedirs(work_dir)
            # save as h5
            dd.io.save(work_dir + os.path.basename(song).replace(params['input_audio_format'], '') + '.h5',
                       feature_dict)
        except:
            _ERRORS.append(input_txt_file)
            _ERRORS.append(song)
            _LOG_FILE.debug("Error: skipping computing features for audio file --%s-- " % song)
        progress_bar.next()
    progress_bar.finish()
    _LOG_FILE.info("Process finished in - %s - seconds" % (start_time - time.time()))


def batch_feature_extractor(audio_dir, feature_dir, n_workers, extractor_profile):
    import tempfile
    import glob
    from joblib import Parallel, delayed
    from acoss.extractors import compute_features_from_list_file

    temp = tempfile.NamedTemporaryFile(prefix="tmp", dir="./")

    for path, subdirs, files in os.walk(audio_dir):
        with open(temp.name, 'a') as f:
            for name in files:
                if name.endswith(".mp3"):
                    f.write(f'{os.path.join(path, name)}\n')

    collection_files = glob.glob(temp.name)
    feature_path = [feature_dir for i in range(len(collection_files))]
    param_list = [extractor_profile for i in range(len(collection_files))]
    args = zip(collection_files, feature_path, param_list)

    print("Computing batch feature extraction using '%s' mode the profile: %s \n" % ("parallel", extractor_profile))

    try:
        Parallel(n_jobs=n_workers, verbose=1)(
            delayed(compute_features_from_list_file)(cpath, fpath, param) for cpath, fpath, param in args)
    except:
        print("Skipping extraction for a single file with unkown ID")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="With command-line args, it does batch feature extraction of  \
            collection of audio files using multiple threads", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-d", "--dataset_csv", action="store",
                        help="path to input dataset csv file")
    parser.add_argument("-a", "--audio_dir", action="store",
                        help="path to the main audio directory of dataset")
    parser.add_argument("-p", "--feature_dir", action="store",
                        help="path to directory where the audio features should be stored")
    parser.add_argument("-f", "--feature_list", action="store", type=str, default="['hpcp', 'crema', "
                                                                                  "'chroma_cqt', 'chroma_cqt_processed', "
                                                                                  "'mfcc_htk']",
                        help="List of features to compute. Eg. ['hpcp' 'crema']")
    parser.add_argument("-m", "--run_mode", action="store", default='parallel',
                        help="Whether to run the extractor in single or parallel mode. "
                             "Choose one of ['single', 'parallel']")
    parser.add_argument("-n", "--workers", action="store", default=-1,
                        help="No of workers for running the batch extraction process. Only valid in 'parallel' mode.")

    cmd_args = parser.parse_args()

    print("Args: %s" % cmd_args)

    if not os.path.exists(cmd_args.p):
        os.mkdir(cmd_args.p)

    feature_list = list(cmd_args.f)
    updated_profile = PROFILE.copy()
    del updated_profile['features']
    updated_profile['features'] = feature_list

    batch_feature_extractor(audio_dir=cmd_args.a, feature_dir=cmd_args.p, n_workers=cmd_args.n, extractor_profile=updated_profile)

    print("... Done ....")
    print(" -- PROFILE INFO -- \n %s" % PROFILE)

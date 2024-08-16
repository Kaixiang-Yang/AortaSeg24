import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)
import shutil
from multiprocessing import Pool

import numpy as np
import torch, nnunet, importlib, pkgutil
from copy import deepcopy
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.inference.predict import preprocess_multithreaded
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from nnunet.postprocessing.connected_components import load_postprocessing, load_remove_save
from torch import cuda
from torch.nn import functional as F

def recursive_find_python_class(folder, trainer_name, current_module):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules(folder):
        # print(modname, ispkg)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, trainer_name):
                tr = getattr(m, trainer_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules(folder):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class([join(folder[0], modname)], trainer_name, current_module=next_current_module)
            if tr is not None:
                break

    return tr

def restore_model(pkl_file, checkpoint=None, train=False, fp16=None):
    """
    This is a utility function to load any nnUNet trainer from a pkl. It will recursively search
    nnunet.trainig.network_training for the file that contains the trainer and instantiate it with the arguments saved in the pkl file. If checkpoint
    is specified, it will furthermore load the checkpoint file in train/test mode (as specified by train).
    The pkl file required here is the one that will be saved automatically when calling nnUNetTrainer.save_checkpoint.
    :param pkl_file:
    :param checkpoint:
    :param train:
    :param fp16: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """
    info = load_pickle(pkl_file)
    init = info['init']
    name = info['name']
    search_in = join(nnunet.__path__[0], "training", "network_training")
    tr = recursive_find_python_class([search_in], name, current_module="nnunet.training.network_training")

    if tr is None:
        """
        Fabian only. This will trigger searching for trainer classes in other repositories as well
        """
        try:
            import meddec
            search_in = join(meddec.__path__[0], "model_training")
            tr = recursive_find_python_class([search_in], name, current_module="meddec.model_training")
        except ImportError:
            pass

    if tr is None:
        raise RuntimeError("Could not find the model trainer specified in checkpoint in nnunet.trainig.network_training. If it "
                           "is not located there, please move it or change the code of restore_model. Your model "
                           "trainer can be located in any directory within nnunet.trainig.network_training (search is recursive)."
                           "\nDebug info: \ncheckpoint file: %s\nName of trainer: %s " % (checkpoint, name))

    # this is now deprecated
    """if len(init) == 7:
        print("warning: this model seems to have been saved with a previous version of nnUNet. Attempting to load it "
              "anyways. Expect the unexpected.")
        print("manually editing init args...")
        init = [init[i] for i in range(len(init)) if i != 2]"""

    # ToDo Fabian make saves use kwargs, please...

    trainer = tr(*init)

    # We can hack fp16 overwriting into the trainer without changing the init arguments because nothing happens with
    # fp16 in the init, it just saves it to a member variable
    if fp16 is not None:
        trainer.fp16 = fp16

    trainer.process_plans(info['plans'])
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint, train)
    return trainer


def load_model_and_checkpoint_files(folder, folds=None, mixed_precision=None, checkpoint_name="model_best",
                                    modality=14):
    """
    used for if you need to ensemble the five models of a cross-validation. This will restore the model from the
    checkpoint in fold 0, load all parameters of the five folds in ram and return both. This will allow for fast
    switching between parameters (as opposed to loading them from disk each time).

    This is best used for inference and test prediction
    :param folder:
    :param folds:
    :param mixed_precision: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """
    if isinstance(folds, str):
        folds = [join(folder, "all")]
        assert isdir(folds[0]), "no output folder for fold %s found" % folds
    elif isinstance(folds, (list, tuple)):
        if len(folds) == 1 and folds[0] == "all":
            folds = [join(folder, "all")]
        else:
            folds = [join(folder, "fold_%d" % i) for i in folds]
        assert all([isdir(i) for i in folds]), "list of folds specified but not all output folders are present"
    elif isinstance(folds, int):
        folds = [join(folder, "fold_%d" % folds)]
        assert all([isdir(i) for i in folds]), "output folder missing for fold %d" % folds
    elif folds is None:
        print("folds is None so we will automatically look for output folders (not using \'all\'!)")
        folds = subfolders(folder, prefix="fold")
        print("found the following folds: ", folds)
    else:
        raise ValueError("Unknown value for folds. Type: %s. Expected: list of int, int, str or None", str(type(folds)))

    trainer = restore_model(join(folds[0], "%s.model.pkl" % checkpoint_name), fp16=mixed_precision)
    trainer.output_folder = folder
    trainer.output_folder_base = folder
    trainer.update_fold(0)
    try:
        trainer.initialize(False,modality=modality)
    except:
        trainer.initialize(False)
    all_best_model_files = [join(i, "%s.model" % checkpoint_name) for i in folds]
    print("using the following model files: ", all_best_model_files)
    all_params = [torch.load(i, map_location=torch.device('cpu')) for i in all_best_model_files]
    return trainer, all_params


def predict_cases_segrap2023(model, list_of_lists, output_filenames, folds, save_npz=False, num_threads_preprocessing=1,
                             num_threads_nifti_save=1, segs_from_prev_stage=None, do_tta=False,
                             overwrite_existing=False, step_size=0.5, checkpoint_name="model_final_checkpoint",
                             disable_postprocessing: bool = True):
    assert len(list_of_lists) == len(output_filenames)
    if segs_from_prev_stage is not None:
        assert len(segs_from_prev_stage) == len(output_filenames)
    
    all_in_gpu = torch.cuda.is_available()
    mixed_precision = True

    pool = Pool(num_threads_nifti_save)
    results = []

    cleaned_output_files = []
    for o in output_filenames:
        dr, f = os.path.split(o)
        if len(dr) > 0:
            maybe_mkdir_p(dr)
        cleaned_output_files.append(join(dr, f))

    if not overwrite_existing:
        print("number of cases:", len(list_of_lists))
        # if save_npz=True then we should also check for missing npz files
        not_done_idx = [i for i, j in enumerate(cleaned_output_files) if
                        (not isfile(j)) or (save_npz and not isfile(j[:-7] + '.npz'))]

        cleaned_output_files = [cleaned_output_files[i] for i in not_done_idx]
        print(list_of_lists)
        list_of_lists = [list_of_lists[i] for i in not_done_idx]
        if segs_from_prev_stage is not None:
            segs_from_prev_stage = [segs_from_prev_stage[i]
                                    for i in not_done_idx]

        print("number of cases that still need to be predicted:",
              len(cleaned_output_files))

    print("emptying cuda cache")
    torch.cuda.empty_cache()

    print("loading parameters for folds,", folds)
    trainer, params = load_model_and_checkpoint_files(model, folds, mixed_precision=True,
                                                      checkpoint_name=checkpoint_name)
    trainer.load_checkpoint_ram(params[0], False)
    del params

    print("starting preprocessing generator")
    preprocessing = preprocess_multithreaded(trainer, list_of_lists, cleaned_output_files, num_threads_preprocessing,
                                             segs_from_prev_stage)
    print("starting prediction...")
    all_output_files = []
    with torch.no_grad():
        for preprocessed in preprocessing:
            output_filename, (d, dct) = preprocessed
            all_output_files.append(all_output_files)
            if isinstance(d, str):
                data = np.load(d)
                os.remove(d)
                d = data

            print("predicting", output_filename)
            ##### for Aorta2024 #####
            ##### 2024.7.25 #####
            # d = d.transpose(0, 3, 2, 1)
            # print(d.shape)
            
            softmax = trainer.predict_preprocessed_data_return_seg_and_softmax(
                d, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
                step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
                mixed_precision=mixed_precision)[1]

            transpose_forward = trainer.plans.get('transpose_forward')
            if transpose_forward is not None:
                transpose_backward = trainer.plans.get('transpose_backward')
                softmax = softmax.transpose([0] + [i + 1 for i in transpose_backward])

            if save_npz:
                npz_file = output_filename[:-7] + ".npz"
            else:
                npz_file = None

            if hasattr(trainer, 'regions_class_order'):
                region_class_order = trainer.regions_class_order
            else:
                region_class_order = None
            
            ##### for Aorta2024 #####
            ##### 2024.7.25 #####
            # print(softmax.shape)
            # softmax = softmax.transpose(0, 3, 2, 1)
            # print(softmax.shape)


            bytes_per_voxel = 4
            if all_in_gpu:
                bytes_per_voxel = 2  # if all_in_gpu then the return value is half (float16)
            if np.prod(softmax.shape) > (2e9 / bytes_per_voxel * 0.85):  # * 0.85 just to be save
                print(
                    "This output is too large for python process-process communication. Saving output temporarily to disk")
                np.save(output_filename[:-4] + ".npy", softmax)
                softmax = output_filename[:-4] + ".npy"

            results.append(pool.starmap_async(save_segmentation_nifti_from_softmax,
                                            ((softmax, output_filename, dct, 1, region_class_order,
                                                None, None,
                                                npz_file, None, False, 1),)
                                            ))
            # save_segmentation_nifti_from_softmax(softmax, output_filename, dct, 1, region_class_order,
            #                                     None, None,
            #                                     npz_file, None, False, 1)

    print("inference done. Now waiting for the segmentation export to finish...")
    _ = [i.get() for i in results]

    pool.close()
    pool.join()
    import gc
    gc.collect()


def check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities):
    print("This model expects %d input modalities for each image" %
          expected_num_modalities)
    files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)

    maybe_case_ids = np.unique([i[:-12] for i in files])

    remaining = deepcopy(files)
    missing = []

    assert len(
        files) > 0, "input folder did not contain any images (expected to find .nii.gz file endings)"

    # now check if all required files are present and that no unexpected files are remaining
    for c in maybe_case_ids:
        for n in range(expected_num_modalities):
            expected_output_file = c + "_%04.0d.nii.gz" % n
            if not isfile(join(input_folder, expected_output_file)):
                missing.append(expected_output_file)
            else:
                remaining.remove(expected_output_file)

    print("Found %d unique case ids, here are some examples:" % len(maybe_case_ids),
          np.random.choice(maybe_case_ids, min(len(maybe_case_ids), 10)))
    print("If they don't look right, make sure to double check your filenames. They must end with _0000.nii.gz etc")

    if len(remaining) > 0:
        print("found %d unexpected remaining files in the folder. Here are some examples:" % len(remaining),
              np.random.choice(remaining, min(len(remaining), 10)))

    if len(missing) > 0:
        print("Some files are missing:")
        print(missing)
        raise RuntimeError("missing files in input_folder")

    return maybe_case_ids


def predict_from_folder_segrap2023(model: str, input_folder: str, output_folder: str, folds: 0, part_id:0, num_parts:1):
    """
        here we use the standard naming scheme to generate list_of_lists and output_files needed by predict_cases

    :param model:
    :param input_folder:
    :param output_folder:
    :param folds:
    :param save_npz:
    :param num_threads_preprocessing:
    :param num_threads_nifti_save:
    :param lowres_segmentations:
    :param part_id:
    :param num_parts:
    :param tta:
    :param mixed_precision:
    :param overwrite_existing: if not None then it will be overwritten with whatever is in there. None is default (no overwrite)
    :return:
    """
    maybe_mkdir_p(output_folder)
    # shutil.copy(join(model, 'plans.pkl'), output_folder)

    # assert isfile(join(model, "plans.pkl")
    #               ), "Folder with saved model weights must contain a plans.pkl file"
    expected_num_modalities = 1

    # check input folder integrity
    case_ids = [sorted([i.replace('.mha','') for i in os.listdir(input_folder) if i.endswith('.mha') or i.endswith('.tiff')])[0]]

    output_files = [join(output_folder, 'output' + ".mha") for i in case_ids]
    list_of_lists = [[join(input_folder, j + ".mha")] for j in case_ids]
    
    return predict_cases_segrap2023(model, list_of_lists[part_id::num_parts], output_files[part_id::num_parts], folds=0)
   

# predict_from_folder_segrap2023("weight/", "images/", "test/", 0, 0, 1)

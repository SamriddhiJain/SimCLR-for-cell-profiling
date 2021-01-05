import yaml
import pandas as pd
import numpy as np
import os
import time

from feature_eval.profile_metrics import calculate_nsc_and_nscb
from generate_features import get_embeddings, get_data

# create moa meta data file
def get_meta(sc_meta_path):
    meta_sc = pd.read_csv(sc_meta_path)
    meta_sc['Metadata_Plate'] = [s.split("-")[0] for s in meta_sc['Image_Name']]
    meta_sc['Metadata_Well'] = [s.split("-")[1] for s in meta_sc['Image_Name']]
    meta_sc['Metadata_Site'] = [s.split("-")[2][:2] for s in meta_sc['Image_Name']]
    meta_sc['compound'] = [s.split("_")[0] for s in meta_sc['Class_Name']]
    meta_sc['concentration'] = [float(s.split("_")[1]) for s in meta_sc['Class_Name']]
    meta_sc['Replicate'] = 1

    moa = pd.read_csv("https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_moa.csv")
    moa['concentration'] = moa['concentration'].astype(str)
    moa['Class_Name'] = moa[['compound', 'concentration']].agg('_'.join, axis=1)
    moa_dict = dict(zip(moa['Class_Name'], moa['moa']))
    meta_sc['moa'] = [moa_dict[c] for c in meta_sc['Class_Name']]

    return meta_sc


def all_checkpoints_exist(checkpoint_list):
    for checkpoint in checkpoint_list:
        if not os.path.isfile(checkpoint):
            return False
    return True


def checkpoint_exists(checkpoint):
    if not os.path.isfile(checkpoint):
        return False
    return True


def nsc_nscb(epoch_list_dic):
    # Pre check for checkpoints
    for model_dir, epoch_list in epoch_list_dic.items():
        checkpoints_dir = f"runs/{model_dir}/checkpoints/"
        status = all_checkpoints_exist([f"{checkpoints_dir}/model_epoch_{e}.pth" for e in epoch_list])
        print(f"Checking directory {checkpoints_dir} for epochs {epoch_list} to all exist: {status}")

    config = yaml.load(open("config.yaml", "r"))
    loader = get_data(config)
    meta = get_meta(config['eval_dataset']['path'])

    for model_dir, epoch_list in epoch_list_dic.items():
        checkpoints_dir = f"runs/{model_dir}/checkpoints/"
        print("Evaluating", checkpoints_dir)
        for epoch in epoch_list:
            model_path = f"{checkpoints_dir}/model_epoch_{epoch}.pth"

            print(f"Waiting for the epoch {epoch} checkpoint to become available in {checkpoints_dir} ...")
            while not checkpoint_exists(model_path):
                time.sleep(60)

            features = get_embeddings(config, model_path, loader)
            np.save(f"runs/{model_dir}/features_epoch_{epoch}.npy", features)

            # without TVN transformation
            plot_file_structure = f"{checkpoints_dir}/" + "{}_epoch_" + f"{epoch}_untransformed.jpg"
            nsc, nscb = calculate_nsc_and_nscb(features=features,
                                               meta=meta,
                                               plot_file_structure=plot_file_structure,
                                               DO_WHITENING=False,
                                               DO_CORAL=False)

            print(f"Results for {model_dir} epoch {epoch};")
            print(f"NSC:{nsc} NSCB:{nscb}")

            # with whitening transformation
            plot_file_structure = f"{checkpoints_dir}/" + "{}_epoch_" + f"{epoch}_whitened.jpg"
            nsc, nscb = calculate_nsc_and_nscb(features=features,
                                               meta=meta,
                                               plot_file_structure=plot_file_structure,
                                               DO_WHITENING=True,
                                               DO_CORAL=False)
            print(f"whitening-NSC:{nsc} whitening-NSCB:{nscb}")

            # with TVN transformation
            plot_file_structure = f"{checkpoints_dir}/" + "{}_epoch_" + f"{epoch}_TVN.jpg"
            nsc, nscb = calculate_nsc_and_nscb(features=features,
                                               meta=meta,
                                               plot_file_structure=plot_file_structure,
                                               DO_WHITENING=True,
                                               DO_CORAL=True)
            print(f"TVN-NSC:{nsc} TVN-NSCB:{nscb}")


if __name__ == "__main__":
    epoch_list_dic = {"Jan03_14-55-43_lo-g2-013": [50, 100, 150, 200, 250],
                      "Jan04_14-32-13_lo-g2-009": [50, 100, 150, 200, 250]}

    nsc_nscb(epoch_list_dic)

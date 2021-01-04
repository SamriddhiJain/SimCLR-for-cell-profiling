import yaml
import pandas as pd

from feature_eval.profile_metrics import calculate_nsc_and_nscb
from generate_features import get_embeddings, get_data


def get_meta(sc_meta_path):
    meta_sc = pd.read_csv(sc_meta_path)
    meta_sc['Metadata_Plate'] = [s.split("-")[0] for s in meta_sc['Image_Name']]
    meta_sc['Metadata_Well'] = [s.split("-")[1] for s in meta_sc['Image_Name']]
    meta_sc['Metadata_Site'] = [s.split("-")[2][:2] for s in meta_sc['Image_Name']]
    meta_sc['compound'] = [s.split("_")[0] for s in meta_sc['Class_Name']]
    meta_sc['concentration'] = [s.split("_")[1] for s in meta_sc['Class_Name']]
    meta_sc['Replicate'] = 1

    moa = pd.read_csv("https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_moa.csv")
    moa['concentration'] = moa['concentration'].astype(str)
    moa['Class_Name'] = moa[['compound', 'concentration']].agg('_'.join, axis=1)
    moa_dict = dict(zip(moa['Class_Name'], moa['moa']))
    meta_sc['moa'] = [moa_dict[c] for c in meta_sc['Class_Name']]

    return meta_sc


def nsc_nscb(epoch_list_dic):
    config = yaml.load(open("config.yaml", "r"))
    loader = get_data(config)
    meta = get_meta(config['eval_dataset']['path'])

    for model_dir, epoch_list in epoch_list_dic.items():
        checkpoints_dir = f"runs/{model_dir}/checkpoints/"
        print("Evaluating", checkpoints_dir)

        for epoch in epoch_list:
            model_path = f"{checkpoints_dir}/checkpoints/model_epoch_{epoch}.pth"
            plot_file_structure = f"{checkpoints_dir}/checkpoints/" + "{}_epoch_" + f"{epoch}.jpg"
            features = get_embeddings(config, model_path, loader)

            # without TVN transformation
            nsc, nscb = calculate_nsc_and_nscb(features=features,
                                               meta=meta,
                                               plot_file_structure=plot_file_structure,
                                               DO_WHITENING=False,
                                               DO_CORAL=False)

            print(f"Results for {model_dir} epoch {epoch};")
            print(f"NSC:{nsc} NSCB:{nscb}")

            # with TVN transformation
            nsc, nscb = calculate_nsc_and_nscb(features=features,
                                               meta=meta,
                                               plot_file_structure=plot_file_structure,
                                               DO_WHITENING=True,
                                               DO_CORAL=True)
            print(f"TVN-NSC:{nsc} TVN-NSCB:{nscb}")


if __name__ == "__main__":
    epoch_list_dic = {"Jan03_14-55-43_lo-g2-013": [50, 100, 150, 200, 250]}

    nsc_nscb(epoch_list_dic)

from simclr import SimCLR
import yaml
from data_aug.dataset_wrapper_simclr import DataSetWrapperSimCLR
from data_aug.dataset_wrapper import DataSetWrapper

def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = DataSetWrapperSimCLR(config['batch_size'], **config['dataset'])

    debug_eval_dataset = None
    if "eval_classifier_n_epoch" in config.keys():
        debug_eval_dataset = DataSetWrapper(config['batch_size'], **config['eval_dataset'])

    simclr = SimCLR(dataset, config, debug_eval_dataset)
    simclr.train()


if __name__ == "__main__":
    main()

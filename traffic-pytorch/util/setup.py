import argparse 
from util.logging import *
import importlib

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', '--mode', default='traffic', type=str,
        help='traffic or flow')
    parser.add_argument('-model', '--model', type=str,
        help='Model name')
    parser.add_argument('-config', '--config', default=None, type=str,
        help='Config name')
    parser.add_argument('-dataset_dir', '--ddir', type=str,
        help='Path to dataset')
    parser.add_argument('-dataset_name', '--dname', type=str,
        help='Name of the dataset')
    parser.add_argument('-train_ratio', '--train_ratio', default=0.7, type=float,
        help='Train set ratio (default: 0.7)')
    parser.add_argument('-test_ratio', '--test_ratio', default=0.2, type=float,
        help='Test set ratio (default: 0.2)')
    parser.add_argument('-num_pred', '--num_pred', default=3, type=int,
        help='Time to predict')
    parser.add_argument('-device', '--device', default='cuda:0', type=str,
        help='GPU to enable (default: cuda:0)')
    args = parser.parse_args()
    if args.config == None:
        args.config = '{}_config'.format(args.model)
    if args.train_ratio + args.test_ratio > 1.0:
        print(toRed('Sum of train ratio and test ratio exceeds 1.0'))
        raise ValueError
    print(toGreen('Arguments loaded succesfully'))
    return args

def load_trainer(args):
    try: 
        confg_class = getattr(importlib.import_module("config.{}".format(args.config)), args.config)
        config = confg_class(args.device, args.ddir, args.dname, args.train_ratio, args.test_ratio)
        config.num_pred = args.num_pred
    except:
        print(toRed('Config undefined'))
        raise 
    print(toGreen('Config loaded succesfully'))
    try: 
        model_class = getattr(importlib.import_module("model.{}".format(args.model)), "{}Model".format(args.model))
        trainer_class = getattr(importlib.import_module("trainer.{}_trainer".format(args.model)), "{}Trainer".format(args.model))
        trainer = trainer_class(model_class, config, args)
        if args.mode not in ['traffic', 'demand']: 
            raise AssertionError("Please specify mode: ['traffic', 'demand']")
    except:
        print(toRed('Model undefined'))
        raise 
    print(toGreen('Trainer loaded succesfully'))
    return trainer

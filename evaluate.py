import os
import sys
import argparse
from tqdm import tqdm

from evaluation.dose_evaluation_class import *
from networks.AttUnet_model import Att_UNet
from networks.trainer import *
from dataloader.data_loader import DataLoader
from torch.utils import data


# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    torch.device('cpu')


parser = argparse.ArgumentParser()
parser.add_argument('--test_data_path', type=str,
                    default='./Data/test-pats/', help='test data path folder')

parser.add_argument('--plot_DVH', type=list,
                    default= [], help='the data we want to plot dose figures')

args = parser.parse_args()

if __name__ == "__main__":
    print(args.test_data_path)
    if not os.path.exists(args.test_data_path):
        raise SystemExit('test folder not exist')
    

    data_loader_eval = DataLoader(data_folder = args.test_data_path, mode_name='evaluate')

    dose_evaluator = EvaluateDose(data_loader_eval, data_loader_eval)

    print('\n\n# Start evaluation !')
    dvh_score, dose_score = dose_evaluator.make_metrics()
    print('For this out-of-sample test:\n'
            '\tthe DVH score is {:.3f}\n '
            '\tthe dose score is {:.3f}'.format(dvh_score, dose_score))


    # Evaluation
#    print('\n\n# Start evaluation !')
#    Dose_score, DVH_score = get_Dose_score_and_DVH_score(prediction_dir=trainer.setting.output_dir + '/Prediction', gt_dir='../../Data/OpenKBP_C3D')

#    print('\n\nDose score is: ' + str(Dose_score))
#    print('DVH score is: ' + str(DVH_score))
    
import os
import sys
import argparse
from tqdm import tqdm

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

parser.add_argument('--model_weight_path', type=str,
                    default='./trained_models/best_Unet_model.pt', help='model weight path')

args = parser.parse_args()

if __name__ == "__main__":
    print(args.test_data_path)
    if not os.path.exists(args.test_data_path):
        raise SystemExit('test folder not exist')
    

    test_dataset = DataLoader(data_folder = args.test_data_path)

    model = Att_UNet(n_channels=11, n_classes=1).to(device)
    tester = Trainer(model=model,
                  device=device,
                  test_DataLoader=test_dataset,
                  model_weight_path = args.model_weight_path
                  )

    print('\n\n# Start inference !')
    tester.run_test()


    # Evaluation
#    print('\n\n# Start evaluation !')
#    Dose_score, DVH_score = get_Dose_score_and_DVH_score(prediction_dir=trainer.setting.output_dir + '/Prediction', gt_dir='../../Data/OpenKBP_C3D')

#    print('\n\nDose score is: ' + str(Dose_score))
#    print('DVH score is: ' + str(DVH_score))
    
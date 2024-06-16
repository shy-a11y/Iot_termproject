from flask import Flask, request
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
import logging
from vit import ViT
import yaml
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# get configs
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


config = get_config('C:/Users/swp10/Desktop/flaskProject/flaskProject/config.yaml')
activities = config['application']['client']["activity_labels"]

app = Flask(__name__)

csi_data = pd.DataFrame()  # Initialize as an empty DataFrame
P_COUNT = 0

num_sub = 64
window_size = 10
model = None

@app.route('/upload', methods=['POST'])
def upload():
    global csi_data
    global P_COUNT
    global model

    data = request.json['csi_data']
    new_csi_data = np.array(data).reshape(1, num_sub)

    P_COUNT += 1

    columns = ['_' + str(i) for i in range(num_sub)]
    new_csi_df = pd.DataFrame(new_csi_data, columns=columns)

    csi_data = pd.concat([csi_data, new_csi_df], ignore_index=True)
    '''
    try:
        print(csi_data.shape)
        print(len(csi_data))
        print(P_COUNT, window_size)

        if len(csi_data) == window_size and P_COUNT == window_size:
            print("1")
            csi_data_np = np.array(csi_data)
            csi_data_tensor = torch.from_numpy(csi_data_np).unsqueeze(0).unsqueeze(0).float()

            outputs = model(csi_data_tensor)
            outputs = F.log_softmax(outputs, dim=1)
            # y_hat = torch.from_numpy(
            #     np.array([np.argmax(outputs.cpu().data.numpy()[ii]) for ii in range(len(outputs))]))
            y_hat = torch.argmax(outputs, dim=1)

            # print('Predict result: {}'.format(activities[y_hat[0]]))
            # logger.info('Predict result: %s', activities[y_hat[0]])

            logger.info('Predict result: %s', activities[y_hat.item()])

            csi_data = pd.DataFrame()  # Clear the DataFrame
            P_COUNT = 0

        elif len(csi_data) > window_size and P_COUNT != 0:
            print("2")
            csi_data = pd.DataFrame()  # Clear the DataFrame
            P_COUNT = 0

            logger.error("CSI data exceeded window size!")

    except Exception as e:
        logger.error('Error processing CSI data: %s', e)

    '''
    try:
        if len(csi_data) == window_size and P_COUNT == window_size:
            csi_data_np = np.array(csi_data)

            c_data = torch.from_numpy(csi_data_np).unsqueeze(0).unsqueeze(0).float()

            pred = model(c_data)
            print('Predict result: {}'.format(pred))

            # Drop first row
            csi_data.drop(0, inplace=True)
            csi_data.reset_index(drop=True, inplace=True)

            P_COUNT = 0

        elif len(csi_data) == window_size and P_COUNT == window_size // 2:
            c_data = np.array(csi_data)
            c_data = torch.from_numpy(c_data).unsqueeze(0).unsqueeze(0).float()

            outputs = model(c_data)
            outputs = F.log_softmax(outputs, dim=1)
            y_hat = torch.from_numpy(
                np.array([np.argmax(outputs.cpu().data.numpy()[ii]) for ii in range(len(outputs))]))

            print('Predict result: {}'.format(activities[y_hat[0]]))

            # Drop first row
            csi_data.drop(0, inplace=True)
            csi_data.reset_index(drop=True, inplace=True)

            P_COUNT = 0

        elif len(csi_data) == window_size:
            # Drop first row
            csi_data.drop(0, inplace=True)
            csi_data.reset_index(drop=True, inplace=True)

        elif len(csi_data) > window_size:
            print("Error!")

    except Exception as e:
        logger.error('Error processing CSI data: %s', e)

    return "OK"


if __name__ == '__main__':
    # Load model
    model = ViT(
        in_channels=config['application']['model']['ViT']["in_channels"],
        patch_size=(config['application']['model']['ViT']["patch_size"],
                    config['subcarrier'][config['application']['client']["bandwidth"]]),
        embed_dim=config['application']['model']['ViT']["embed_dim"],
        num_layers=config['application']['model']['ViT']["num_layers"],
        num_heads=config['application']['model']['ViT']["num_heads"],
        mlp_dim=config['application']['model']['ViT']["mlp_dim"],
        num_classes=len(config['application']['client']["activity_labels"]),
        in_size=[config['application']['client']["window_size"],
                 config['subcarrier'][config['application']['client']["bandwidth"]]]
    )

    # Load pretrained model

    logger.info('======> Load model')
    device = torch.device('cpu')  # CPU로 로드
    model.load_state_dict(torch.load("C:/Users/swp10/Desktop/flaskProject/flaskProject/svl_best_model.pt", map_location=torch.device('cpu')))
    model.eval()  # 모델을 evaluation 모드로 설정
    logger.info('======> Success')

    app.run(host='0.0.0.0', port=8080)

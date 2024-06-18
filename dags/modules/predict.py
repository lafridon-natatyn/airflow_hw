# <YOUR_IMPORTS>
import json
import dill
import os
import joblib
from datetime import datetime

import pandas as pd

from pydantic import BaseModel
path = os.environ.get('PROJECT_PATH', '..')
with open(f'{path}/data/models/cars_pipe.pkl', 'rb') as input_file:
    model = dill.load(input_file)

df = pd.read_csv(f'{path}/data/train/homework.csv')



def predict():
    pred = []
    names_json = ['7310993818.json','7313922964.json', '7315173150.json', '7316152972.json','7316509996.json']
    for i in range (0, 5):
        with open(f'{path}/data/test/{names_json[i]}') as fin:
            form = json.load(fin)
            data = pd.DataFrame.from_dict([form])
            y = model.predict(data)
            pred.append((str(data['id']).split()[1], y[0]))
    pred_df = pd.DataFrame(pred,columns=['id','pred'])
    pred_df.to_csv(f'{path}/data/predictions/pred_df.csv', index=False)
    return pred_df
if __name__ == '__main__':
    predict()

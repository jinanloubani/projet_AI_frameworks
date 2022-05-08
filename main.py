import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import torch.nn as nn
import torchvision.transforms as transforms
from statistics import mean
import math
import argparse
import pandas as pd
from model import NCF


class Ratings_Datset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index()

    def __len__(self):
        return len(self.df)
  
    def __getitem__(self, idx):
        user = user2id[self.df['user_id'][idx]]
        user = torch.tensor(user, dtype=torch.long)
        item = item2id[self.df['recipe_id'][idx]]
        item = torch.tensor(item, dtype=torch.long)
        rating = torch.tensor(self.df['rating'][idx], dtype=torch.float)
        return user, item, rating

def test(model, testloader, m_eval=False):
    
    running_mae = 0
    with torch.no_grad():
        corrects = 0
        total = 0
        for users, items, r in testloader:
            users = users.cuda()
            items = items.cuda()
            y = r.cuda() / 5
            y_hat = model(users, items).flatten()
            error = torch.abs(y_hat - y).sum().data
            
            running_mae += error
            total += y.size(0)
    
    mae = running_mae/total
    return mae * 5


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--weights', type=str, default=None, metavar='N',
                    help='path to weights of pretrained model (default: None)')
    parser.add_argument('--test_data', type=str, default=None, metavar='N',
                    help='path to csv file of test data  (default: None)')

    args = parser.parse_args()

    weights = args.weights
    test_data = args.test_data

  # dataloaders
    test_final = pd.read_csv(test_data)
    user_list = test_final.user_id.unique()
    item_list = test_final.recipe_id.unique()
    user2id = {w: i for i, w in enumerate(user_list)}
    item2id = {w: i for i, w in enumerate(item_list)}
    testloader = DataLoader(Ratings_Datset(test_final), batch_size=64, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_user = 12925
    n_items = 160901
    model = NCF(n_user, n_items)
    model = model.to(device)
    model.load_state_dict(torch.load(weights))
    model.eval()

    users, recipes, r = next(iter(testloader))
    users = users.cuda()
    recipes = recipes.cuda()
    r = r.cuda()

    y = model(users, recipes)*5
    print("ratings", r[:10].data)
    print("predictions:", y.flatten()[:10].data)
    

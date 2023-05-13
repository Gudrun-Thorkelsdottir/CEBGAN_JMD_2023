import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from models.cgans import AirfoilAoAGenerator
from train_final_cbgan import read_configs
from utils.shape_plot import plot_samples, plot_grid, plot_comparision
from utils.dataloader import NoiseGenerator
from datetime import datetime

def load_generator(gen_cfg, save_dir, checkpoint, device='cpu'):
    ckp = torch.load(os.path.join(save_dir, checkpoint), map_location=device)
    generator = AirfoilAoAGenerator(**gen_cfg).to(device)
    generator.load_state_dict(ckp['generator'])
    generator.eval()
    return generator

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = '../saves/final/'
    _, gen_cfg, _, cz, _ = read_configs('cbgan')

    epoch = 5000

    inp_paras = np.load('../data/inp_paras_train.npy')
    mean, std = inp_paras.mean(0), inp_paras.std(0)
    inp_paras = np.load('../data/inp_paras_test.npy') # reload inp_paras from Jun's test set.
    tr_inp_paras = (inp_paras - mean) / std


    generator = load_generator(gen_cfg, save_dir, 'cbgan{}.tar'.format(epoch-1), device=device)
    params = torch.tensor(tr_inp_paras, dtype=torch.float, device=device)
    samples = []; aoas = []

    noise = torch.zeros([len(params), cz[0]], device=device, dtype=torch.float)
    pred = generator(noise, params)[0]
    samples.append(pred[0].cpu().detach().numpy().transpose([0, 2, 1]))
    aoas.append(pred[1].cpu().detach().numpy())
    
    np.save('../data/pred_cbgan/single/new/aoas_pred.npy', aoas[0])
    np.save('../data/pred_cbgan/single/new/airfoils_pred.npy', samples[0])
    np.save('../data/pred_cbgan/single/new/inp_params_pred.npy', params.cpu().detach().numpy())

    test_airfoils = np.load('../data/airfoils_opt_test.npy')
    test_aoas = np.load('../data/aoas_opt_test.npy')

    time = datetime.now().strftime('%b%d_%H-%M-%S')
    tb_dir = os.path.join(save_dir, 'runs')

    test_airfoils = test_airfoils.transpose(0,2,1)
    airfoils_pred = samples[0]
    #test_aoas = test_aoas.reshape(test_aoas.shape[0], 1)
    aoas_pred = aoas[0].reshape(aoas[0].shape[0])

    plot_comparision(None, test_airfoils, [airfoils_pred], test_aoas, aoas_pred, scale=1.0, scatter=False, symm_axis=None,fname=os.path.join(tb_dir, 'comparison_plot'))



    # num_trials = 10
    # for _ in range(num_trials):
    #     noise = torch.randn([len(params), cz[0]], device=device, dtype=torch.float)
    #     pred = generator(noise, params)[0]
    #     samples.append(pred[0].cpu().detach().numpy().transpose([0, 2, 1]))
    #     aoas.append(pred[1].cpu().detach().numpy())

    # np.save('aoas_pred.npy', np.stack(aoas, axis=1))
    # np.save('airfoils_pred.npy', np.stack(samples, axis=1))
    # np.save('inp_params_pred.npy', params.cpu().detach().numpy())

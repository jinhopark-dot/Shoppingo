import os
import time
from tqdm import tqdm
import torch
import math
import numpy as np
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch.nn import DataParallel

from .nets.attention_model import set_decode_type
from .utils import move_to

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model

def rollout(model, dataset, opts):
    set_decode_type(model, "greedy")
    model.eval()
    all_best_costs = []
    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size, num_workers=opts.num_workers)

    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        x = move_to(batch, opts.device)
        B = x.batch.max().item() + 1
        k = opts.num_policies

        num_continuous = 4
        num_categorical = 12
        assert opts.latent_dim == num_continuous + num_categorical
        
        z_continuous = torch.rand(B, k, num_continuous, device=opts.device) * 2 - 1
        indices = torch.randint(0, num_categorical, (B, k), device=opts.device)
        z_categorical = F.one_hot(indices, num_classes=num_categorical).float()
        z = torch.cat([z_continuous, z_categorical], dim=-1)

        x_expanded = Batch.from_data_list([d for d in x.to_data_list() for _ in range(k)])
        z_reshaped = z.view(B * k, opts.latent_dim)
        
        with torch.no_grad():
            costs, _, _ = model(x_expanded, z_reshaped, return_pi=True)
        
        costs = costs.view(B, k)
        min_costs, _ = costs.min(dim=1)
        all_best_costs.append(min_costs.cpu())
        
    return torch.cat(all_best_costs, 0)

def validate(model, dataset, opts):
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(avg_cost, torch.std(cost) / math.sqrt(len(cost))))
    return avg_cost

def clip_grad_norms(param_groups, max_norm=math.inf):
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(group['params'], max_norm if max_norm > 0 else math.inf, norm_type=2)
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped

def train_epoch(model, optimizer, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    training_dataset = problem.make_dataset(
        size=opts.graph_size,
        num_samples=opts.epoch_size
    )
    
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers)

    model.train()
    get_inner_model(model).set_decode_type("sampling", temp=1.0)
    
    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        train_batch(model, optimizer, epoch, batch_id, step, batch, tb_logger, opts)
        step += 1
    
    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save({
            'model': get_inner_model(model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all(),
        }, os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch)))

    avg_reward = validate(model, val_dataset, opts)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)
        
    lr_scheduler.step()
    return avg_reward

def train_batch(model, optimizer, epoch, batch_id, step, batch, tb_logger, opts):
    x = move_to(batch, opts.device)
    
    B = x.batch.max().item() + 1
    k = opts.num_policies
    
    num_continuous = 4
    num_categorical = 12
    assert opts.latent_dim == num_continuous + num_categorical, "Latent dim must match Z_3 distribution (16)"
    
    z_continuous = torch.rand(B, k, num_continuous, device=opts.device) * 2 - 1
    indices = torch.randint(0, num_categorical, (B, k), device=opts.device)
    z_categorical = F.one_hot(indices, num_classes=num_categorical).float()
    z = torch.cat([z_continuous, z_categorical], dim=-1)
    
    x_expanded = Batch.from_data_list([d for d in x.to_data_list() for _ in range(k)])
    z_reshaped = z.view(B * k, opts.latent_dim)
    
    costs, log_likelihoods, _ = model(x_expanded, z_reshaped, return_pi=True)
    
    costs = costs.view(B, k)
    log_likelihoods = log_likelihoods.view(B, k)
    
    best_indices = torch.argmin(costs, dim=1)
    best_costs = torch.gather(costs, 1, best_indices.unsqueeze(1)).squeeze(1)
    best_ll = torch.gather(log_likelihoods, 1, best_indices.unsqueeze(1)).squeeze(1)
    
    mean_costs = costs.mean(dim=1)
    std_costs = costs.std(dim=1).clamp(min=1e-8)
    advantage_weight = torch.abs(best_costs - mean_costs) / std_costs
    loss = (advantage_weight.detach() * -best_ll).mean()
    
    optimizer.zero_grad()
    loss.backward()
    grad_norms, grad_norms_clipped = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()
    
    if step % opts.log_step == 0:
        
        avg_cost = costs.mean().item()
        avg_best_cost = best_costs.mean().item()
        
        print(f'Step {step}, Epoch {epoch}, Batch {batch_id}: '
              f'avg_cost: {avg_cost:.4f}, '
              f'avg_best_cost: {avg_best_cost:.4f}, '
              f'loss: {loss.item():.4f}')
        if not opts.no_tensorboard:
            tb_logger.log_value('avg_cost', costs.mean().item(), step)
            tb_logger.log_value('avg_best_cost', best_costs.mean().item(), step)
            tb_logger.log_value('asil_loss', loss.item(), step)
            tb_logger.log_value('grad_norm', grad_norms[0].item(), step)
            tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)
# -*- coding: utf-8 -*-
import os
import sys  
import torch
import numpy as np
from collections import defaultdict
from models import TransferNetwork # MODIFIED import
import torch.nn as nn
from myloss import InfoMax_loss # MODIFIED import
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, auc, precision_recall_curve
from itertools import cycle
import config
from utility import classification_metric, edge_extract

def auprc(y_true, y_score):
    lr_precision, lr_recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_score)
    return auc(lr_recall, lr_precision)

def multi_eval_epoch(model, loader, node_x, edge_index, drug, device):
    model.eval()
    total_loss = 0
    # alpha = 0 # No longer needed
    y_true, y_pred, y_mask = [], [], []
    auc_list, aupr_list, acc_list, f1_list = [], [], [], []
    for x, y, mask, _ in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        mask = (mask > 0)
        with torch.no_grad():
            yp, _ = model(x, node_x, edge_index) # MODIFIED: returns class_output, feature
            loss_mat = nn.BCEWithLogitsLoss()(yp, y.double())
            loss_mat = torch.where(
                mask, loss_mat,
                torch.zeros(loss_mat.shape).to(device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat) / torch.sum(mask)
            total_loss += loss
            y_true += y.cpu().detach().numpy().tolist()
            y_pred += yp.cpu().detach().numpy().tolist()
            y_mask += mask.cpu().detach().numpy().tolist()
              
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_mask = np.array(y_mask)

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1.0) > 0 and np.sum(y_true[:, i] == 0.0) > 0:
            is_valid = (y_mask[:, i] > 0)
            auc_list.append(roc_auc_score(y_true[is_valid, i], y_pred[is_valid, i]))
            aupr_list.append(auprc(y_true[is_valid, i], y_pred[is_valid, i]))
            f1_list.append(f1_score(y_true[is_valid, i], (y_pred[is_valid, i]>=0.5).astype('int')))
            acc_list.append(accuracy_score(y_true[is_valid, i], (y_pred[is_valid, i]>=0.5).astype('int')))
        # else:
            # print('{} is invalid'.format(i))
    
    all_results=[auc_list, aupr_list, acc_list, f1_list]
    
    return total_loss/len(loader), np.array(all_results), y_true, y_pred, y_mask


def training(encoder, classifier, s_dataloader, t_dataloader, drug, prototype, params_str, **kwargs):
    network = TransferNetwork(encoder, classifier, len(drug)).to(kwargs['device'])
    

    input_dim = network.encoder.output_layer[0].in_features
    loss_infomax = InfoMax_loss(input_dim).to(kwargs['device'])
    

    all_params = list(network.parameters()) + list(loss_infomax.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=kwargs['lr'])
    
    LossFunc_class = nn.BCEWithLogitsLoss(reduction='none')
    best_loss_sum = np.inf

    node_x = torch.from_numpy(config.drug_feat.astype('float32'))
    node_x = node_x.to(kwargs['device'])
    edge_index = edge_extract(config.label_graph)
    edge_index = torch.from_numpy(edge_index.astype('int'))
    edge_index = edge_index.to(kwargs['device'])

    print("\n============TSJT (TSB) Training for TCGA data================")
    for epoch in range(int(kwargs['uda_num_epochs'])):
        return_loss_sum = 0
        len_loader = min(len(s_dataloader), len(t_dataloader))
        network.train(True)
        
        # Begining training
        for i, batch in enumerate(zip(t_dataloader, cycle(s_dataloader))):
            optimizer.zero_grad()
            
            # --- 1. Target Data Flow (Guidance) ---
            t_x = batch[0][0].to(kwargs['device'])
            t_type = batch[0][3].to(kwargs['device'])
            
            # Get target features for alignment
            _, t_feat = network(t_x, node_x, edge_index)
            
            # --- 2. Source Data Flow (Task) ---
            s_x = batch[1][0].to(kwargs['device'])
            s_y = batch[1][1].to(kwargs['device'])
            s_mask = batch[1][2].to(kwargs['device'])
            s_type = batch[1][3].to(kwargs['device'])
            
            # Get source features and predictions
            s_yp, s_feat = network(s_x, node_x, edge_index)
            
            # --- 3. Compute Losses ---
            # Source Loss: Classification
            loss_mat = LossFunc_class(s_yp, s_y.double())
            loss_mat = torch.where(s_mask>0, loss_mat,
                                   torch.zeros(loss_mat.shape).to(kwargs['device']))
            loss_source_val = torch.sum(loss_mat) / torch.sum(s_mask)
            
            # Target (Alignment) Loss: InfoMax (Prototype Alignment)
            # This serves as the "Target Gradient" driver in TSB
            loss_target_val = kwargs['beta'] * loss_infomax(s_feat, t_feat, s_type, t_type, prototype)
            
            # Total loss for logging (approx)
            total_loss_log = loss_source_val + loss_target_val
            
            # --- 4. TSB Gradient Projection Strategy ---
            # We need to manually handle gradients for the SHARED ENCODER.
            # Classifier and InfoMax weights are updated standardly.
            
            params_encoder = list(network.encoder.parameters())
            
            # Calculate gradients w.r.t losses SEPARATELY
            # Retain graph because we need to backward twice through encoder
            
            # A. Gradient from Target Objective (InfoMax) -> theta_T
            # We use autograd.grad to get gradients without accumulating into .grad yet
            grads_target = torch.autograd.grad(loss_target_val, params_encoder, retain_graph=True, allow_unused=True)
            
            # B. Gradient from Source Objective (Classification) -> theta_S
            grads_source = torch.autograd.grad(loss_source_val, params_encoder, retain_graph=True, allow_unused=True)
            
            # C. Update Non-Encoder Parameters (Classifier & InfoMax weights)
            # These don't need TSB, just standard backward
            # We backward loss_source to update classifier (encoder grads handled manually above)
            # We backward loss_target to update infomax params
            
            # Trick: backward on the sum, but we will OVERWRITE encoder grads with TSB later
            total_loss_combined = loss_source_val + loss_target_val
            total_loss_combined.backward()
            
            # --- D. Apply TSB Projection to Encoder Gradients ---
            # Flatten gradients for vector operations
            with torch.no_grad():
                # Helper to flatten
                def flatten_grads(grads):
                    flat = []
                    for g in grads:
                        if g is not None:
                            flat.append(g.view(-1))
                        else:
                            # Handle None grads (if any param unused)
                            pass 
                    return torch.cat(flat) if flat else torch.tensor([]).to(kwargs['device'])

                g_t_vec = flatten_grads(grads_target)
                g_s_vec = flatten_grads(grads_source)
                
                if len(g_t_vec) > 0 and len(g_s_vec) > 0:
                    # Calculate Cosine Similarity / Dot Product
                    dot_product = torch.dot(g_s_vec, g_t_vec)
                    norm_t = torch.dot(g_t_vec, g_t_vec) + 1e-8
                    
                    # Calculate Projection of Source onto Target direction
                    # theta_ST = ( <theta_S, theta_T> / |theta_T|^2 ) * theta_T
                    proj_scale = dot_product / norm_t
                    
                    # Re-construct the projected gradients layer by layer
                    idx = 0
                    for p, g_s, g_t in zip(params_encoder, grads_source, grads_target):
                        if p.grad is not None and g_s is not None and g_t is not None:
                            # g_s is the source gradient component
                            # g_t is the target gradient component
                            
                            # Calculate g_st (Parallel component) for this layer
                            # Note: proj_scale is global scalar computed from flattened vectors
                            g_st = proj_scale * g_t
                            
                            # Calculate g_sv (Vertical component)
                            g_sv = g_s - g_st
                            
                            # TSB Logic:
                            # 1. Always keep g_sv (does not hurt target, helps source)
                            # 2. Keep g_st only if dot_product > 0 (aligned)
                            
                            final_grad = g_sv
                            
                            if dot_product > 0:
                                final_grad += g_st
                            
                            # Add the pure target gradient (theta_T) as well?
                            # The standard update is: theta_new = theta - lr * (theta_total)
                            # In TSB paper: theta = theta - gamma1*theta_T - gamma2*(theta_SV + ...)
                            # Here loss_target_val already included 'beta', so g_t is scaled.
                            # We combine: Final Grad = (Optimized Source Grad) + (Target Grad)
                            
                            final_grad += g_t
                            
                            # Overwrite the accumulated gradient in the parameter
                            p.grad = final_grad
            
            # Step Optimizer
            optimizer.step()
            return_loss_sum += total_loss_log.cpu().detach().item()
        
        if (best_loss_sum > return_loss_sum):
            best_loss_sum = return_loss_sum
            # Renamed file to avoid confusion
            torch.save(network.state_dict(), os.path.join(kwargs['model_save_folder'], 'TransferNetwork.pt')) 
            
        if (epoch+1) % 1 == 0:
            print('TSB training epoch = {}'.format(epoch+1))
            print('Loss : {:.4f}'.format(return_loss_sum/len_loader))
    
    network.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'TransferNetwork.pt')))   
    
    return network


def testing(model, t_dataloader, drug, device):
    model.train(False)  
    node_x = torch.from_numpy(config.drug_feat.astype('float32'))
    node_x = node_x.to(device)  
    edge_index = edge_extract(config.label_graph)
    edge_index = torch.from_numpy(edge_index.astype('int'))
    edge_index = edge_index.to(device)
    test_loss, results, y_true, y_pred, y_mask = multi_eval_epoch(model=model,
                                          loader=t_dataloader, 
                                          node_x=node_x,
                                          edge_index=edge_index,
                                          drug=drug,
                                          device=device) 

    return test_loss, results, y_true, y_pred, y_mask
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import submitit
import os
import argparse
from graph_datasets.bipartite_graph_loader import BipartiteGraphLoader
import torch
from torch import autograd
import glob
import torch.nn.functional as F
import torch_geometric
import time
from graph_datasets.bipartite_graph_dataset import BipartiteGraphDataset, BipartiteGraphDatasets
from neural_nets.gnn_policy import GNNPolicy
from neural_nets.losses import LogScoreLoss, LinearScoreLoss
from tensorboardX import SummaryWriter as SummaryWriter
import numpy as np
import math
from IPython import embed
from graph_datasets.bipartite_graph_observations import augment_variable_features_with_dynamic_ones
from torchmetrics.functional import auroc
from os.path import exists
import pickle
import sys
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import DotProductSimilarity


class Args:
    pass


def multi_hot_encoding(input):
    max_val = torch.max(input, -1, keepdim=True).values - 1.0e-10
    multihot = input >= max_val
    return multihot.float()

initial_solution = dict()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#DEVICE = 'cpu'

log_score_loss_function = LogScoreLoss().to(DEVICE)
linear_score_loss_function = LinearScoreLoss().to(DEVICE)
bce_loss_function = torch.nn.BCEWithLogitsLoss(reduction="none").to(DEVICE)
infoNCE_loss_function = losses.NTXentLoss(temperature=0.07,distance=DotProductSimilarity()).to(DEVICE)

#data_loc = "training_data/"


def pad_tensor(input, pad_sizes, normalize, pad_value=-1e10):
    """
    This utility function splits a tensor and pads each split to make them all the same size, then stacks them.
    """
    max_pad_size = pad_sizes.max()
    output = input.split(pad_sizes.cpu().numpy().tolist())
    processed = []

    for i in range(len(output)):
        slice = output[i]
        if normalize:
            # Normalize the scores to ensure they fall in the [-1, 1] range
            max_val = torch.max(abs(output[i]))
            print(max_val)
            slice /= max_val
        processed.append(F.pad(slice, (0, max_pad_size-slice.size(0)), 'constant', pad_value))

    output = torch.stack(processed, dim=0)
    #output = torch.stack([F.pad(slice_, (0, max_pad_size-slice_.size(0)), 'constant', pad_value)
    #                      for slice_ in output], dim=0)
    return output

def load_policy_from_checkpoint(args):

    policy = GNNPolicy(args.gnn_type)
    
    try:
        ckpt = torch.load(args.warmstart, map_location=DEVICE)
        try_again = False
    except Exception as e:
        print("Checkpoint " + args.checkpoint + " not found, bailing out: " + str(e))
        sys.exit(1)
    

    policy.load_state_dict(ckpt.state_dict())
    #policy = policy.to(DEVICE)
    #model_version = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print("Loaded checkpoint")
    print(f"Will run evaluation on {DEVICE} device", flush=True)
    #embed()
    return policy


def process(args, policy, data_loader, optimizer=None):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    prefix = "Train" if optimizer else "Eval"

    #embed()
    if args.loss == "linear_score":
        loss_function = linear_score_loss_function
    elif args.loss == "log_score":
        loss_function = log_score_loss_function
    else:
        loss_function = bce_loss_function

    mean_loss = 0.0
    mean_acc = 0.0
    mean_auc = 0.0

    mean_offby = 0.0

    top_k = [1, 3, 5, 10]
    k_acc = [0.0, 0.0, 0.0, 0.0]

    n_iters = 0
    n_samples_processed = 0
    n_positive_samples = 0
    n_negative_samples = 0

    start = time.time()
    n_samples_previously_processed = 0

    history_window_size = 3

    with torch.set_grad_enabled(optimizer is not None):
        for batch in data_loader:
            assert not torch.isnan(batch.constraint_features).any()
            assert not torch.isnan(batch.edge_attr).any()
            assert not torch.isnan(batch.variable_features).any()
            assert not torch.isnan(batch.edge_index).any()
            assert not torch.isinf(batch.constraint_features).any()
            assert not torch.isinf(batch.edge_attr).any()
            assert not torch.isinf(batch.variable_features).any()
            assert not torch.isinf(batch.edge_index).any()

            batch = batch.to(DEVICE)

            # TO DO: Fix the dataset instead
            if torch.isnan(batch.candidate_scores).any():
                print("Skipping batch with NaN scores")
                continue


            global initial_solution
            batch = augment_variable_features_with_dynamic_ones(batch, args, initial_solution)
            
            # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
            try:
                logits = policy(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)
            except RuntimeError as e:
                print("Skipping batch due to error: " + str(e))
                continue

            # Index the results by the candidates, and split and pad them
            #pred_scores = pad_tensor(logits[batch.candidates], batch.nb_candidates, normalize=False)
            pred_scores = pad_tensor(logits, batch.nb_candidates, normalize=False)
            #pred_scores = torch.sigmoid(pred_scores)
            true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates, normalize=False)

            assert not torch.isnan(pred_scores).any()
            assert not torch.isnan(true_scores).any()
            #assert not torch.isnan(batch.candidate_choices).any()
            if args.loss == "cross_entropy":
                # Compute the usual cross-entropy classification loss
                loss = F.cross_entropy(pred_scores, batch.candidate_choices)
            elif args.loss == "bce":
                multi_hot_labels = multi_hot_encoding(true_scores)
                #print("lost function is bce")
                raw_loss = bce_loss_function(pred_scores, multi_hot_labels)
                batch_loss = torch.mean(raw_loss, 1)
                loss_sum = torch.sum(torch.mul(batch_loss, batch.batch_weight))
                loss = torch.div(loss_sum, torch.sum(batch.batch_weight))
                
            elif args.loss == "nt_xent":
                #    # Try https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#ntxentloss
                #    # Can also try https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#supconloss.
                #   assert False  # TBD     
                #    loss = loss_function(pred_labels, true_labels)
                    #embed()
                batch_size = pred_scores.shape[0]
                multi_hot_labels = multi_hot_encoding(true_scores)
                embeddings = torch.sigmoid(pred_scores)
                anchor_positive = []
                anchor_negative = []
                positive_idx = []
                negative_idx = []
                total_sample = batch_size
                #embed()
                for i in range(batch_size):
                    if batch.batch_weight[i].item() == 1:
                        #embed()
                        #anchor.append(i)
                        if len(batch.info["positive_samples"][i]) == 0: #due to unknown bugs for SC
                            #embed()
                            continue
                        ground_truth_improvement = max(batch.info["positive_labels"][i])
                        for j in range(len(batch.info["positive_samples"][i])):
                            improvement_j = batch.info["positive_labels"][i][j]
                            if improvement_j >= ground_truth_improvement * 0.5:
                                anchor_positive.append(i)
                                positive_idx.append(total_sample)
                                embeddings = torch.cat([embeddings, torch.tensor([batch.info["positive_samples"][i][j]]).to(DEVICE)])
                                total_sample += 1
                                n_positive_samples += 1
                        for j in range(len(batch.info["negative_samples"][i])):
                            improvement_j = batch.info["negative_labels"][i][j]
                            if improvement_j <= ground_truth_improvement * 0.05:
                                anchor_negative.append(i)
                                negative_idx.append(total_sample)
                                embeddings = torch.cat([embeddings, torch.tensor([batch.info["negative_samples"][i][j]]).to(DEVICE)])
                                total_sample += 1
                                n_negative_samples += 1

                triplets = (torch.tensor(anchor_positive).to(DEVICE), torch.tensor(positive_idx).to(DEVICE), torch.tensor(anchor_negative).to(DEVICE), torch.tensor(negative_idx).to(DEVICE))
                loss = infoNCE_loss_function(embeddings, indices_tuple = triplets)
            else:
                # use the log or linear score loss
                normalized_scores = normalize_tensor(batch.candidate_scores)
                loss = loss_function(logits[batch.candidates], normalized_scores)
            
            if  math.isnan(loss.item()):
                continue

            assert not math.isnan(loss.item())
            if not (loss.item() >= 0 or  torch.sum(batch.batch_weight).item() == 0):
                print("Error")
                embed()

            assert loss.item() >= 0 or  torch.sum(batch.batch_weight).item() == 0, f"loss = {loss.item()}, #samples = {torch.sum(batch.batch_weight).item()}"
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #embed()
            mean_loss += loss.item() * torch.sum(batch.batch_weight).item()
            #mean_loss += loss_sum.item()
            n_samples_processed += torch.sum(batch.batch_weight).item()# batch.num_graphs
            n_iters += 1
            #embed()
            for i in range(multi_hot_labels.shape[0]):
                if batch.batch_weight[i].item() == 0:
                    continue
                mean_auc += auroc(torch.sigmoid(pred_scores)[i], multi_hot_labels.int()[i], pos_label = 1).item()

            if n_iters % args.checkpoint_every == 0:
                end = time.time()
                speed = (n_samples_processed - n_samples_previously_processed) / (end - start)
                start = time.time()
                n_samples_previously_processed = n_samples_processed
                print(f"{prefix} loss: {mean_loss/n_samples_processed:0.3f}, auc: {mean_auc/n_samples_processed:0.3f}, speed: {speed} samples/s")

                if optimizer:
                    print("Checkpointing model")
                    torch.save(policy, args.checkpoint)
            

    if n_samples_processed > 0:
        mean_loss /= n_samples_processed
        mean_acc /= n_samples_processed
        mean_auc /= n_samples_processed
        mean_offby /= n_samples_processed
        for i in range(len(k_acc)):
            k_acc[i] /= n_samples_processed
    else:
        mean_loss = float("inf")
        mean_acc = 0
        mean_offby = float("inf")
        mean_auc = 0
        for i in range(len(k_acc)):
            k_acc[i] = 0

    print("n_samples_processed", n_samples_processed)
    return mean_loss, mean_auc #, mean_offby, k_acc



def train_model(args):

    train_loader = BipartiteGraphLoader(args.train_db, shuffle=True, first_k=args.train_db_first_k)
    valid_loader = BipartiteGraphLoader(args.valid_db, shuffle=False)

    print(f"Training on {train_loader.num_examples()} examples")
    print(f"Evaluating on {valid_loader.num_examples()} examples")

    #from IPython import embed; embed()
    print(F"Using DEVICE {DEVICE}")

    tb_writer = SummaryWriter(log_dir=args.tensorboard, comment="neural_LNS")

    policy = GNNPolicy(args.gnn_type).to(DEVICE)

    if not (args.warmstart is None):
        print("Warnstarting training, loading from checkpoint %s"%(args.warmstart))
        policy = load_policy_from_checkpoint(args)
        policy = policy.to(DEVICE)

    print(f"Checkpoint will be saved to {args.checkpoint}")

    num_of_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print("number of parameters =", num_of_parameters)

    learning_rate = args.lr
    best_valid_loss = float("inf")
    last_improved = 0

    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=args.weight_decay, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=learning_rate/10, verbose=False)

    for epoch in range(args.num_epochs):
        start = time.time()

        print(f"Starting epoch {epoch+1}", flush=True)
        with autograd.set_detect_anomaly(args.detect_anomalies):
            train_iterator = train_loader.load(batch_size=args.batch_size) #32
            train_loss, train_auc = process(args, policy, train_iterator, optimizer)
        print(f"Train loss: {train_loss:0.3f}, Train auc: {train_auc:0.3f}")

        valid_iterator = valid_loader.load(batch_size=args.batch_size) #32
        valid_loss, valid_auc = process(args, policy, valid_iterator, None)
        print(f"Valid loss: {valid_loss:0.3f}, Valid auc: {valid_auc:0.3f}")

        end = time.time()

        tb_writer.add_scalar("Train/Loss", train_loss, global_step=epoch)
        tb_writer.add_scalar("Train/Auc", train_auc, global_step=epoch)
        tb_writer.add_scalar("Valid/Loss", valid_loss, global_step=epoch)
        tb_writer.add_scalar("Valid/Auc", valid_auc, global_step=epoch)

        # Done with one epoch, we can freeze the normalization
        policy.freeze_normalization()
        # Anneal the learning rate if requested
        if args.anneal_lr:
            scheduler.step()

        # Save the trained model
        print(f"Done with epoch {epoch+1} in {end-start:.1f}s, checkpointing model", flush=True)
        torch.save(policy, args.checkpoint+"_epoch%d"%(epoch))

        # Check if we need to abort, adjust the learning rate, or just give up
        if math.isnan(train_loss) or math.isnan(valid_loss):
            print("NaN detected in loss, aborting")
            break
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            last_improved = epoch
            print("Checkpointing new best model in " + args.checkpoint + "_best")
            torch.save(policy, args.checkpoint + "_best")
        elif epoch - last_improved > args.give_up_after:
            print("Validation loss didn't improve for too many epochs, giving up")
            break
        elif epoch - last_improved > args.decay_lr_after:
            learning_rate /= 2
            print(f"Adjusting the learning rate to {learning_rate}")
            optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=learning_rate/10, verbose=False)
            # Give the model some time to improve with the new learning rate
            last_improved = epoch




def train(problem, gnn_type = "gat", feature_set = "feat2", batch_size = 32, warmstart = None, loss = "bce", notes = '', data_loc = None):
    print("Starting training model on " + problem, flush=True)
    print("gnn_type = ", gnn_type, "feature_set=", feature_set)

    assert not (data_loc is None), "no training data location provided"

    save_to_folder = "model/model_%s_%s_%s_%s_%s/" % (problem, feature_set, "no" if warmstart is None else "warmstart", loss, notes)
    try:
        os.mkdir(save_to_folder)
    except OSError as error:
        print(error) 

    

    args = Args()
    args.problem = problem
    args.num_epochs=30
    args.batch_size = batch_size
    args.lr=0.001
    args.anneal_lr = False
    args.decay_lr_after=20
    args.give_up_after=100
    args.train_db_first_k=None
    args.weight_decay=0.00005
    args.window_size = 3
    args.loss = loss
    args.gnn_type = gnn_type
    experiment = feature_set + "_" + args.gnn_type
    args.experiment =  experiment
    args.warmstart = warmstart
    args.tensorboard = save_to_folder + "neural_LNS_" + problem + "_" + experiment + ".tb"
    args.checkpoint = save_to_folder + "neural_LNS_" + problem + "_" + experiment + ".pt"
    args.checkpoint_every=40

    train_dbs = []
    valid_dbs = []
    dir = data_loc+"/*.db"
    num_data_file = 0
    for dataset in glob.glob(dir):
        num_data_file += 1
 
    validation_cutoff = int( num_data_file * 0.125)
    for i, dataset in enumerate(glob.glob(dir)):
        try:
            train_loader = BipartiteGraphLoader(dataset, shuffle=True)
        except:
            continue
        if train_loader.num_examples() == 0:
            continue
        if i >= validation_cutoff:
            train_dbs.append(dataset)
        else:
            valid_dbs.append(dataset)
    
    args.train_db = "+".join(train_dbs)
    args.valid_db = "+".join(valid_dbs)
    args.detect_anomalies = False
    train_model(args)

torch.cuda.empty_cache()



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-set", default="INDSET_train",
                        help="Problem set")
    parser.add_argument("--gnn-type", default="gat", type=str,
                        help="GNN type: gasse or gat")
    parser.add_argument("--feature-set", default="feat2", type=str,
                        help="feat1: Gasse's feature only; feat2: Gasse+Khalil features; feat3: feat2+LB RELAX features")
    parser.add_argument("--loss", default="nt_xent", type=str,
                        help="nt_xent: contrastive loss; bce: bce loss")
    parser.add_argument("--data-loc", default=None, type=str, 
                        help="Provide the dataset folder location")
    parser.add_argument("--wind-size", default=3, type = int,
                        help="window size = the number of past incumbent features in features")

    input_args = parser.parse_args()

    if input_args.data_loc is None:
        input_args.data_loc = "training_data/" + input_args.problem_set

    train(input_args.problem_set, gnn_type = input_args.gnn_type, feature_set = input_args.feature_set, loss = input_args.loss, data_loc = input_args.data_loc)


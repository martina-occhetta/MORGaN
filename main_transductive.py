import logging
import numpy as np
from tqdm import tqdm
import torch
import time

from src.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    WBLogger,
    get_current_lr,
    load_best_configs,
)
from src.datasets.data_util import load_dataset
from src.evaluation import node_classification_evaluation
from src.models import build_model


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def pretrain(model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, num_edge_types = 1, logger=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    # model_dtype = next(model.parameters()).dtype
    # x = x.to(model_dtype)
    # loss, loss_dict = model(x, graph.edge_index)

    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        #epoch_start = time.time()
        model.train()
        if num_edge_types != 1:
            loss, loss_dict = model(graph, x, num_edge_types)
        else:
            loss, loss_dict = model(graph, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

        if (epoch + 1) % 200 == 0:
            (acc, estp_acc), (auc, estp_auc), (aupr, estp_aupr), (precision, estp_precision), (recall, estp_recall), (f1, estp_f1) = node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob=False) 
            logger.note({
                **loss_dict,
                "accuracy": acc,
                "estp_accuracy": estp_acc,
                "auc": auc,
                "estp_auc": estp_auc,
                "aupr": aupr,
                "estp_aupr": estp_aupr,
                "precision": precision,
                "estp_precision": estp_precision,
                "recall": recall,
                "estp_recall": estp_recall
                },
            step=epoch)

        #epoch_end = time.time()
        #print(f"Epoch {epoch} runtime: {epoch_end - epoch_start:.2f} seconds")
            
    # return best_model
    return model


def main(args):
    start_time = time.time()  # Record the start time
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate
    num_edge_types = args.num_edge_types if args.num_edge_types is not None else 1

    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler

    graph, (num_features, num_classes) = load_dataset(dataset_name)
    # print(graph)
    args.num_features = num_features

    acc_list = []
    estp_acc_list = []
    test_auc_list = []
    estp_test_auc_list = []
    test_aupr_list = []
    estp_test_aupr_list = []
    test_precision_list = []
    estp_precision_f_list = []
    test_recall_list = []
    estp_recall_f_list = []
    
    for i, seed in enumerate(seeds):

        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        run_start = time.time()
        
        if logs:
            logger = WBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(args)
        model.to(device)
        #print(model)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                    # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None
        
        x = graph.x
        model_dtype = next(model.parameters()).dtype
        x = x.to(model_dtype)

        if not load_model:
            model = pretrain(model, graph, x, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, num_edge_types, logger)
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saving Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")
        
        model = model.to(device)
        model.eval()

        (test_acc, estp_test_acc), (test_auc, estp_test_auc), (test_aupr, estp_test_aupr), (test_precision, estp_test_precision), (test_recall, estp_test_recall), (test_f1, estp_f1) = node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob=False)
        
        logger.note({
            "test_accuracy": test_acc,
            "test_estp_accuracy": estp_test_acc,
            "test_auc": test_auc,
            "test_estp_auc": estp_test_auc,
            "test_aupr": test_aupr,
            "test_estp_aupr": estp_test_aupr,
            "test_precision": test_precision,
            "test_estp_precision": estp_test_precision,
            "test_recall": test_recall,
            "test_estp_recall": estp_test_recall,
            "test_f1": test_f1,
            "test_estp_f1": estp_f1
            },
            step=max_epoch)
        
        acc_list.append(test_acc)
        estp_acc_list.append(estp_test_acc)
        test_auc_list.append(test_auc)
        estp_test_auc_list.append(estp_test_auc)
        test_aupr_list.append(test_aupr)
        estp_test_aupr_list.append(estp_test_aupr)
        test_precision_list.append(test_precision)
        estp_precision_f_list.append(estp_test_precision)
        test_recall_list.append(test_recall)
        estp_recall_f_list.append(estp_test_recall)

        run_end = time.time()
        print(f"Run {i} total runtime: {run_end - run_start:.2f} seconds")

        if logger is not None:
            logger.finish()

    end_time = time.time()  # Record the end time after everything is done
    total_runtime = end_time - start_time
    print(f"Total runtime: {total_runtime:.2f} seconds")

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    final_auc, final_auc_std = np.mean(test_auc_list), np.std(test_auc_list)
    estp_auc, estp_auc_std = np.mean(estp_test_auc_list), np.std(estp_test_auc_list)
    final_aupr, final_aupr_std = np.mean(test_aupr_list), np.std(test_aupr_list)
    estp_aupr, estp_aupr_std = np.mean(estp_test_aupr_list), np.std(estp_test_aupr_list)
    print(f"# Final Accuracy: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# Early Stopping Accuracy: {estp_acc:.4f}±{estp_acc_std:.4f}")
    print(f"# Final AUC: {final_auc:.4f}±{final_auc_std:.4f}")
    print(f"# Early Stopping AUC: {estp_auc:.4f}±{estp_auc_std:.4f}")
    print(f"# Final AUPR: {final_aupr:.4f}±{final_aupr_std:.4f}")
    print(f"# Early Stopping AUPR: {estp_aupr:.4f}±{estp_aupr_std:.4f}")

    # final_metrics = {
    #     "final_accuracy": final_acc,
    #     "final_accuracy_std": final_acc_std,
    #     "early_stopping_accuracy": estp_acc,
    #     "early_stopping_accuracy_std": estp_acc_std,
    #     "final_auc": final_auc,
    #     "final_auc_std": final_auc_std,
    #     "final_aupr": final_aupr,
    #     "final_aupr_std": final_aupr_std,
    #     }
        
    # if logger is not None:
    #     logger.note(final_metrics, step=max_epoch)   

if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs/configs.yml")
    print(args)
    main(args)

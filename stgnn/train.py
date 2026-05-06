import json
import os
import pickle
import re
from copy import deepcopy
from itertools import product
from json import dumps

import dgl
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import utils
from args import get_args
from data.dataset import ReadmissionDataset
from model.model import GraphRNN, GConvLayers
from model.simple_lstm import SimpleLSTM
from model.simple_rnn import SimpleRNN


def get_device(args):
    args.cuda = torch.cuda.is_available()
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")

    if not args.cuda:
        message = (
            "CUDA is unavailable. "
            "torch.version.cuda={}, CUDA_VISIBLE_DEVICES={}".format(
                torch.version.cuda, cuda_visible_devices
            )
        )
        if args.require_cuda:
            raise RuntimeError(message)
        return torch.device("cpu"), message

    gpu_id = 0 if args.gpu_id is None else args.gpu_id
    device_count = torch.cuda.device_count()
    if gpu_id >= device_count:
        raise ValueError(
            "Requested gpu_id={} but PyTorch sees {} CUDA device(s). "
            "CUDA_VISIBLE_DEVICES={}".format(gpu_id, device_count, cuda_visible_devices)
        )

    torch.cuda.set_device(gpu_id)
    device = torch.device("cuda:{}".format(gpu_id))
    message = (
        "Using {} ({}) with torch.version.cuda={}, CUDA_VISIBLE_DEVICES={}".format(
            device,
            torch.cuda.get_device_name(device),
            torch.version.cuda,
            cuda_visible_devices,
        )
    )
    return device, message


def infer_early_stop_state(log_path):
    val_losses = []
    if not os.path.exists(log_path):
        return 1e10, 0

    with open(log_path) as f:
        for line in f:
            if "VAL -" not in line or "loss:" not in line:
                continue
            match = re.search(r"loss:\s*([0-9.eE+-]+)", line)
            if match is not None:
                val_losses.append(float(match.group(1)))

    if not val_losses:
        return 1e10, 0

    best_loss = float("inf")
    patience_count = 0
    for loss in val_losses:
        if loss < best_loss:
            best_loss = loss
            patience_count = 0
        else:
            patience_count += 1

    return val_losses[-1], patience_count


def set_cosine_scheduler_epoch(scheduler, optimizer, completed_epochs):
    scheduler.last_epoch = completed_epochs
    if hasattr(scheduler, "get_closed_form_lr"):
        lrs = scheduler.get_closed_form_lr()
    else:
        lrs = scheduler._get_closed_form_lr()
    for param_group, lr in zip(optimizer.param_groups, lrs):
        param_group["lr"] = lr
    scheduler._last_lr = lrs


def run_baseline_hparam_search(args):
    if args.model_name not in ("rnn", "lstm"):
        raise ValueError("Hyperparameter search is only implemented for model_name rnn or lstm.")
    if args.resume_run_dir is not None:
        raise ValueError("Hyperparameter search does not support resume_run_dir.")
    if not args.do_train:
        raise ValueError("Hyperparameter search requires do_train=True.")

    search_dir = utils.get_save_dir(args.save_dir, training=True)
    logger = utils.get_logger(search_dir, "hparam_search")
    logger.info("Running {} hyperparameter search in {}".format(args.model_name, search_dir))
    logger.info("Args: {}".format(dumps(vars(args), indent=4, sort_keys=True)))

    cat_emb_dims = (
        args.hparam_search_cat_emb_dims
        if args.ehr_encoder_name is not None
        else [args.cat_emb_dim]
    )
    search_space = list(
        product(
            args.hparam_search_lrs,
            args.hparam_search_num_rnn_layers,
            args.hparam_search_hidden_dims,
            args.hparam_search_dropouts,
            args.hparam_search_max_seq_len_ehr,
            cat_emb_dims,
        )
    )
    logger.info("Search space has {} trials.".format(len(search_space)))

    results = []
    trials_dir = os.path.join(search_dir, "trials")
    os.makedirs(trials_dir, exist_ok=True)

    for trial_idx, (
            lr,
            num_rnn_layers,
            hidden_dim,
            dropout,
            max_seq_len_ehr,
            cat_emb_dim,
    ) in enumerate(search_space, start=1):
        trial_args = deepcopy(args)
        trial_args.hparam_search = False
        trial_args.metric_name = "auroc"
        trial_args.maximize_metric = True
        trial_args.lr = lr
        trial_args.num_rnn_layers = num_rnn_layers
        trial_args.hidden_dim = hidden_dim
        trial_args.dropout = dropout
        trial_args.max_seq_len_ehr = max_seq_len_ehr
        trial_args.cat_emb_dim = cat_emb_dim
        trial_args.save_dir = os.path.join(trials_dir, "trial_{:03d}".format(trial_idx))

        hparams = {
            "lr": lr,
            "num_rnn_layers": num_rnn_layers,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "max_seq_len_ehr": max_seq_len_ehr,
            "cat_emb_dim": cat_emb_dim,
        }
        logger.info(
            "Starting trial {}/{}: {}".format(trial_idx, len(search_space), hparams)
        )
        val_auroc = main(trial_args)
        result = {
            "trial": trial_idx,
            "val_auroc": val_auroc,
            "save_dir": trial_args.save_dir,
            **hparams,
        }
        results.append(result)

        results_file = os.path.join(search_dir, "hparam_search_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4, sort_keys=True)

        best_result = max(results, key=lambda item: item["val_auroc"])
        with open(os.path.join(search_dir, "best_hparams.json"), "w") as f:
            json.dump(best_result, f, indent=4, sort_keys=True)
        logger.info(
            "Finished trial {}/{} with val_auroc={:.4f}. Current best={:.4f} from trial {}.".format(
                trial_idx,
                len(search_space),
                val_auroc,
                best_result["val_auroc"],
                best_result["trial"],
            )
        )

    best_result = max(results, key=lambda item: item["val_auroc"])
    logger.info("Best hyperparameters: {}".format(dumps(best_result, indent=4, sort_keys=True)))
    logger.info("Search results saved to {}".format(search_dir))
    return best_result["val_auroc"]


def auc_ci(y_true, y_pred, num_bootstraps=1000, ci=95):
    bootstrap_means = torch.empty(num_bootstraps)

    for i in range(num_bootstraps):
        indices = torch.randint(0, len(y_pred), (len(y_pred),))
        bootstrap_means[i] = roc_auc_score(y_true[indices], y_pred[indices])

    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile

    lower_bound = bootstrap_means.quantile(lower_percentile / 100)
    upper_bound = bootstrap_means.quantile(upper_percentile / 100)

    return lower_bound.item(), upper_bound.item()


def evaluate(
        args,
        model,
        graph,
        features,
        labels,
        nid,
        loss_fn,
        best_thresh=0.5,
        save_file=None,
        thresh_search=False,
        device="cpu",
        evaluate_ci=False,
):
    model.eval()
    with torch.no_grad():
        if args.model_name == "stgnn":
            logits, _ = model(graph, features.to(device))
            logits = logits.squeeze()[nid].cpu()
            loss = loss_fn(logits.to(device), labels[nid].to(device)).cpu()
        else:
            dataset = TensorDataset(features[nid], labels[nid])
            dataloader = DataLoader(dataset, batch_size=2048, shuffle=False)
            preds, losses = [], []
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs).squeeze()
                loss = loss_fn(logits, targets)

                losses.append(loss.item())
                preds.append(logits.cpu())

            loss = torch.tensor(losses).mean()
            logits = torch.cat(preds, dim=0)
        probs = torch.sigmoid(logits)

        preds = (probs >= best_thresh).int().numpy()

        eval_results = utils.eval_dict(
            y=labels[nid].numpy(),
            y_pred=preds,
            y_prob=probs.numpy(),
            average="binary",
            thresh_search=thresh_search,
            best_thresh=best_thresh,
        )
        eval_results["loss"] = loss.item()

        if evaluate_ci:
            lower_bound, upper_bound = auc_ci(labels[nid], probs, num_bootstraps=1000)
            eval_results["ci_lower"] = lower_bound
            eval_results["ci_upper"] = upper_bound
        if save_file is not None:
            with open(save_file, "wb") as pf:
                pickle.dump(
                    {
                        "labels": labels[nid].numpy(),
                        "probs": probs.numpy(),
                        "preds": preds,
                        "results": eval_results,
                    },
                    pf,
                )
    return eval_results


def main(args):
    device, device_message = get_device(args)

    # set random seed
    utils.seed_torch(seed=args.rand_seed)

    if args.hparam_search:
        return run_baseline_hparam_search(args)

    is_resume = args.resume_run_dir is not None

    # get save directories
    if is_resume:
        args.save_dir = args.resume_run_dir
    else:
        args.save_dir = utils.get_save_dir(
            args.save_dir, training=True if args.do_train else False
        )

    # save args
    args_file = os.path.join(
        args.save_dir, "resume_args.json" if is_resume else "args.json"
    )
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    logger = utils.get_logger(args.save_dir, "train")
    logger.info("Args: {}".format(dumps(vars(args), indent=4, sort_keys=True)))
    logger.info(device_message)

    # load graph
    logger.info("Constructing graph...")
    dataset = ReadmissionDataset(
        demo_file=args.demo_file,
        edge_ehr_file=args.edge_ehr_file,
        ehr_feature_file=args.ehr_feature_file,
        test_hadm_ids_file=args.test_hadm_ids_file,
        edge_modality=args.edge_modality,
        top_perc=args.edge_top_perc,
        gauss_kernel=args.use_gauss_kernel,
        max_seq_len_ehr=args.max_seq_len_ehr,
        standardize=True,
        ehr_types=args.ehr_types,
        is_graph=False,
    )
    g = dataset[0]
    cat_idxs = dataset.cat_idxs
    cat_dims = dataset.cat_dims

    features = g.ndata["feat"]
    labels = g.ndata["label"].float()
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    # ensure self-edges
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    n_edges = g.number_of_edges()
    n_nodes = g.number_of_nodes()
    logger.info(
        """----Graph Stats------
            # Nodes %d
            # Undirected edges %d
            # Average degree %d """
        % (
            n_nodes,
            int(n_edges / 2),
            g.in_degrees().float().mean().item(),
        )
    )

    train_nid = torch.nonzero(train_mask).squeeze()
    val_nid = torch.nonzero(val_mask).squeeze()
    test_nid = torch.nonzero(test_mask).squeeze()

    logger.info(
        "#Train samples: {:,}; positive percentage: {:.2%}".format(
            train_mask.sum(), labels[train_mask].mean()
        )
    )
    logger.info(
        "#Val samples: {:,}; positive percentage: {:.2%}".format(
            val_mask.sum(), labels[val_mask].mean()
        )
    )
    logger.info(
        "#Test samples: {:,}; positive percentage: {:.2%}".format(
            test_mask.sum(), labels[test_mask].mean(),
        )
    )

    if args.model_name == "stgnn":
        in_dim = features.shape[-1]
        print("Input dim:", in_dim)
        config = utils.get_config(args.model_name, args)
        model = GraphRNN(
            in_dim=in_dim,
            n_classes=args.num_classes,
            device=device,
            is_classifier=True,
            ehr_encoder_name=args.ehr_encoder_name,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=args.cat_emb_dim,
            **config
        )

    elif args.model_name == "rnn":
        in_dim = features.shape[-1]
        print("Input dim:", in_dim)
        model = SimpleRNN(
            input_size=in_dim,
            hidden_size=args.hidden_dim,
            output_size=1,
            num_layers=args.num_rnn_layers,
            dropout=args.dropout,
            ehr_encoder_name=args.ehr_encoder_name,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=args.cat_emb_dim,
        )
    elif args.model_name == "lstm":
        in_dim = features.shape[-1]
        print("Input dim:", in_dim)
        model = SimpleLSTM(
            input_size=in_dim,
            hidden_size=args.hidden_dim,
            output_size=1,
            num_layers=args.num_rnn_layers,
            dropout=args.dropout,
            ehr_encoder_name=args.ehr_encoder_name,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=args.cat_emb_dim,
        )
    else:
        in_dim = features.shape[-1]
        print("Input dim:", in_dim)
        config = utils.get_config(args.model_name, args)
        model = GConvLayers(
            in_dim=in_dim,
            num_classes=args.num_classes,
            is_classifier=True,
            device=device,
            **config
        )

    model.to(device)
    if args.compile_model and args.model_name != "stgnn":
        print("Compiling model...")
        model = torch.compile(model)

    # define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.l2_wd
    )

    # load model checkpoint
    checkpoint = None
    if args.load_model_path is not None:
        model, optimizer, checkpoint = utils.load_model_checkpoint(
            args.load_model_path, model, optimizer, return_checkpoint=True
        )

    # count params
    params = utils.count_parameters(model)
    logger.info("Trainable parameters: {}".format(params))

    # loss func
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor(args.pos_weight)).to(
        device
    )

    # checkpoint saver
    saver = utils.CheckpointSaver(
        save_dir=args.save_dir,
        metric_name=args.metric_name,
        maximize_metric=args.maximize_metric,
        log=logger,
    )

    # scheduler
    logger.info("Using cosine annealing scheduler...")
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    start_epoch = 0
    prev_val_loss = 1e10
    patience_count = 0

    if checkpoint is not None:
        start_epoch = checkpoint.get("epoch", 0)
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        else:
            set_cosine_scheduler_epoch(scheduler, optimizer, start_epoch)

        prev_val_loss = checkpoint.get("prev_val_loss", None)
        patience_count = checkpoint.get("patience_count", None)
        if prev_val_loss is None or patience_count is None:
            prev_val_loss, patience_count = infer_early_stop_state(
                os.path.join(args.save_dir, "log.txt")
            )

        logger.info(
            "Resuming {} from checkpoint epoch {}. Next epoch will be {}.".format(
                args.save_dir, start_epoch, start_epoch + 1
            )
        )
        logger.info(
            "Resume early-stop state: prev_val_loss={:.3f}, patience_count={}".format(
                prev_val_loss, patience_count
            )
        )

    if is_resume:
        best_path = os.path.join(args.save_dir, "best.pth.tar")
        if os.path.exists(best_path):
            resume_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            model = utils.load_model_checkpoint(best_path, model)
            best_eval_results = evaluate(
                args=args,
                model=model,
                graph=g,
                features=features,
                labels=labels,
                nid=val_nid,
                loss_fn=loss_fn,
                device=device,
            )
            saver.best_val = best_eval_results[args.metric_name]
            logger.info(
                "Existing best checkpoint validation {}: {:.3f}".format(
                    args.metric_name, saver.best_val
                )
            )
            model.load_state_dict(resume_model_state)
            model.to(device)

    if args.do_train:
        # Train
        logger.info("Training...")
        model.train()
        epoch = start_epoch
        early_stop = False

        while (epoch != args.num_epochs) and (not early_stop):

            epoch += 1
            scheduler_stepped = False
            logger.info("Starting epoch {}...".format(epoch))
            train_loss = []

            if args.model_name == "stgnn":
                optimizer.zero_grad()
                logits, _ = model(g, features.to(device))
                loss = loss_fn(logits.squeeze()[train_nid], labels[train_nid].to(device))
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
            else:
                dataset = TensorDataset(features[train_nid], labels[train_nid])
                dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
                for inputs, targets in tqdm(dataloader):
                    optimizer.zero_grad()
                    inputs, targets = inputs.to(device), targets.to(device)
                    logits = model(inputs).squeeze()
                    loss = loss_fn(logits, targets)
                    train_loss.append(loss.item())
                    loss.backward()
                    optimizer.step()

            # evaluate on val set
            if epoch % args.eval_every == 0:
                logger.info("Evaluating at epoch {}...".format(epoch))
                eval_results = evaluate(
                    args=args,
                    model=model,
                    graph=g,
                    features=features,
                    labels=labels,
                    nid=val_nid,
                    loss_fn=loss_fn,
                    device=device,
                )
                model.train()
                # accumulate patience for early stopping
                if eval_results["loss"] < prev_val_loss:
                    patience_count = 0
                else:
                    patience_count += 1
                prev_val_loss = eval_results["loss"]

                scheduler.step()
                scheduler_stepped = True
                saver.save(
                    epoch,
                    model,
                    optimizer,
                    eval_results[args.metric_name],
                    scheduler=scheduler,
                    prev_val_loss=prev_val_loss,
                    patience_count=patience_count,
                )

                # Early stop
                if patience_count == args.patience:
                    early_stop = True

                # Log to console

                logger.info("TRAIN - Epoch: {} | Loss: {:.4f}".format(epoch, sum(train_loss) / len(
                    train_loss)))
                results_str = ", ".join(
                    "{}: {:.3f}".format(k, eval_results[k]) for k in ["auroc", "loss"]
                )
                logger.info("VAL - {}".format(results_str))

            # step lr scheduler
            if not scheduler_stepped:
                scheduler.step()

        logger.info("Training DONE.")
        best_path = os.path.join(args.save_dir, "best.pth.tar")
        model = utils.load_model_checkpoint(best_path, model)
        model.to(device)

    # evaluate
    val_results = evaluate(
        args=args,
        model=model,
        graph=g,
        features=features,
        labels=labels,
        nid=val_nid,
        loss_fn=loss_fn,
        save_file=os.path.join(args.save_dir, "val_predictions.pkl"),
        thresh_search=args.thresh_search,
        device=device,
    )
    val_results_str = ", ".join(
        "{}: {:.3f}".format(k, v) for k, v in val_results.items()
    )
    logger.info("VAL - {}".format(val_results_str))

    # eval on test set
    test_results = evaluate(
        args=args,
        model=model,
        graph=g,
        features=features,
        labels=labels,
        nid=test_nid,
        loss_fn=loss_fn,
        save_file=os.path.join(args.save_dir, "test_predictions.pkl"),
        best_thresh=val_results["best_thresh"],
        device=device,
        evaluate_ci=True
    )
    test_results_str = ", ".join(
        "{}: {:.3f}".format(k, v) for k, v in test_results.items()
    )
    logger.info("TEST - {}".format(test_results_str))

    logger.info("Results saved to {}".format(args.save_dir))

    return val_results[args.metric_name]


if __name__ == "__main__":
    main(get_args())

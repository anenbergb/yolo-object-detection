import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Tuple
import logging

# Huggingface
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, gather_object
from safetensors.torch import load_model

from yolo.model import Yolo
from yolo.data import (
    CollateWithAnchors,
    CocoDataset,
    get_val_transforms,
    get_train_transforms,
)
from yolo.torchvision_utils import set_weight_decay
from yolo.utils import YoloLoss, DetectionMetrics
from yolo.anchors import DecodeDetections
from yolo.visualize import plot_grid


@dataclass
class TrainingConfig:
    output_dir: str
    overwrite_output_dir: bool = field(default=True)  # overwrite the old model
    eval_only: bool = field(default=False)
    start_epoch: int = field(default=0)
    resume_from_checkpoint: str = field(default=None)

    coco_dataset_root: str = field(default="/media/bryan/ssd01/fiftyone/coco-2017")

    train_batch_size: int = field(default=128)
    val_batch_size: int = field(default=256)

    epochs: int = field(default=100)
    limit_train_iters: int = field(default=0)
    limit_val_iters: int = field(default=0)

    # Optimizer configuration
    optimizer_name: str = field(default="adamw")
    momentum: float = field(default=0.937)
    # Linear warmup + CosineAnnealingLR
    # 2e-4 for AdamW
    # 0.01-0.05 for SGD
    lr: float = field(default=2e-4)
    lr_warmup_epochs: int = field(default=5)
    lr_warmup_decay: float = field(default=0.01)
    lr_min: float = field(default=0.0)

    # Regularization and Augmentation
    # 0.01 for AdamW
    # 0.0005 for SGD
    weight_decay: float = field(default=0.01)
    norm_weight_decay: float = field(default=0.0)
    gradient_max_norm: float = field(default=2.0)
    label_smoothing: float = 0.1

    # EMA configuration
    model_ema: bool = field(default=True)
    model_ema_steps: int = field(default=32)
    model_ema_decay: float = field(default=0.99998)

    mixed_precision: str = field(default="bf16")  # no for float32

    checkpoint_total_limit: int = field(default=3)
    checkpoint_epochs: int = field(default=1)
    save_image_epochs: int = field(default=1)
    seed: int = field(default=0)

    num_workers: int = field(default=2)

    # Yolo model configuration
    image_size: int = field(default=608)
    anchors: List[Tuple[int, int]] = field(
        default_factory=lambda: [
            (10, 13),
            (16, 30),
            (33, 23),
            (30, 61),
            (62, 45),
            (59, 119),
            (116, 90),
            (156, 198),
            (373, 326),
        ]
    )
    # hardcoded scales for now
    scales: List[int] = field(default_factory=lambda: [8, 16, 32])
    torchvision_backbone_name: str = field(default="resnext50_32x4d")

    # eval
    box_min_size: float = field(default=5.0)
    box_min_area: float = field(default=50.0)


def train_yolo(config: TrainingConfig):
    """
    Default values from https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/
    """
    num_anchors_per_scale = len(config.anchors) // len(config.scales)
    assert len(config.anchors) % len(config.scales) == 0

    project_config = ProjectConfiguration(
        project_dir=config.output_dir,
        # logging_dir
        automatic_checkpoint_naming=True,
        total_limit=config.checkpoint_total_limit,
        save_on_each_node=False,
        iteration=config.start_epoch,  # the current save iteration
    )
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        log_with="tensorboard",
        project_config=project_config,
        step_scheduler_with_optimizer=False,
        split_batches=False,
    )
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers(os.path.basename(config.output_dir))

    # logger = get_logger(__name__, log_level="DEBUG")
    # logger.info("INFO LEVEL", main_process_only=True)
    # logger.debug("DEBUG LEVEL", main_process_only=True)

    train_dataset = CocoDataset(
        dataset_root=config.coco_dataset_root,
        split="train",
        transform=get_train_transforms(resize_size=config.image_size),
    )
    val_dataset = CocoDataset(
        dataset_root=config.coco_dataset_root,
        split="validation",
        transform=get_val_transforms(resize_size=config.image_size),
    )

    collate_instance = CollateWithAnchors(
        config.anchors,
        config.scales,
        config.image_size,
        config.image_size,
        num_anchors_per_scale=num_anchors_per_scale,
        num_classes=train_dataset.num_classes,
    )

    detection_decoder = DecodeDetections(
        config.anchors,
        config.scales,
        config.image_size,
        config.image_size,
        class_names=train_dataset.class_names,
        num_anchors_per_scale=num_anchors_per_scale,
        box_min_size=config.box_min_size,
        box_min_area=config.box_min_area,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=config.num_workers,
        collate_fn=collate_instance,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=config.num_workers,
        collate_fn=collate_instance,
    )

    model = Yolo(
        train_dataset.num_classes,
        num_anchors_per_scale,
        backbone_name=config.torchvision_backbone_name,
    )

    parameters = set_weight_decay(
        model,
        config.weight_decay,
        norm_weight_decay=config.norm_weight_decay,
    )

    # consider swapping this for Adam
    optimizer_name = config.optimizer_name.lower()
    if config.optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    else:
        raise RuntimeError(f"Invalid optimizer {optimizer_name}. Only SGD and AdamW are supported.")

    scheduler1 = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=config.lr_warmup_decay,
        total_iters=config.lr_warmup_epochs,
    )
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs - config.lr_warmup_epochs, eta_min=config.lr_min
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [scheduler1, scheduler2], milestones=[config.lr_warmup_epochs]
    )

    yololoss = YoloLoss(label_smoothing=config.label_smoothing)

    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    # ONLY load the model weights from the checkpoint. Leave the optimizer and scheduler as is.
    if config.resume_from_checkpoint is not None and os.path.exists(config.resume_from_checkpoint):
        model_fpath = os.path.join(config.resume_from_checkpoint, "model.safetensors")
        assert os.path.exists(model_fpath), f"Model file {model_fpath} not found"
        accelerator.print(f"Loading model weights from {model_fpath}")
        weights_before = model.module.body.conv1.weight.detach().clone()
        load_model(
            accelerator._models[0],
            model_fpath,
            device=str(accelerator.device),
        )
        weight_after = model.module.body.conv1.weight.detach().clone()
        assert not torch.allclose(
            weights_before, weight_after
        ), "Model weights did not change after loading from checkpoint"

    if config.start_epoch > 0:
        accelerator.print(f"Resuming training from epoch {config.start_epoch}")
        for _ in range(config.start_epoch):
            scheduler.step()

    global_step = 0
    for epoch in range(config.start_epoch, config.epochs):
        total_loss = 0
        model.train()
        for step, batch in (
            progress_bar := tqdm(
                enumerate(train_dataloader),
                total=(len(train_dataloader) if config.limit_train_iters == 0 else config.limit_train_iters),
                disable=not accelerator.is_local_main_process,
                desc=f"Epoch {epoch}",
            )
        ):
            if config.limit_train_iters > 0 and step >= config.limit_train_iters:
                break

            optimizer.zero_grad()

            outputs = model(batch["image"])
            with accelerator.autocast():
                loss_dict = yololoss(outputs, batch)
                loss = loss_dict["loss"]

            total_loss += loss.detach().item()
            accelerator.backward(loss)

            accelerator.clip_grad_norm_(model.parameters(), config.gradient_max_norm)
            optimizer.step()

            current_lr = scheduler.get_last_lr()[0]
            logs = {
                "loss/train": loss.detach().item(),
                "lr": current_lr,
                "epoch": epoch,
            }
            progress_bar.set_postfix(**logs)
            logs["loss-objectness/train"] = loss_dict["objectness_loss"].detach().item()
            logs["loss-classification/train"] = loss_dict["class_loss"].detach().item()
            logs["loss-coordinates/train"] = loss_dict["coordinates_loss"].detach().item()
            accelerator.log(logs, step=global_step)
            global_step += 1

        if epoch % config.checkpoint_epochs == 0:
            accelerator.save_state()

        scheduler.step()  # once per epoch
        val_metrics = run_validation(
            accelerator,
            model,
            yololoss,
            val_dataloader,
            detection_decoder,
            limit_val_iters=config.limit_val_iters,
            global_step=global_step,
        )
        if accelerator.is_main_process:
            val_print_str = f"Validation metrics [Epoch {epoch}]: "
            AP = val_metrics.get("AP", 0.0)
            AP50 = val_metrics.get("AP50", 0.0)
            AP_person = val_metrics.get("AP-per-class/person", 0.0)
            AP_cat = val_metrics.get("AP-per-class/cat", 0.0)
            val_print_str += f"AP: {AP:.3f} AP50: {AP50:.3f} AP-person: {AP_person:.3f} AP-cat: {AP_cat:.3f}"
            accelerator.print(val_print_str)
            accelerator.log(val_metrics, step=global_step)

    accelerator.end_training()


def run_validation(
    accelerator,
    model,
    criterion,
    val_dataloader,
    detection_decoder,
    limit_val_iters=0,
    global_step=0,
):
    # debug multiprocessing by printing accelerator.local_process_index

    if accelerator.is_main_process:
        metrics = DetectionMetrics(val_dataloader.dataset.class_names)
        total_loss = torch.tensor(0.0, device=accelerator.device)
        total_objectness_loss = torch.tensor(0.0, device=accelerator.device)
        total_class_loss = torch.tensor(0.0, device=accelerator.device)
        total_coordinates_loss = torch.tensor(0.0, device=accelerator.device)

    model.eval()
    with torch.inference_mode():
        for step, batch in tqdm(
            enumerate(val_dataloader),
            total=len(val_dataloader) if limit_val_iters == 0 else limit_val_iters,
            disable=not accelerator.is_local_main_process,
            desc="Validation",
        ):
            if limit_val_iters > 0 and step >= limit_val_iters:
                break

            outputs = model(batch["image"])
            with accelerator.autocast():
                loss_dict = criterion(outputs, batch)

            outputs = accelerator.gather_for_metrics(outputs)
            loss_dict = accelerator.gather_for_metrics(loss_dict)

            images = accelerator.gather(batch["image"])
            # These are all lists of tensors
            boxes = gather_object(batch["boxes"])
            class_idx = gather_object(batch["class_idx"])
            iscrowd = gather_object(batch["iscrowd"])

            if accelerator.is_main_process:
                total_loss += loss_dict["loss"].sum()
                total_objectness_loss += loss_dict["objectness_loss"].sum()
                total_class_loss += loss_dict["class_loss"].sum()
                total_coordinates_loss += loss_dict["coordinates_loss"].sum()

                preds = detection_decoder(outputs, objectness_threshold=0.5, iou_threshold=0.5)  # cpu

                num_images = outputs["tx_ty_tw_th"].shape[0]
                # accelerator.gather_for_metrics will automatically truncate the last batch
                gathered_batch = {
                    "image": images[:num_images].detach().cpu(),
                    "boxes": [x.detach().cpu() for x in boxes[:num_images]],
                    "class_idx": [x.detach().cpu() for x in class_idx[:num_images]],
                    "iscrowd": [x.detach().cpu() for x in iscrowd[:num_images]],
                }
                metrics.update(preds, gathered_batch)

                # log the predictions for the first batch
                # Accelerate tensorboard tracker
                # https://github.com/huggingface/accelerate/blob/main/src/accelerate/tracking.py#L165
                if step == 0:
                    # lower objectness threshold yields more predictions
                    preds_low = detection_decoder(outputs, objectness_threshold=0.25, iou_threshold=0.5)  # cpu

                    batch_flat = []
                    for i in range(gathered_batch["image"].shape[0]):
                        item = {k: v[i] for k, v in gathered_batch.items()}
                        item["class_names"] = [
                            val_dataloader.dataset.class_names[c] for c in item["class_idx"].tolist()
                        ]
                        batch_flat.append(item)
                        preds[i]["image"] = gathered_batch["image"][i]
                        preds_low[i]["image"] = gathered_batch["image"][i]

                    vis_gt = plot_grid(
                        batch_flat,
                        max_images=25,
                        num_cols=5,
                        font_size=20,
                        box_color="green",
                        fig_scaling=3,
                    )
                    vis_preds = plot_grid(
                        preds,
                        max_images=25,
                        num_cols=5,
                        font_size=20,
                        box_color="red",
                        fig_scaling=3,
                    )
                    vis_preds_low = plot_grid(
                        preds_low,
                        max_images=25,
                        num_cols=5,
                        font_size=20,
                        box_color="red",
                        fig_scaling=3,
                    )

                    tensorboard = accelerator.get_tracker("tensorboard")
                    tensorboard.log_images(
                        {
                            "val-ground-truth": vis_gt,
                            "val-predictions": vis_preds,
                            "val-predictions-low": vis_preds_low,
                        },
                        step=global_step,
                        dataformats="HWC",
                    )

    val_metrics = {}
    if accelerator.is_main_process:
        val_metrics = metrics.compute()
        num_batches = len(val_dataloader) * accelerator.num_processes
        val_metrics["loss/val"] = (total_loss / num_batches).item()
        val_metrics["loss-objectness/val"] = (total_objectness_loss / num_batches).item()
        val_metrics["loss-classification/val"] = (total_class_loss / num_batches).item()
        val_metrics["loss-coordinates/val"] = (total_coordinates_loss / num_batches).item()

    accelerator.wait_for_everyone()
    return val_metrics


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """
Run training loop for YoloV3 object detection model on the COCO dataset.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/media/bryan/ssd01/expr/yolo_from_scratch/run01",
        help="Path to save the model",
    )
    parser.add_argument(
        "--coco-dataset-root",
        type=str,
        default="/media/bryan/ssd01/fiftyone/coco-2017",
        help="Path to the COCO dataset",
    )
    parser.add_argument("--train-batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=600, help="Epochs")
    parser.add_argument("--lr-warmup-epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument(
        "--limit-train-iters",
        type=int,
        default=0,
        help="Limit number of training iterations per epoch",
    )
    parser.add_argument(
        "--limit-val-iters",
        type=int,
        default=0,
        help="Limit number of val iterations per epoch",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="Start epoch, useful for resuming training.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint folder that the training should resume from.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation on the validation set.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig()
    args = get_args()
    config = TrainingConfig(
        output_dir=args.output_dir,
        coco_dataset_root=args.coco_dataset_root,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        epochs=args.epochs,
        lr_warmup_epochs=args.lr_warmup_epochs,
        limit_train_iters=args.limit_train_iters,
        limit_val_iters=args.limit_val_iters,
        start_epoch=args.start_epoch,
        resume_from_checkpoint=args.resume_from_checkpoint,
        eval_only=args.eval_only,
    )

    sys.exit(train_yolo(config))

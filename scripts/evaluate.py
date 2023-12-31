import os
import json
import logging

import torch
import torch.nn.functional as F

from open_clip import get_cast_dtype
from precision import get_autocast
from distributed import is_master
from zero_shot import zero_shot_classifier
from zero_shot import run as run_zero_shot
from train import get_clip_metrics, maybe_compute_generative_loss

from tqdm import tqdm 

def evaluate(model, data, epoch, args, tb_writer=None, eval_path = ""):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    if ('val' in data or 'test' in data) and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        # Pick the correct data loader
        dataloader = data['val'].dataloader if 'val' in data else data['test'].dataloader

        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0 and len(eval_path) > 0:
                    with open(eval_path, 'a') as f:
                        f.write(f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\n")
                        f.write(f"Clip Loss: {cumulative_loss / num_samples:.6f}\n")
                    # logging.info(
                    #     f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                    #     f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    # if gen_loss is not None:
                    #     cumulative_gen_loss += gen_loss * batch_size
                    #     logging.info(
                    #         f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics
    if len(eval_path) > 0:
        with open(eval_path, 'a') as f:
           f.write(f"Eval Epoch: {epoch} " + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()]) + '\n')

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)
        if len(eval_path) > 0:
            base_path = '/'.join(eval_path.split('/')[:-1])
            with open(os.path.join(base_path, "results.txt"), "a+") as f:
                f.write(str(metrics))
                f.write('\n')
    return metrics


def zero_shot_eval(model, data, epoch, args):
    if "zeroshot-val" not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}

    logging.info("Running zero-shot evaluation.")
    classifier = zero_shot_classifier(model, data["classnames"], data["template"], args)

    results = {}
    top1, top5 = run_zero_shot(model, classifier, data["zeroshot-val"].dataloader, args)
    results["zeroshot-val-top1"] = top1

    return results
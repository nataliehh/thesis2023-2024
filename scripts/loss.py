import torch
import torch.nn.functional as F
from open_clip.loss import ClipLoss
import ot

has_distributed = False


def create_loss(args):
    return SemiSupervisedClipLoss(
        args.method,
        local_loss=args.local_loss,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
    )


class SemiSupervisedClipLoss(ClipLoss):
    def __init__(self, method, pseudo_label_type="ot-image", local_loss=False, gather_with_grad=False, cache_labels=False,
            rank=0, world_size=1, use_horovod=False):
        super().__init__(local_loss=local_loss, gather_with_grad=gather_with_grad, cache_labels=cache_labels, rank=rank,
            world_size=world_size, use_horovod=use_horovod)

        assert method in ["base", "ours"]
        self.method = method
        self.pseudo_label_type = pseudo_label_type

    def supervised_loss(self, logits_per_image, logits_per_text, labels):
        return (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels))/2
    def forward(self, image_features, text_features, logit_scale, output_dict=False,
                query_features=None, keyword_features=None, keyword_labels=None):
        device = image_features.device
        losses = dict()  # dict of losses

        logits_per_image = logit_scale * image_features @ text_features.T
        labels = self.get_ground_truth(device, image_features.shape[0])

        # compute loss
        if self.method == "base": # Supervised CLIP loss
            logits_per_text = logits_per_image.T
            losses["contrastive_loss"] = self.supervised_loss(logits_per_image, logits_per_text, labels)

        else:
            logits_per_query = logit_scale * query_features @ text_features.T
            logits_per_text = torch.cat([logits_per_image, logits_per_query]).T

            # Supervised CLIP loss
            losses["contrastive_loss"] = self.supervised_loss(logits_per_image, logits_per_text, labels)

            # caption-level loss
            plan = get_assignments(query_features, image_features, text_features, logit_scale, self.pseudo_label_type)
            pseudo_labels = plan @ F.one_hot(labels).float()

            losses["caption_loss"] = (soft_cross_entropy(logits_per_query, pseudo_labels)) / 2

            # keyword-level loss
            selected = []
            pseudo_labels_keyword = torch.zeros(len(query_features), len(keyword_features), device=device)
            for query_id, q in enumerate(query_features):
                sample_id = int(plan[query_id].max(dim=0)[1])  # nearest one
                candidates = keyword_labels[sample_id, :, 0].nonzero().flatten().tolist()

                if len(candidates) > 0:
                    selected.append(query_id)
                    if len(candidates) == 1:
                        pseudo_labels_keyword[query_id, candidates[0]] = 1
                    else:
                        k = torch.stack([keyword_features[i] for i in candidates])
                        sim = (q @ k.T * logit_scale).detach()
                        prob = sim / sim.sum()
                        for i in range(len(sim)):
                            pseudo_labels_keyword[query_id, candidates[i]] = prob[i]

            logits_per_query_keyword = logit_scale * query_features @ keyword_features.T
            losses["keyword_loss"] = (soft_cross_entropy(logits_per_query_keyword, pseudo_labels_keyword)) / 2

        return losses if output_dict else sum(losses.items())


def get_assignments(query, image, text, logit_scale, pseudo_label_type):
    if pseudo_label_type == "hard-image":
        plan = hard_nn(query, image)
    elif pseudo_label_type == "hard-text":
        plan = hard_nn(query, text)
    elif pseudo_label_type == "soft-image":
        plan = soft_nn(query, image, logit_scale)
    elif pseudo_label_type == "soft-text":
        plan = soft_nn(query, text, logit_scale)
    elif pseudo_label_type == "ot-image":
        plan = ot_plan(query, image, logit_scale)
    elif pseudo_label_type == "ot-text":
        plan = ot_plan(query, text, logit_scale)
    else:
        raise NotImplementedError
    return plan

# Hard PL takes the most likely label as the pseudo-label
def hard_nn(query, support): 
    _, idx = (query @ support.T).max(dim=1)
    plan = F.one_hot(idx, len(support)).float()
    return plan

# Soft PL takes the softmax-ed similarity to the other entries as the pseudo-label
def soft_nn(query, support, logit_scale):
    plan = F.softmax(query @ support.T * logit_scale, dim=1)
    return plan


def ot_plan(query, support, logit_scale):
    C = 1 - query @ support.T  # (query, batch)
    reg = 1 / logit_scale  # learned temperature

    dim_p, dim_q = C.shape
    p = torch.ones(dim_p, device=C.device, dtype=torch.double) / dim_p
    q = torch.ones(dim_q, device=C.device, dtype=torch.double) / dim_q
    P = ot.bregman.sinkhorn(p, q, C, reg=reg, numItermax=10)

    plan = P / P.sum(dim=1, keepdim=True)
    plan = plan.type_as(support)
    return plan


def soft_cross_entropy(outputs, targets, weight=1.):
    loss = -targets * F.log_softmax(outputs, dim=1)
    return (loss * weight).sum(dim=1).mean()
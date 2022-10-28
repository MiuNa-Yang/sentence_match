from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW


def get_scheduler(model, conf):
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=conf.lr)

    def lr_lambda(current_step):
        if current_step < conf.warmup_steps:
            return float(current_step) / float(max(1, conf.warmup_steps))
        return 1
        # return max(0, 1 - (current_step - warmup_steps) / (all_steps - warmup_steps))

    scheduler = {
        "scheduler": LambdaLR(optimizer, lr_lambda),
        "name": "learning_rate",
        "interval": "step",
        "frequency": 1,
    }
    return optimizer, scheduler

from .loss import cross_entropy
from .optim import clip_grad_norm
from .schedule import lr_schedule


def loss_on_batch(model, x, y, ignore_index=-1):
    logits = model(x)
    return cross_entropy(logits, y, ignore_index=ignore_index)


def train_step(model, opt, x, y, max_norm=1.0, ignore_index=-1):
    opt.zero_grad()
    loss = loss_on_batch(model, x, y, ignore_index)
    loss.backward()
    clip_grad_norm(opt.params, max_norm)
    opt.step()
    return float(loss.data)


def train(model, opt, batches, peak_lr, warmup_steps, total_steps,
          min_lr=0.0, max_norm=1.0, ignore_index=-1, start_step=0, log_every=0):
    history = []
    for i, (x, y) in enumerate(batches):
        step = start_step + i
        opt.lr = lr_schedule(step, peak_lr, warmup_steps, total_steps, min_lr)
        loss = train_step(model, opt, x, y, max_norm, ignore_index)
        history.append(loss)
        if log_every and (step + 1) % log_every == 0:
            print(f"step {step + 1}/{total_steps} lr={opt.lr:.5f} loss={loss:.4f}")
    return history

import inspect
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


from pycocoevalcap.bleu.bleu import Bleu


class LossChecker:
    def __init__(self, num_losses):
        self.num_losses = num_losses

        self.losses = [ [] for _ in range(self.num_losses) ]

    def update(self, *loss_vals):
        assert len(loss_vals) == self.num_losses

        for i, loss_val in enumerate(loss_vals):
            self.losses[i].append(loss_val)

    def mean(self, last=0):
        mean_losses = [ 0. for _ in range(self.num_losses) ]
        for i, loss in enumerate(self.losses):
            _loss = loss[-last:]
            mean_losses[i] = sum(_loss) / len(_loss)
        return mean_losses


def parse_batch(batch):
    vids, feats, captions = batch
    feats = [ feat.cuda() for feat in feats ]
    feats = torch.cat(feats, dim=2)
    captions = captions.long().cuda()
    return vids, feats, captions

def entropy_loss(x, ignore_mask):
    b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    b = b.sum(dim=2)
    b[ignore_mask] = 0 # Mask after sum to avoid memory issue.
    b = -1.0 * b.sum(dim=0).mean() # Sum along words and mean along batch
    return b

def train(e, model, optimizer, train_iter, vocab, teacher_forcing_ratio, reg_lambda, gradient_clip):
    model.train()

    loss_checker = LossChecker(3)
    PAD_idx = vocab.word2idx['<PAD>']
    t = tqdm(train_iter)
    for batch in t:
        _, feats, captions = parse_batch(batch)
        optimizer.zero_grad()
        output = model(feats, captions, teacher_forcing_ratio)
        cross_entropy_loss = F.nll_loss(output[1:].view(-1, vocab.n_vocabs),
                                        captions[1:].contiguous().view(-1),
                                        ignore_index=PAD_idx)
        entropy_loss = losses.entropy_loss(output[1:], ignore_mask=(captions[1:] == PAD_idx))
        loss = cross_entropy_loss + reg_lambda * entropy_loss
        loss.backward()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        loss_checker.update(loss.item(), cross_entropy_loss.item(), entropy_loss.item())
        t.set_description("[Epoch #{0}] loss: {2:.3f} = (CE: {3:.3f}) + (Ent: {1} * {4:.3f})".format(
            e, reg_lambda, *loss_checker.mean(last=10)))

    total_loss, cross_entropy_loss, entropy_loss = loss_checker.mean()
    loss = {
        'total': total_loss,
        'cross_entropy': cross_entropy_loss,
        'entropy': entropy_loss,
    }
    return loss


def test(model, val_iter, vocab, reg_lambda):
    model.eval()

    loss_checker = LossChecker(3)
    PAD_idx = vocab.word2idx['<PAD>']
    for b, batch in enumerate(val_iter, 1):
        _, feats, captions = parse_batch(batch)
        output = model(feats, captions)
        cross_entropy_loss = F.nll_loss(output[1:].view(-1, vocab.n_vocabs),
                          captions[1:].contiguous().view(-1),
                          ignore_index=PAD_idx)
        entropy_loss = losses.entropy_loss(output[1:], ignore_mask=(captions[1:] == PAD_idx))
        loss = cross_entropy_loss + reg_lambda * entropy_loss
        loss_checker.update(loss.item(), cross_entropy_loss.item(), entropy_loss.item())

    total_loss, cross_entropy_loss, entropy_loss = loss_checker.mean()
    loss = {
        'total': total_loss,
        'cross_entropy': cross_entropy_loss,
        'entropy': entropy_loss,
    }
    return loss


def get_predicted_captions(data_iter, model, vocab, beam_width=5, beam_alpha=0.):
    def build_onlyonce_iter(data_iter):
        onlyonce_dataset = {}
        for batch in iter(data_iter):
            vids, feats, _ = parse_batch(batch)
            for vid, feat in zip(vids, feats):
                if vid not in onlyonce_dataset:
                    onlyonce_dataset[vid] = feat
        onlyonce_iter = []
        vids = onlyonce_dataset.keys()
        feats = onlyonce_dataset.values()
        batch_size = 100
        while len(vids) > 0:
            onlyonce_iter.append(( vids[:batch_size], torch.stack(feats[:batch_size]) ))
            vids = vids[batch_size:]
            feats = feats[batch_size:]
        return onlyonce_iter

    model.eval()

    onlyonce_iter = build_onlyonce_iter(data_iter)

    vid2pred = {}
    EOS_idx = vocab.word2idx['<EOS>']
    for vids, feats in onlyonce_iter:
        captions = model.describe(feats, beam_width=beam_width, beam_alpha=beam_alpha)
        captions = [ idxs_to_sentence(caption, vocab.idx2word, EOS_idx) for caption in captions ]
        vid2pred.update({ v: p for v, p in zip(vids, captions) })
    return vid2pred


def get_groundtruth_captions(data_iter, vocab):
    vid2GTs = {}
    EOS_idx = vocab.word2idx['<EOS>']
    for batch in iter(data_iter):
        vids, _, captions = parse_batch(batch)
        captions = captions.transpose(0, 1)
        for vid, caption in zip(vids, captions):
            if vid not in vid2GTs:
                vid2GTs[vid] = []
            caption = idxs_to_sentence(caption, vocab.idx2word, EOS_idx)
            vid2GTs[vid].append(caption)
    return vid2GTs


def score(vid2pred, vid2GTs):
    assert set(vid2pred.keys()) == set(vid2GTs.keys())
    vid2idx = { v: i for i, v in enumerate(vid2pred.keys()) }
    refs = { vid2idx[vid]: GTs for vid, GTs in vid2GTs.items() }
    hypos = { vid2idx[vid]: [ pred ] for vid, pred in vid2pred.items() }

    scores = calc_scores(refs, hypos)
    return scores


# refers: https://github.com/zhegan27/SCN_for_video_captioning/blob/master/SCN_evaluation.py
def calc_scores(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


def evaluate(data_iter, model, vocab, beam_width=5, beam_alpha=0.):
    vid2pred = get_predicted_captions(data_iter, model, vocab, beam_width=5, beam_alpha=0.)
    vid2GTs = get_groundtruth_captions(data_iter, vocab)
    scores = score(vid2pred, vid2GTs)
    return scores


# refers: https://stackoverflow.com/questions/52660985/pytorch-how-to-get-learning-rate-during-training
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def idxs_to_sentence(idxs, idx2word, EOS_idx):
    words = []
    for idx in idxs[1:]:
        idx = idx.item()
        if idx == EOS_idx:
            break
        word = idx2word[idx]
        words.append(word)
    sentence = ' '.join(words)
    return sentence


def cls_to_dict(cls):
    properties = dir(cls)
    properties = [ p for p in properties if not p.startswith("__") ]
    d = {}
    for p in properties:
        v = getattr(cls, p)
        if inspect.isclass(v):
            v = cls_to_dict(v)
            v['was_class'] = True
        d[p] = v
    return d


# refers https://stackoverflow.com/questions/1305532/convert-nested-python-dict-to-object
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def dict_to_cls(d):
    cls = Struct(**d)
    properties = dir(cls)
    properties = [ p for p in properties if not p.startswith("__") ]
    for p in properties:
        v = getattr(cls, p)
        if isinstance(v, dict) and 'was_class' in v and v['was_class']:
            v = dict_to_cls(v)
        setattr(cls, p, v)
    return cls


def load_checkpoint(model, ckpt_fpath):
    checkpoint = torch.load(ckpt_fpath)
    model.decoder.load_state_dict(checkpoint['decoder'])
    return model


def save_checkpoint(e, model, ckpt_fpath, config):
    ckpt_dpath = os.path.dirname(ckpt_fpath)
    if not os.path.exists(ckpt_dpath):
        os.makedirs(ckpt_dpath)

    torch.save({
        'epoch': e,
        'decoder': model.decoder.state_dict(),
        'config': cls_to_dict(config),
    }, ckpt_fpath)


def save_result(vid2pred, vid2GTs, save_fpath):
    assert set(vid2pred.keys()) == set(vid2GTs.keys())

    save_dpath = os.path.dirname(save_fpath)
    if not os.path.exists(save_dpath):
        os.makedirs(save_dpath)

    vids = vid2pred.keys()
    with open(save_fpath, 'w') as fout:
        for vid in vids:
            GTs = ' / '.join(vid2GTs[vid])
            pred = vid2pred[vid]
            line = ', '.join([ str(vid), pred, GTs ])
            fout.write("{}\n".format(line))
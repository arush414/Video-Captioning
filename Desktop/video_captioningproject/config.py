import time
import os
class MSVDConfig:
    model="MSVD_InceptionV4"
    size=1536
    video_fpath="/kaggle/input/msvd-inceptionv4-weights/MSVD_InceptionV4.hdf5" 
    caption_fpath="/kaggle/input/msvdvideodataset/MSVDtable.csv"
    train_video_fpath = "/kaggle/working/videofile/{}_train.hdf5".format(model)
    val_video_fpath = "/kaggle/working/videofile/{}_val.hdf5".format(model)
    test_video_fpath = "/kaggle/working/videofile/{}_test.hdf5".format(model)

    train_metadata_fpath = "/kaggle/working/msvdvideo/MSVDtrain.csv"
    val_metadata_fpath = "/kaggle/working/msvdvideo/MSVDval.csv"
    test_metadata_fpath = "/kaggle/working/msvdvideo/MSVDtest.csv"
    

class VocabConfig:
    init_word2idx = { '<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3 } #provided tokens for some cases like unknown words or end of sentences etc
    embedding_size = 468
    


class MSVDLoaderConfig:
    train_caption_fpath="data/MSVD/metadata/train.csv"
    val_caption_fpath="data/MSVD/metadata/val.csv"
    test_caption_fpath="data/MSVD/metadata/test.csv"
    min_count=1
    max_caption_len=30
    
    phase_video_feat_fpath_tpl="data/{}/features/{}_{}.hdf5"
    frame_sampling_method='uniform'; assert frame_sampling_method in ['uniform','random'] # we would be defining uniform sampling and random sampling methods in the dataset notebook
    frame_max_len=300//5
    frame_sample_len=28
    num_workers=3
    
class DecoderConfig:
    rnn_type='LSTM'; assert rnn_type in ['LSTM','GRU'] # feel free to try out Gru
    rnn_num_layers=1
    rnn_num_directions=1; assert rnn_num_directions in [1,2]
    rnn_hidden_size=512
    rnn_attn_size=256
    rnn_dropout=0.4
    rnn_teacher_forcing_ratio=1.0
    
    
class TrainConfig:
    
   

    corpus="MSVD"
    vocab = VocabConfig
    loader = MSVDLoaderConfig
    decoder = DecoderConfig
    feat="InceptionV4" # model to be used....
    # Make sure to get its corresponding weights as well 

    """ Optimization """
    epochs = 30
    batch_size = 200
    shuffle = True
    optimizer = "AMSGrad"
    gradient_clip = 5.0 # None if not used
    lr = 5e-5
    lr_decay_start_from = 20
    lr_decay_gamma = 0.5
    lr_decay_patience = 5
    weight_decay = 1e-5
    reg_lambda = 0.

    """ Pretrained Model """
    pretrained_decoder_fpath = None

    """ Evaluate """
    metrics = 'Bleu_4'

    """ ID """
    exp_id = "SA-LSTM"
    feat_id = "FEAT {} mfl-{} fsl-{} mcl-{}".format('+'.join(feat), loader.frame_max_len, loader.frame_sample_len,
                                                    loader.max_caption_len)
    embedding_id = "EMB {}".format(vocab.embedding_size)
    decoder_id = "DEC {}-{}-l{}-h{} at-{}".format(
        ["uni", "bi"][decoder.rnn_num_directions-1], decoder.rnn_type,
        decoder.rnn_num_layers, decoder.rnn_hidden_size, decoder.rnn_attn_size)
    optimizer_id = "OPTIM {} lr-{}-dc-{}-{}-{}-wd-{} rg-{}".format(
        optimizer, lr, lr_decay_start_from, lr_decay_gamma, lr_decay_patience, weight_decay, reg_lambda)
    hyperparams_id = "bs-{}".format(batch_size)
    if gradient_clip is not None:
        hyperparams_id += " gc-{}".format(gradient_clip)
    timestamp = time.strftime("%y%m%d-%H:%M:%S", time.gmtime())
    model_id = " | ".join([ exp_id, corpus, feat_id, embedding_id, decoder_id, optimizer_id, timestamp ])

    """ Log """
    log_dpath = "logs/{}".format(model_id)
    ckpt_dpath = os.path.join("checkpoints", model_id)
    ckpt_fpath_tpl = os.path.join(ckpt_dpath, "{}.ckpt")
    save_from = 1
    save_every = 1

    """ TensorboardX """
    tx_train_loss = "loss/train"
    tx_train_cross_entropy_loss = "loss/train/decoder_CE"
    tx_train_entropy_loss = "loss/train/decoder_reg"
    tx_val_loss = "loss/val"
    tx_val_cross_entropy_loss = "loss/val/decoder_CE"
    tx_val_entropy_loss = "loss/val/decoder_reg"
    tx_lr = "params/decoder_LR"


class EvalConfig:
    ckpt_fpath = "checkpoints/SA-LSTM | MSVD | FEAT InceptionV4 mcl-30 | EMB 468 | DEC uni-LSTM-l1-h512 at-256 | OPTIM AMSGrad lr-0.0002-dc-20-0.9-5-wd-1e-05 rg-0.001 | 190307-19:10:55/35.ckpt"
    result_dpath = "results"
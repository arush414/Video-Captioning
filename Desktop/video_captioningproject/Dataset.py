from __future__ import print_function, division
import numpy as np
import pandas as pd
import h5py
import torch
import math
from nltk.tokenize import wordpunct_tokenize # This takes in text input and tokenizes text based on whitespace and punctuation
import re
import string

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms







class uniformsampling:
    def __init__(self,n_sample):
        self.n_sample=n_sample

    def __call__(self,frames):

        n_frames=len(frames)
        if(n_frames<self.n_sample):

            return frames

        sample_indices= np.linspace(0,n_frames-1,self.n_sample, dtype= int)
        samples=[frames[i] for i in sample_indices]
        return samples
    
class randomsampling:
    def __init__(self,n_sample):
        self.n_sample=n_sample

    def __call__(self,frames):
        n_frames= len(frames)

        if(n_frames<self.n_sample):
            return frames

        block_len=int(n_frames/ self.n_sample) # split into blocks
        start_final= n_frames-block_len-1
        uniformly_sampled_indices= np.linspace(0,start_final,self.n_sample,dtype=int)
        random_noise= np.random.choice(block_len,self.n_sample,replace=True)# Since block_len might be less tham self.n_sample we do replace=True so some frames might be repeated
        randomly_sampled_indices= uniformly_sampled_indices+random_noise
        samples=[frames[i] for i in randomly_sampled_indices]
        return samples


    '''By using both uniformly spaced indices and random noise,
    the code attempts to sample frames from different positions within each block,
    creating a more diverse selection. This approach ensures that the frames are not concentrated in a single region of the frames
    but are spread out across different sections.'''



class trimiflonger:
    def __init__(self,n_sample):
        self.n_sample=n_sample

    def __call__(self,frames):

        if(len(frames)>self.n_sample):
            frames= frames[:self.n_sample]

        return frames
    
    
class ToTensor:
    
    def __init__(self,dtype=None):
        self.dtype=dtype

    def __call__(self,array):

        np_array= np.asarray(array)
        t= torch.from_numpy(np_array)
        if(self.dtype):
            t=t.type(self.dtype)

        return t
    
    
class zeropaddiflesser:
    
    def __init__(self,n_sample):
        self.n_sample=n_sample

    def __call__(self,frames):

        while(len(frames)<self.n_sample):
            
            frames=np.vstack([frames,np.zeros_like(frames[0])])

        return frames
    
    
class NLTKWordpunctTokenizer:
    
    def __call__(self, sentence):
        return wordpunct_tokenize(sentence)
    
    
    
class TrimExceptAscii:
    def __init__(self, corpus):
        self.corpus = corpus

    def __call__(self, sentence):
        if self.corpus == "MSVD":
            s = sentence.encode('ascii', 'ignore')
        elif self.corpus == "MSR-VTT":
            s = sentence.encode('ascii', 'ignore')
        return s


# Create an instance of the class for the "MSVD" corpus
trimmer = TrimExceptAscii("MSVD")

# Example sentence containing non-ASCII characters
sentence = "This is a café."

# Call the instance as a function to trim and encode the sentence
result = trimmer(sentence)

# Print the result
print(result)



class removepunctuation:
    
    def __init__(self):

        self.remove=re.compile('[%s]' % re.escape(string.punctuation))

    def __call__(self,sentence):
        return self.remove.sub('',sentence)


    

class Lowercase:
    
    def __call__(self, sentence):
        return sentence.lower()
    
   
class SplitWithWhiteSpace:
    
    def __call__(self, sentence):
        return sentence.split()
    
    
class Truncate:
    def __init__(self, n_word):
        self.n_word = n_word

    def __call__(self, words):
        return words[:self.n_word]


class PadFirst:
    def __init__(self, token):
        self.token = token

    def __call__(self, words):
        return [ self.token ] + words


class PadLast:
    def __init__(self, token):
        self.token = token

    def __call__(self, words):
        return words + [ self.token ]


class PadToLength:
    def __init__(self, token, length):
        self.token = token
        self.length = length

    def __call__(self, words):
        n_pads = self.length - len(words)
        return words + [ self.token ] * n_pads


class ToIndex:
    def __init__(self, word2idx):
        self.word2idx = word2idx

    def __call__(self, words):
        return [ self.word2idx[word] for word in words ]
    
    
    
class MSVDVocab(object):
    
    def __init__(self,caption_fpath, init_word2idx, min_count=1, transform=str.split):

        self.caption_fpath = caption_fpath
        self.min_count = min_count
        self.transform = transform
        self.word2idx= defaultdict(lambda: init_word2idx['<UNK>'])
        self.word2idx.update(init_word2idx)
        self.idx2word = { v: k for k, v in self.word2idx.items() }
        self.word_freq_dict=defaultdict(lambda: 0)
        self.n_vocabs= len(self.word2idx)
        self.n_words=self.n_vocabs
        self.max_sentence_len=-1

        self.load_captions()
        self.build()


    def load_captions(self):
        df=pd.read_csv(self.caption_fpath)
        df=df[df['Language']=='English']
        df=df[pd.notnull(df['Description'])]
        captions=df['Description'].values
        self.captions=captions


    def build(self) :

        for caption in self.captions:
            
            
            words=self.transform(caption)
            self.max_sentence_len=max( self.max_sentence_len,len(words))
            for word in words:
                
                self.word_freq_dict[word]+=1

        self.n_vocabs_untrimmed= len(self.word_freq_dict)
        self.n_words_untrimmed= sum(list(self.word_freq_dict.values()))

        keep_words= [word for word, freq in self.word_freq_dict.items() if freq>=self.min_count]

        for idx, word in enumerate(keep_words,len(self.word2idx)):
            
            self.word2idx[word]=idx
            self.idx2word[idx]=word

        self.n_vocabs=len(self.word2idx)
        self.n_words=sum([self.word_freq_dict[word] for word in keep_words])




class MSVDDataset(Dataset):
    

    def __init__(self, C, phase, caption_fpath, transform_frame=None, transform_caption=None):

        self.C = C
        self.phase = phase # train,val,test
        self.caption_fpath = caption_fpath
        self.transform_frame = transform_frame
        self.transform_caption = transform_caption
      #optional transformations to be applied to video frames and captions
        self.video_feats = defaultdict(lambda: []) # defaultdicts, which will store video features,with video IDs as keys.
        self.captions = defaultdict(lambda: [])
        self.data = []

        self.build_video_caption_pairs()

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        video,video_features,caption= self.data[idx]

        if self.transform_frame:
            video_features=[ self.transform_frame(feat) for feat in video_features]
        if self.transform_caption:
            caption= self.transform_caption(caption)

        return video,video_features,caption

    def load_video_feats(self):

        for model in self.C.feat.models:

            fpath = self.C.loader.phase_video_feat_fpath_tpl.format(self.C.corpus, model, self.phase)

            fin = h5py.File(fpath, 'r')
            for vid in fin.keys():
                feats = fin[vid].value

            # Fix the number of frames for each video
                if len(feats) < self.C.loader.frame_max_len:
                    num_paddings = self.C.loader.frame_max_len - len(feats)
                    feats = feats.tolist() + [ np.zeros_like(feats[0]) for _ in range(num_paddings) ]
                    feats = np.asarray(feats)
                else:
                    feats = feats[:self.C.loader.frame_max_len]
                assert len(feats) == self.C.loader.frame_max_len

            # Sample fixed number of frames
                sampled_idxs = np.linspace(0, len(feats) - 1, self.C.loader.frame_sample_len, dtype=int)
                feats = feats[sampled_idxs]

                self.video_feats[vid].append(feats)
        fin.close()
    
    def build_video_caption_pairs(self):
        
        self.load_video_feats()
        self.load_captions()

        for vid in self.video_feats.keys():
            video_feats = self.video_feats[vid]
            for caption in self.captions[vid]:
                self.data.append(( vid, video_feats, caption ))
                
    def load_captions(self):
        df = pd.read_csv(self.caption_fpath)
        df = df[df['Language'] == 'English']
        df = df[[ 'VideoID', 'Start', 'End', 'Description' ]]
        df = df[pd.notnull(df['Description'])]

        for video_id, start, end, caption in df.values:
            vid = "{}_{}_{}".format(video_id, start, end)
            self.captions[vid].append(caption)
            
                   




class MSVDCorpus(object):
    def __init__(self,C,vocab_cls=MSVDVocab,dataset_cls=MSVDDataset):
        self.C=C # TrainConfig class
        self.vocab=None
        self.train_dataset=None
        self.val_dataset=None
        self.val_data_loader = None
        self.test_dataset = None
        self.test_data_loader = None
        self.CustomVocab=vocab_cls
        self.CustomDataset=dataset_cls
        
        self.transform_sentence= transform.Compose([TrimExceptAscii(self.C.corpus),
            Lowercase(),
            RemovePunctuation(),
            SplitWithWhiteSpace(),
            Truncate(self.C.loader.max_caption_len),])
        self.build()
        
    def build(self):
        self.build_vocab()
        self.build_data_loaders()
    
    def build_vocab(self):
        
        self.vocab=self.MSVDVocab(self.C.loader.train_caption_fpath,self.C.vocab.init_word2idx,
                                  self.C.loader.min_count,transform=self.transform_sentence
                                 )
    def build_data_loaders(self):
        
        if self.C.loader.frame_sampling_method=="uniform":
            Sample=UniformSample
        elif self.C.loader.frame_sampling_method=="random":
            Sample=RandomSample
        
        else:
            raise NotImplementedError("Unknown frame sampling method: {}".format(self.C.loader.frame_sampling_method))
            
           
        self.transform_frame=transforms.Compose([Sample(self.C.loader.frame_sample_len),
                                               ToTensor(torch.float),])
        self.transform_caption=transforms.Compose([
            self.transform_sentence,
            ToIndex(self.vocab.word2idx),
            PadFirst(self.vocab.word2idx['<SOS>']),
            PadLast(self.vocab.word2idx['<EOS>']),
            PadToLength(self.vocab.word2idx['<PAD>'], self.vocab.max_sentence_len + 2), # +2 for <SOS> and <EOS>
            ToTensor(torch.long),
            
        ])
        self.train_dataset = self.build_dataset("train", self.C.loader.train_caption_fpath)
        self.val_dataset = self.build_dataset("val", self.C.loader.val_caption_fpath)
        self.test_dataset = self.build_dataset("test", self.C.loader.test_caption_fpath)

        self.train_data_loader = self.build_data_loader(self.train_dataset)
        self.val_data_loader = self.build_data_loader(self.val_dataset)
        self.test_data_loader = self.build_data_loader(self.test_dataset)

        
    def build_dataset(self,phase,caption_fpath):
        dataset=self.MSVDDataser(self.C,phase,caption_fpath,transform_frame=self.transform_frame,transform_caption=self.transform_caption)
        return dataset
    
    def collate_fn(self, batch):
        vids, video_feats, captions = zip(*batch)
        video_feats_list = zip(*video_feats)

        video_feats_list = [ torch.stack(video_feats) for video_feats in video_feats_list ]
        captions = torch.stack(captions)

        video_feats_list = [ video_feats.float() for video_feats in video_feats_list ]
        captions = captions.float()

        """ (batch, seq, feat) -> (seq, batch, feat) """
        captions = captions.transpose(0, 1)

        return vids, video_feats_list, captions

    def build_data_loader(self, dataset):
        data_loader = DataLoader(
            dataset,
            batch_size=self.C.batch_size,
            shuffle=False, # If sampler is specified, shuffle must be False.
            sampler=RandomSampler(dataset, replacement=False),
            num_workers=self.C.loader.num_workers,
            collate_fn=self.collate_fn)
        return data_loader





    
    


    
    
    
    
        



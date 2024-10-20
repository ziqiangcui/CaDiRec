import sys
sys.path.append('./')
import time
import functools
import torch
import numpy as np
from tqdm import tqdm
import numpy as np
import torch
from tqdm import tqdm, trange
from models.cadirec import CaDiRec 
from models.gaussian_diffusion import SpacedDiffusion,space_timesteps
from utils import get_full_sort_score, EarlyStopping
from models import gaussian_diffusion as gd
from .step_sample import UniformSampler

class Trainer:
    def __init__(self, args, device, generator):

        self.args = args
        self.device = device
        self.start_epoch = 0    # define the start epoch for keepon trainingzhonss

        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self.generator = generator
        self.train_dataloader = generator.train_dataloader
        self.valid_dataloader = generator.valid_dataloader
        self.test_dataloader = generator.test_dataloader
        self.item_size = generator.item_size
        self.args.item_size = generator.item_size
        self.generator = generator
        
        self._create_model()
        self._set_optimizer()
        # self._set_stopper()
        

    def _create_model(self):
        self.model = CaDiRec(self.device, self.args)
        self.model.to(self.device)
        
        betas = gd.get_named_beta_schedule(self.args.noise_schedule, self.args.diffusion_steps)
        timestep_respacing = [self.args.diffusion_steps]
        self.diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(self.args.diffusion_steps, timestep_respacing),
            betas=betas,
            rescale_timesteps=self.args.rescale_timesteps,
            predict_xstart=self.args.predict_xstart,
            learn_sigmas = self.args.learn_sigma,
            sigma_small = self.args.sigma_small,
            use_kl = self.args.use_kl,
            rescale_learned_sigmas=self.args.rescale_learned_sigmas
        )
        self.schedule_sampler = UniformSampler(self.diffusion)
    
    
    def _set_optimizer(self):

        self.optimizer = torch.optim.Adam(self.model.parameters(),  
                                        lr=self.args.learning_rate,
                                        # betas=(0.9, 0.999),
                                        weight_decay=self.args.weight_decay)


    def _train_one_epoch(self, epoch, only_bert_train=False, is_diffusion_train=True):

        tr_loss = 0
        tr_diff_loss = 0
        tr_sas_rec_loss = 0
        tr_sas_cl_loss = 0
        train_time = []
      
        self.model.train()
        prog_iter = tqdm(self.train_dataloader, leave=False, desc='Training')
  
        for batch in prog_iter:

            train_start = time.time()
       
            input_ids, target_pos, target_neg, attention_mask, masked_indices0 = \
                                                                                    batch["input_ids"].to(self.device), \
                                                                                    batch["target_pos"].to(self.device), \
                                                                                    batch["target_neg"].to(self.device), \
                                                                                    batch["attention_mask"].to(self.device), \
                                                                                    batch["masked_indices0"].to(self.device)
         
            self.optimizer.zero_grad()
            
            t, weights = self.schedule_sampler.sample(input_ids.shape[0], self.device)
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.model,
                t,
                input_ids,
                masked_indices0.long(), 
                attention_mask
            )
            
            diff_mse_loss, diff_nll_loss, aug_seq1, aug_seq2 = compute_losses()
            model_emb = torch.nn.Embedding(
                            num_embeddings=self.args.item_size, 
                            embedding_dim=self.args.hidden_size, 
                            _weight=self.model.item_embedding.weight.clone().cpu()
                            ).eval().requires_grad_(False)
            model_emb.to(self.device)
            aug_seq1_emb = model_emb(aug_seq1)
            aug_seq2_emb = model_emb(aug_seq2)
            

            sas_rec_loss = self.model.calculate_rec_loss(input_ids, target_pos, target_neg)
            
            
            if epoch <= self.args.warm_up_epochs:
                sas_cl_loss = 0.0
                loss = sas_rec_loss + self.args.gamma * diff_mse_loss + self.args.beta * diff_nll_loss
            else:
                sas_cl_loss = self.model.calculate_cl_loss(aug_seq1, aug_seq2, aug_seq1_emb, aug_seq2_emb) 
                loss = sas_rec_loss + self.args.gamma * diff_mse_loss + self.args.beta * diff_nll_loss + self.args.alpha * sas_cl_loss

        
            loss.backward()
            self.optimizer.step()
            
            tr_diff_loss += diff_nll_loss / len(self.train_dataloader)
         
            tr_sas_rec_loss += sas_rec_loss / len(self.train_dataloader)
            tr_sas_cl_loss += sas_cl_loss / len(self.train_dataloader)
            tr_loss += loss.item() / len(self.train_dataloader)

            train_end = time.time()
            train_time.append(train_end-train_start)
        # if epoch %10==0:
        #     print("aug_seq1", aug_seq1[masked_indices0])
        #     print("aug_seq2", aug_seq2[masked_indices0])
        print(f' epoch {epoch}: diff_loss {tr_diff_loss:.4f}', end='   ')
        print(f'sas_rec_loss {tr_sas_rec_loss:.4f}', end='   ')
        print(f'sas_cl_loss {tr_sas_cl_loss:.4f}', end='   ')
        print(f'total_loss {tr_loss:.4f}')


    def train(self):
        print("********** Running training **********")
        train_time = []
        early_stopping = EarlyStopping(self.args.checkpoint_path, patience=40, verbose=True)
   
        for epoch in trange(self.start_epoch, self.start_epoch + int(self.args.epochs), desc="Epoch"):
            
            t = self._train_one_epoch(epoch, only_bert_train=False, is_diffusion_train=True)
        
            train_time.append(t) 
            
            if epoch % 10 == 0:
                self.eval(epoch, test=False)  #valid
                
                self.eval(epoch, test=True)  #test
            
    
    def eval(self, epoch, test=False):
      
        self.model.eval()
        if not test:
            print("********** Running eval **********")
            prog_iter = tqdm(self.valid_dataloader, leave=False, desc='eval')
        else:
            print("********** Running test **********")
            prog_iter = tqdm(self.test_dataloader, leave=False, desc='test')

        scores = []
        labels = []
        for batch in prog_iter:
            user_ids, input_ids, label_items = \
                                                batch["user_id"].to(self.device), \
                                                batch["input_ids"].to(self.device), \
                                                batch["answer"].to(self.device), \
                                                                      
            bs_scores = self.model.full_sort_predict(input_ids).detach().cpu()
            
            batch_user_index = user_ids.cpu().numpy()
            # print("bs_scores", bs_scores.shape)
            # print("valid_rating_matrix", (self.generator.test_rating_matrix[batch_user_index].toarray() > 0).shape)
            if not test:
                bs_scores[self.generator.valid_rating_matrix[batch_user_index].toarray() > 0] = -100
            else:
                bs_scores[self.generator.test_rating_matrix[batch_user_index].toarray() > 0] = -100
            bs_labels = label_items.reshape(-1,1).cpu()
            scores.append(bs_scores)
            labels.append(bs_labels)
            
        scores = torch.cat(scores, axis=0).numpy()
        partitioned_indices = np.argpartition(-scores, 20, axis=1)[:, :20]
        pred_list = partitioned_indices[np.arange(scores.shape[0])[:, None], np.argsort(-scores[np.arange(scores.shape[0])[:, None], partitioned_indices], axis=1)].tolist()
        labels = torch.cat(labels, axis=0).numpy().tolist()
        get_full_sort_score(epoch, labels, pred_list)
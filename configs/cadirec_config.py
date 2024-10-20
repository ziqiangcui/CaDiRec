import argparse
from typing import List

def get_config():
    parser = argparse.ArgumentParser()
    #************SASRec*******************
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu')
    parser.add_argument("--dataset", default="Toys_and_Games", choices=['ml-1m', 'Beauty', 'Sports_and_Outdoors', 'Yelp', 'Toys_and_Games'], help="Choose the dataset")
    parser.add_argument("--model_name", default="diffsas", help="Choose the model")
    parser.add_argument("--model_idx", default="0", help="Choose the idx")
    parser.add_argument("--data_path", default="./data/", help="Choose the dataset path")
    parser.add_argument("--output_dir", default="./saved_models/", help="save the model")
    parser.add_argument("--check_path", default='', type=str,help="the save path of checkpoints for different running")
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--warm_up_epochs', type=int, default=-1, help='Number of warmup epochs')
    parser.add_argument('--filter_num', type=int, default=5, help='filter_num')
    parser.add_argument('--train_batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=512, help='Batch size for testing')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    
    parser.add_argument('--n_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--n_heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size (embedding size)')
    parser.add_argument('--inner_size', type=int, default=128, help='Dimensionality in feed-forward layer')
    parser.add_argument('--sasrec_dropout_prob', type=float, default=0.5, help='Hidden dropout probability of SASRec')
    parser.add_argument('--attn_dropout_prob', type=float, default=0.5, help='Hidden dropout probability of SASRec')
    parser.add_argument('--hidden_act', type=str, default='gelu', help='Activation function for hidden layers')
    parser.add_argument('--layer_norm_eps', type=float, default=1e-12, help='Epsilon value for layer normalization')

    parser.add_argument('--initializer_range', type=float, default=0.02, help='Initializer range for model parameters')
    parser.add_argument('--loss_type', type=str, default='BPR', help='Type of loss function')
    parser.add_argument('--max_seq_length', type=int, default=50, help='max sequence length')
    
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature for nce loss')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    

    # #*************Diffusion***************
    parser.add_argument("--num_hidden_layers", type=int, default=1, help='num_hidden_layers.')
    parser.add_argument("--intermediate_size", type=int, default=128, help='intermediate_size.')
    parser.add_argument("--num_attention_heads", type=int, default=2, help='num_attention_heads.')
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.2, help='attention_probs_dropout_prob.')
    parser.add_argument("--max_position_embeddings", type=int, default=50, help='max_position_embeddings.')
    parser.add_argument("--max_relative_positions", type=int, default=-1, help='max_relative_positions.')
    parser.add_argument("--type_vocab_size", type=int, default=0, help='type_vocab_size.')
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.2, help='hidden_dropout_prob.')
    parser.add_argument("--is_decoder", type=bool, default=False, help='is decoder')
    parser.add_argument("--mask_type", type=str, default='prob', help='mask_type.')
    parser.add_argument("--pad_token_id", type=int, default=0, help='padding id.')
    parser.add_argument("--pos_att_type", type=List[str], default=["p2c","c2p"], help='pos_att_type.')
    parser.add_argument("--relative_attention", type=bool, default=True, help='relative_attention')
    parser.add_argument("--position_biased_input", type=bool, default=False, help='position_biased_input')
    
    parser.add_argument('--alpha', type=float, default=0.1, help='ratio of constrastive learning loss')
    parser.add_argument('--beta', type=float, default=0.2, help='ratio of diffusion nll loss')
    parser.add_argument('--gamma', type=float, default=0.0, help='ratio of diffusion mse loss')
    parser.add_argument('--mlm_probability_train', type=float, default=0.2, help='mlm_probability for train')
    parser.add_argument('--mlm_probability', type=float, default=0.2, help='mlm_probability')
    
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--microbatch", type=int, default=0)
    parser.add_argument("--ema_rate", type=float, default=1.0)
    parser.add_argument("--resume_checkpoint", default="none")
    parser.add_argument("--schedule_sampler", default="lossaware")
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--inference_sampling_steps", type=int, default=5)
    parser.add_argument("--noise_schedule", default="sqrt")
    parser.add_argument("--timestep_respacing", default="")
    parser.add_argument("--use_plm_init", default="no")
    parser.add_argument("--notes", default="folder-notes")
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_fp16", type=bool, default=False)
    parser.add_argument("--fp16_scale_growth", type=float, default=0.001)
    parser.add_argument("--gradient_clipping", type=float, default=-1.0)
    parser.add_argument("--learn_sigma", type=bool, default=False)
    parser.add_argument("--use_kl", type=bool, default=False)
    parser.add_argument("--predict_xstart", type=bool, default=True)
    parser.add_argument("--rescale_timesteps", type=bool, default=True)
    parser.add_argument("--rescale_learned_sigmas", type=bool, default=False)
    parser.add_argument("--sigma_small", type=bool, default=False)
    parser.add_argument("--emb_scale_factor", type=float, default=1.0)


    return parser.parse_args()
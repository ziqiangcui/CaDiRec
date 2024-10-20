import os
import torch
from tqdm import tqdm
from data_generators.data_generator import DataGenerator
from CaDiRec.trainers.trainer import Trainer 
from utils import set_seed
from configs.sasrec_diffusion_config import get_config
import os


def main():
    args = get_config()
    set_seed(args.seed) 
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    # os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu_id)
    device = torch.device("cuda:"+str(0) if torch.cuda.is_available() else "cpu")
    print(device)
    args_str = f"{args.model_name}-{args.dataset}-{args.model_idx}"
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    data_generator = DataGenerator(args)
    trainer = Trainer(args, device, data_generator)
    trainer.train()
 

if __name__ == "__main__":
    main()







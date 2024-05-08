import time
from glob import glob
from tqdm import tqdm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
import torchvision.utils as tvu

from models.ddpm.diffusion import DDPM
from models.improved_ddpm.script_util import i_DDPM
from utils.text_dic import SRC_TRG_TXT_DIC
from utils.diffusion_utils import get_beta_schedule, denoising_step
from losses import id_loss
from losses.clip_loss import CLIPLoss
from datasets.data_utils import get_dataset, get_dataloader
from configs.paths_config import DATASET_PATHS, MODEL_PATHS, HYBRID_MODEL_PATHS, HYBRID_CONFIG
from datasets.imagenet_dic import IMAGENET_DIC
from utils.align_utils import run_alignment

class DiffusionCLIP(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
                             (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

        if self.args.edit_attr is None:
            self.src_txts = self.args.src_txts
            self.trg_txts = self.args.trg_txts
        else:
            self.src_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][0]
            self.trg_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][1]

    def clip_finetune(self):
        print(self.args.exp)
        print(f'   {self.src_txts}')
        print(f'-> {self.trg_txts}')

        # ----------- Model -----------#
        """if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset == "CelebA_HQ":
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        elif self.config.data.dataset == "AFHQ":
            pass
        else:
            raise ValueError
        """

        ##### Selet Diffusion Model weights related to the selected dataset #######
        if(self.config.data.dataset == "Retinal_Fundus"):
            url = "../drive/MyDrive/CLIPDiffusionRetinal/ema_0.9999_290000_eyepacs_extra_data_balancing_4_classes_v1.pt"
        elif(self.config.data.dataset == "AFHQ"):
            url = "../drive/MyDrive/CLIPDiffusion/afhq_dog_4m.pt"
        elif(self.config.data.dataset == "CelebA_HQ"):
            url = "../drive/MyDrive/CLIPDiffusion/celeba_hq.ckpt"
        elif(self.config.data.dataset == "Chexpert"):
            url = "../drive/MyDrive/ChestDiffusion/modelchex050000.pt"

        if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
            model = DDPM(self.config)
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.load(url, map_location=self.device)
            learn_sigma = False
            print("Original diffusion Model loaded.")
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "Retinal_Fundus", "Chexpert"]:
            model = i_DDPM(self.config.data.dataset)
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.load(url)
            learn_sigma = True
            print("Improved diffusion Model loaded.")
        else:
            print('Not implemented dataset')
            raise ValueError
        model.load_state_dict(init_ckpt)
        model.to(self.device)
        model = torch.nn.DataParallel(model)

        # ----------- Optimizer and Scheduler -----------#
        print(f"Setting optimizer with lr={self.args.lr_clip_finetune}")
        optim_ft = torch.optim.Adam(model.parameters(), weight_decay=0, lr=self.args.lr_clip_finetune)
        init_opt_ckpt = optim_ft.state_dict()
        scheduler_ft = torch.optim.lr_scheduler.StepLR(optim_ft, step_size=1, gamma=self.args.sch_gamma)
        init_sch_ckpt = scheduler_ft.state_dict()

        # ----------- Loss -----------#
        print("Loading losses")
        if(self.config.data.dataset == "Chexpert"):
            if(self.args.version == "counterfactual"):
                clip_loss_func = CLIPLoss(
                    self.device,
                    lambda_direction=0,
                    our_lambda_direction=1,
                    lambda_patch=0,
                    lambda_global=0,
                    lambda_manifold=0,
                    lambda_texture=0,
                    clip_model=self.args.clip_model_name,
                    grayscale=1,
                    model_embedding=self.args.model_embedding)
                id_loss_func = id_loss.IDLoss().to(self.device).eval()
            if(self.args.version == "standard"):
                clip_loss_func = CLIPLoss(
                    self.device,
                    lambda_direction=1,
                    our_lambda_direction=0,
                    lambda_patch=0,
                    lambda_global=0,
                    lambda_manifold=0,
                    lambda_texture=0,
                    clip_model=self.args.clip_model_name,
                    grayscale=1,
                    model_embedding=self.args.model_embedding)
                id_loss_func = id_loss.IDLoss().to(self.device).eval()
        else:
            if(self.args.version == "counterfactual"):
                clip_loss_func = CLIPLoss(
                    self.device,
                    lambda_direction=0,
                    our_lambda_direction=1,
                    lambda_patch=0,
                    lambda_global=0,
                    lambda_manifold=0,
                    lambda_texture=0,
                    clip_model=self.args.clip_model_name,
                    grayscale=0,
                    model_embedding=self.args.model_embedding)
                id_loss_func = id_loss.IDLoss().to(self.device).eval()
            if(self.args.version == "standard"):
                clip_loss_func = CLIPLoss(
                    self.device,
                    lambda_direction=1,
                    our_lambda_direction=0,
                    lambda_patch=0,
                    lambda_global=0,
                    lambda_manifold=0,
                    lambda_texture=0,
                    clip_model=self.args.clip_model_name,
                    grayscale=0,
                    model_embedding=self.args.model_embedding)
                id_loss_func = id_loss.IDLoss().to(self.device).eval()

        ###### GET COUNTERFACTUAL DATASET #############################
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
        from PIL import Image

        class CustomImageDataset(Dataset):
            def __init__(self, data_folder, transform=None):
                self.data_folder = data_folder
                self.transform = transform

                self.data_files = sorted(os.listdir(data_folder))

            def __len__(self):
                return len(self.data_files)

            def __getitem__(self, idx):
                file_path = os.path.join(self.data_folder, self.data_files[idx])

                image = Image.open(file_path)

                if self.transform:
                    image = self.transform(image)

                return image

        if(self.args.dataset_path == "Chexpert" or self.args.dataset_path == "ChexpertAtel"):
            transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))
            ])
        else:
            transform = transforms.Compose([
                transforms.Lambda(lambda img: img.convert('RGB')),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        counterfactual_array = []

        if(self.args.version == "counterfactual"):
            print("Computing counterfactual")
    
            counterfactual_array = []
            
            data_folder = f"../drive/MyDrive/CLIPDiffusion/{self.args.dataset_path}/raw_counterfactual"
    
            dataset = CustomImageDataset(data_folder, transform=transform)
    
            batch_size = 1
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
            for images in dataloader:
                counterfactual_array.append(images)
        
        ############################################################

        # ----------- Precompute Latents -----------#
        print("Prepare identity latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        n = self.args.bs_train
        img_lat_pairs_dic = {}
        for mode in ['train', 'test']:
            img_lat_pairs = []
            pairs_path = os.path.join(f'../drive/MyDrive/CLIPDiffusion/{self.args.dataset_path}/',
                                      f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            print(pairs_path)
            if os.path.exists(pairs_path):
                print(f'{mode} pairs exists')
                img_lat_pairs_dic[mode] = torch.load(pairs_path)
                for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs_dic[mode]):
                    tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))
                    tvu.save_image((x_id + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                  f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                    if step == self.args.n_precomp_img - 1:
                        break
                continue
            else:
                train_dataset, test_dataset = get_dataset(self.config.data.dataset, DATASET_PATHS, self.config)
                loader_dic = get_dataloader(train_dataset, test_dataset, bs_train=self.args.bs_train,
                                            num_workers=self.config.data.num_workers)
                loader = loader_dic[mode]

            for step, img in enumerate(loader):
                x0 = img.to(self.config.device)
                tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))

                x = x0.clone()
                model.eval()
                with torch.no_grad():
                    with tqdm(total=len(seq_inv), desc=f"Inversion process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_prev, models=model,
                                               logvars=self.logvar,
                                               sampling_type='ddim',
                                               b=self.betas,
                                               eta=0,
                                               learn_sigma=learn_sigma)

                            progress_bar.update(1)
                    x_lat = x.clone()
                    tvu.save_image((x_lat + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                   f'{mode}_{step}_1_lat_ninv{self.args.n_inv_step}.png'))

                    with tqdm(total=len(seq_inv), desc=f"Generative process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_next = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_next, models=model,
                                               logvars=self.logvar,
                                               sampling_type=self.args.sample_type,
                                               b=self.betas,
                                               learn_sigma=learn_sigma)
                            progress_bar.update(1)

                    img_lat_pairs.append([x0, x.detach().clone(), x_lat.detach().clone()])
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                if step == self.args.n_precomp_img - 1:
                    break

            img_lat_pairs_dic[mode] = img_lat_pairs
            pairs_path = os.path.join('precomputed/',
                                      f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            torch.save(img_lat_pairs, pairs_path)

        # ----------- Finetune Diffusion Models -----------#
        print("Start finetuning")
        print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}")
        if self.args.n_train_step != 0:
            seq_train = np.linspace(0, 1, self.args.n_train_step) * self.args.t_0
            seq_train = [int(s) for s in list(seq_train)]
            print('Uniform skip type')
        else:
            seq_train = list(range(self.args.t_0))
            print('No skip')
        seq_train_next = [-1] + list(seq_train[:-1])

        seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
        seq_test = [int(s) for s in list(seq_test)]
        seq_test_next = [-1] + list(seq_test[:-1])

        for src_txt, trg_txt in zip(self.src_txts, self.trg_txts):
            print(f"CHANGE {src_txt} TO {trg_txt}")
            model.module.load_state_dict(init_ckpt)
            optim_ft.load_state_dict(init_opt_ckpt)
            scheduler_ft.load_state_dict(init_sch_ckpt)
            clip_loss_func.target_direction = None

            # ----------- Train -----------#
            for it_out in range(self.args.n_iter):
                exp_id = os.path.split(self.args.exp)[-1]
                #save_name = f'../../../drive/MyDrive/CLIPDiffusion/{self.args.dataset_path}/checkpoints/{exp_id}_{trg_txt.replace(" ", "_")}-{it_out}.pth'
                save_name = f'../drive/MyDrive/CLIPDiffusion/{self.args.dataset_path}/checkpoints/{self.args.dataset_path}_weights_{it_out}.pth'
                if self.args.do_train:
                    if os.path.exists(save_name):
                        print(f'{save_name} already exists.')
                        model.module.load_state_dict(torch.load(save_name))
                        continue
                    else:
                        for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs_dic['train']):
                            model.train()
                            time_in_start = time.time()

                            optim_ft.zero_grad()
                            x = x_lat.clone()

                            with tqdm(total=len(seq_train), desc=f"CLIP iteration") as progress_bar:
                                for t_it, (i, j) in enumerate(zip(reversed(seq_train), reversed(seq_train_next))):
                                    t = (torch.ones(n) * i).to(self.device)
                                    t_next = (torch.ones(n) * j).to(self.device)

                                    x = denoising_step(x, t=t, t_next=t_next, models=model,
                                                       logvars=self.logvar,
                                                       sampling_type=self.args.sample_type,
                                                       b=self.betas,
                                                       eta=self.args.eta,
                                                       learn_sigma=learn_sigma)

                                    progress_bar.update(1)

                            ##### new loss ##########
                            if(self.args.version == "counterfactual"):
                                #loss_clip = (2 - clip_loss_func(x0, src_txt, x, trg_txt)) / 2
                                #loss_clip = (2 - clip_loss_func(x0, x)) / 2
                                counterfactual_array[step] = counterfactual_array[step].to('cuda')
                                loss_clip = (2 - clip_loss_func(counterfactual_array[step], x)) / 2
                                loss_clip = -torch.log(loss_clip)
                                if(self.config.data.dataset != "Chexpert"):
                                    loss_id = torch.mean(id_loss_func(counterfactual_array[step], x))
                                loss_l1 = nn.L1Loss()(counterfactual_array[step], x)
                                if(self.config.data.dataset != "Chexpert"):
                                    loss = self.args.clip_loss_w * loss_clip + self.args.id_loss_w * loss_id + self.args.l1_loss_w * loss_l1
                                else:
                                    loss = self.args.clip_loss_w * loss_clip + self.args.l1_loss_w * loss_l1
                                loss.backward()
                            #######################
                            
                            #### old loss ######
                            if(self.args.version =="standard"):
                                loss_clip = (2 - clip_loss_func(x0, src_txt, x, trg_txt)) / 2
                                loss_clip = -torch.log(loss_clip)
                                loss_id = torch.mean(id_loss_func(x0, x))
                                loss_l1 = nn.L1Loss()(x0, x)
                                loss = self.args.clip_loss_w * loss_clip + self.args.id_loss_w * loss_id + self.args.l1_loss_w * loss_l1
                                loss.backward()
                            ###################

                            optim_ft.step()
                            if(self.config.data.dataset != "Chexpert"):
                                print(f"CLIP {step}-{it_out}: loss_id: {loss_id:.3f}, loss_clip: {loss_clip:.3f}")
                            else:
                                print(f"CLIP {step}-{it_out}: loss_clip: {loss_clip:.3f}")

                            if self.args.save_train_image:
                                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                           f'train_{step}_2_clip_{trg_txt.replace(" ", "_")}_{it_out}_ngen{self.args.n_train_step}.png'))
                                if(self.args.version == "counterfactual" and it_out==0):
                                  tvu.save_image((counterfactual_array[step] + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                           f'counterfactual_{step}_2_clip_{trg_txt.replace(" ", "_")}_{it_out}_ngen{self.args.n_train_step}.png'))
                            time_in_end = time.time()
                            print(f"Training for 1 image takes {time_in_end - time_in_start:.4f}s")
                            if step == self.args.n_train_img - 1:
                                break
                                
                        if isinstance(model, nn.DataParallel):
                            torch.save(model.module.state_dict(), save_name)
                        else:
                            torch.save(model.state_dict(), save_name)
                        print(f'Model {save_name} is saved.')
                        scheduler_ft.step()

                # ----------- Eval -----------#
                if self.args.do_test:
                    if not self.args.do_train:
                        print("loading following pth file:")
                        print(f"../drive/MyDrive/CLIPDiffusion/{self.args.dataset_path}/checkpoints/{self.args.dataset_path}_weight_{self.args.weight_epoch}.pth")
                        model.module.load_state_dict(torch.load(f"../drive/MyDrive/CLIPDiffusion/{self.args.dataset_path}/checkpoints/{self.args.dataset_path}_weights_{self.args.weight_epoch}.pth"))

                    model.eval()
                    img_lat_pairs = img_lat_pairs_dic[mode]
                    for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs):
                        with torch.no_grad():
                            x = x_lat
                            with tqdm(total=len(seq_test), desc=f"Eval iteration") as progress_bar:
                                for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                                    t = (torch.ones(n) * i).to(self.device)
                                    t_next = (torch.ones(n) * j).to(self.device)

                                    x = denoising_step(x, t=t, t_next=t_next, models=model,
                                                       logvars=self.logvar,
                                                       sampling_type=self.args.sample_type,
                                                       b=self.betas,
                                                       eta=self.args.eta,
                                                       learn_sigma=learn_sigma)

                                    progress_bar.update(1)

                            print(f"Eval {step}-{it_out}")
                            tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                       f'{mode}_{step}_2_clip_{trg_txt.replace(" ", "_")}_{it_out}_ngen{self.args.n_test_step}.png'))
                            if(self.args.save_test_drive and (not self.args.do_train)):
                                tvu.save_image((x + 1) * 0.5,f'../drive/MyDrive/CLIPDiffusion/{self.args.dataset_path}/OurGeneratedImages/class/image{step}.png')
                            if step == self.args.n_test_img - 1:
                                break

    def clip_finetune_eff(self):
        print(self.args.exp)
        print(f'   {self.src_txts}')
        print(f'-> {self.trg_txts}')

        # ----------- Model -----------#
        """if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset == "CelebA_HQ":
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
            pass
        else:
            raise ValueError
        """
        url = "../drive/MyDrive/celeba_hq.ckpt"

        if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
            model = DDPM(self.config)
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.load(url, map_location=self.device)
            learn_sigma = False
            print("Original diffusion Model loaded.")
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
            model = i_DDPM(self.config.data.dataset)
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            learn_sigma = True
            print("Improved diffusion Model loaded.")
        else:
            print('Not implemented dataset')
            raise ValueError
        model.load_state_dict(init_ckpt)
        model.to(self.device)
        model = torch.nn.DataParallel(model)

        # ----------- Optimizer and Scheduler -----------#
        print(f"Setting optimizer with lr={self.args.lr_clip_finetune}")
        optim_ft = torch.optim.Adam(model.parameters(), weight_decay=0, lr=self.args.lr_clip_finetune)
        # optim_ft = torch.optim.SGD(model.parameters(), weight_decay=0, lr=self.args.lr_clip_finetune)#, momentum=0.9)
        init_opt_ckpt = optim_ft.state_dict()
        scheduler_ft = torch.optim.lr_scheduler.StepLR(optim_ft, step_size=1, gamma=self.args.sch_gamma)
        init_sch_ckpt = scheduler_ft.state_dict()

        # ----------- Loss -----------#
        print("Loading losses")
        clip_loss_func = CLIPLoss(
            self.device,
            lambda_direction=1,
            lambda_patch=0,
            lambda_global=0,
            lambda_manifold=0,
            lambda_texture=0,
            clip_model=self.args.clip_model_name)
        if self.config.data.dataset == "CelebA_HQ":
            id_loss_func = id_loss.IDLoss().to(self.device).eval()
        else:
            id_loss_func = None

        # ----------- Precompute Latents -----------#
        print("Prepare identity latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        n = self.args.bs_train
        img_lat_pairs_dic = {}

        for mode in ['train', 'test']:
            img_lat_pairs = []
            if self.args.edit_attr in ['female', 'male']:
                self.config.data.dataset = 'GENDER'
                self.config.data.category = 'GENDER'
                if self.args.edit_attr == 'female':
                    pairs_path = os.path.join('precomputed/',
                                              f'{self.config.data.category}_male_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
                else:
                    pairs_path = os.path.join('precomputed/',
                                              f'{self.config.data.category}_female_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')

            elif self.config.data.dataset == "IMAGENET":
                if self.args.target_class_num is not None:
                    pairs_path = os.path.join('precomputed/',
                                              f'{self.config.data.category}_{IMAGENET_DIC[str(self.args.target_class_num)][1]}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
                else:
                    pairs_path = os.path.join('precomputed/',
                                              f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')

            else:
                pairs_path = os.path.join('../drive/MyDrive/',
                                          f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            print(pairs_path)
            if os.path.exists(pairs_path):
                print(f'{mode} pairs exists')
                img_lat_pairs_dic[mode] = torch.load(pairs_path, map_location=torch.device('cpu'))
                for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs_dic[mode]):
                    tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))
                    tvu.save_image((x_id + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                  f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                    if step == self.args.n_precomp_img - 1:
                        break
                continue
            else:
                if self.args.edit_attr == 'female':
                    train_dataset, test_dataset = get_dataset(self.config.data.dataset, DATASET_PATHS, self.config,
                                                              gender='male')
                elif self.args.edit_attr == 'male':
                    train_dataset, test_dataset = get_dataset(self.config.data.dataset, DATASET_PATHS, self.config,
                                                              gender='female')
                else:
                    train_dataset, test_dataset = get_dataset(self.config.data.dataset, DATASET_PATHS, self.config,
                                                              target_class_num=self.args.target_class_num)

                loader_dic = get_dataloader(train_dataset, test_dataset, bs_train=self.args.bs_train,
                                            num_workers=self.config.data.num_workers)
                loader = loader_dic[mode]

            for step, img in enumerate(loader):
                x0 = img.to(self.config.device)
                tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))

                x = x0.clone()
                model.eval()
                time_s = time.time()
                with torch.no_grad():
                    with tqdm(total=len(seq_inv), desc=f"Inversion process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_prev, models=model,
                                               logvars=self.logvar,
                                               sampling_type='ddim',
                                               b=self.betas,
                                               eta=0,
                                               learn_sigma=learn_sigma)

                            progress_bar.update(1)
                    time_e = time.time()
                    print(f'{time_e - time_s} seconds')
                    x_lat = x.clone()
                    tvu.save_image((x_lat + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                   f'{mode}_{step}_1_lat_ninv{self.args.n_inv_step}.png'))

                    with tqdm(total=len(seq_inv), desc=f"Generative process {mode} {step}") as progress_bar:
                        time_s = time.time()
                        for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_next = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_next, models=model,
                                               logvars=self.logvar,
                                               sampling_type=self.args.sample_type,
                                               b=self.betas,
                                               learn_sigma=learn_sigma)
                            progress_bar.update(1)
                        time_e = time.time()
                        print(f'{time_e - time_s} seconds')

                    img_lat_pairs.append([x0, x.detach().clone(), x_lat.detach().clone()])
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                if step == self.args.n_precomp_img - 1:
                    break

            img_lat_pairs_dic[mode] = img_lat_pairs
            # pairs_path = os.path.join('precomputed/',
            #                           f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            torch.save(img_lat_pairs, pairs_path)

        # ----------- Finetune Diffusion Models -----------#
        print("Start finetuning")
        print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}")
        if self.args.n_train_step != 0:
            seq_train = np.linspace(0, 1, self.args.n_train_step) * self.args.t_0
            seq_train = [int(s) for s in list(seq_train)]
            print('Uniform skip type')
        else:
            seq_train = list(range(self.args.t_0))
            print('No skip')
        seq_train_next = [-1] + list(seq_train[:-1])

        seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
        seq_test = [int(s) for s in list(seq_test)]
        seq_test_next = [-1] + list(seq_test[:-1])

        for src_txt, trg_txt in zip(self.src_txts, self.trg_txts):
            print(f"CHANGE {src_txt} TO {trg_txt}")
            model.module.load_state_dict(init_ckpt)
            optim_ft.load_state_dict(init_opt_ckpt)
            scheduler_ft.load_state_dict(init_sch_ckpt)
            clip_loss_func.target_direction = None

            # ----------- Train -----------#
            for it_out in range(self.args.n_iter):
                exp_id = os.path.split(self.args.exp)[-1]
                save_name = f'checkpoint/{exp_id}_{trg_txt.replace(" ", "_")}-{it_out}.pth'
                if self.args.do_train:
                    if os.path.exists(save_name):
                        print(f'{save_name} already exists.')
                        model.module.load_state_dict(torch.load(save_name))
                        continue
                    else:
                        for step, (x0, _, x_lat) in enumerate(img_lat_pairs_dic['train']):
                            model.train()
                            time_in_start = time.time()

                            optim_ft.zero_grad()
                            x = x_lat.clone().to(self.device)
                            x0 = x0.to(self.device)
                            with tqdm(total=len(seq_train), desc=f"CLIP iteration") as progress_bar:
                                for t_it, (i, j) in enumerate(zip(reversed(seq_train), reversed(seq_train_next))):
                                    t = (torch.ones(n) * i).to(self.device)
                                    t_next = (torch.ones(n) * j).to(self.device)

                                    x, x0_t = denoising_step(x, t=t, t_next=t_next, models=model,
                                                             logvars=self.logvar,
                                                             sampling_type=self.args.sample_type,
                                                             b=self.betas,
                                                             eta=self.args.eta,
                                                             learn_sigma=learn_sigma,
                                                             out_x0_t=True)

                                    progress_bar.update(1)
                                    x = x.detach().clone()

                                    loss_clip = -torch.log((2 - clip_loss_func(x0, src_txt, x0_t, trg_txt)) / 2)
                                    loss_l1 = nn.L1Loss()(x0, x0_t)
                                    loss = self.args.clip_loss_w * loss_clip + self.args.l1_loss_w * loss_l1
                                    if self.config.data.dataset == "CelebA_HQ":
                                        loss_id = torch.mean(id_loss_func(x0, x))
                                        loss += self.args.id_loss_w * loss_id
                                    loss.backward()

                                    optim_ft.step()
                                    for p in model.module.parameters():
                                        p.grad = None
                                    print(f"CLIP {step}-{it_out}: loss_clip: {loss_clip:.3f}")
                                    # break

                            if self.args.save_train_image:
                                tvu.save_image((x0_t + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                              f'train_{step}_2_clip_{trg_txt.replace(" ", "_")}_{it_out}_ngen{self.args.n_train_step}.png'))
                            time_in_end = time.time()
                            print(f"Training for 1 image takes {time_in_end - time_in_start:.4f}s")
                            if step == self.args.n_train_img - 1:
                                break

                        if isinstance(model, nn.DataParallel):
                            torch.save(model.module.state_dict(), save_name)
                        else:
                            torch.save(model.state_dict(), save_name)
                        print(f'Model {save_name} is saved.')
                        scheduler_ft.step()

                # ----------- Eval -----------#
                if self.args.do_test:
                    if not self.args.do_train:
                        print(save_name)
                        model.module.load_state_dict(torch.load())

                    model.eval()
                    img_lat_pairs = img_lat_pairs_dic[mode]
                    for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs):
                        with torch.no_grad():
                            x = x_lat.clone().to(self.device)
                            x0 = x0.to(self.device)
                            with tqdm(total=len(seq_test), desc=f"Eval iteration") as progress_bar:
                                for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                                    t = (torch.ones(n) * i).to(self.device)
                                    t_next = (torch.ones(n) * j).to(self.device)

                                    x = denoising_step(x, t=t, t_next=t_next, models=model,
                                                       logvars=self.logvar,
                                                       sampling_type=self.args.sample_type,
                                                       b=self.betas,
                                                       eta=self.args.eta,
                                                       learn_sigma=learn_sigma)

                                    progress_bar.update(1)

                            print(f"Eval {step}-{it_out}")
                            tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                       f'{mode}_{step}_2_clip_{trg_txt.replace(" ", "_")}_{it_out}_ngen{self.args.n_test_step}.png'))
                            tvu.save_image((x + 1) * 0.5,f'../drive/MyDrive/CLIPDiffusion/Mustache/OurGeneratedImages/image{step}.png')
                            if step == self.args.n_test_img - 1:
                                break

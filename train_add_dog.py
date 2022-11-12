from tokenize import String
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import config
import os
from dataset import MyDataset, MyTestDataset
from generator_model import Generator
from discriminator_model import Discriminator
from VGGPytorch import VGGNet
from torch.utils.data import DataLoader
from tqdm import tqdm
from whitebox_utils import save_val_examples, load_checkpoint, save_checkpoint, save_training_images
from losses import VariationLoss
from structure_extractor import SuperPixel
from texture_extractor import ColorShift
from texture_extractor import DoG
from surface_extractor import GuidedFilter
import itertools
import argparse
from torch.utils.tensorboard import SummaryWriter

sw = SummaryWriter("good_model_opt_dog")



def parser():
    parser = argparse.ArgumentParser(description="train.py: Model training script of White-box Cartoonization. Pretraining included.")
    parser.add_argument("--name",default=config.PROJECT_NAME,
                        help="project name. default name:"+f"{config.PROJECT_NAME}")
    parser.add_argument("--batch_size", type=int, default= config.BATCH_SIZE,
                        help="batch size. default batch size:"+f"{config.BATCH_SIZE}")
    parser.add_argument("--num_workers", type=int, default= config.NUM_WORKERS,
                        help="number of workers. default number of workers:"+f"{config.NUM_WORKERS}")
    parser.add_argument("--save_model_freq", type=int, default= config.SAVE_MODEL_FREQ,
                        help="saving model each N epochs. default value:"+f"{config.SAVE_MODEL_FREQ}")
    parser.add_argument("--save_img_freq", type=int, default= config.SAVE_IMG_FREQ,
                        help="saving training image each N steps. default value:"+f"{config.SAVE_IMG_FREQ}")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS,
                        help=" default value:"+f"{config.NUM_EPOCHS}")
    parser.add_argument("--lambda_surface", type=float, default= config.LAMBDA_SURFACE,
                        help="lambda value of surface rep. default:"+f"{config.LAMBDA_SURFACE}")
    parser.add_argument("--lambda_texture", type=float, default= config.LAMBDA_TEXTURE,
                        help="lambda value of texture rep. default:"+f"{config.LAMBDA_TEXTURE}")
    parser.add_argument("--lambda_structure", type=float, default= config.LAMBDA_STRUCTURE,
                        help="lambda value of structure rep. default:"+f"{config.LAMBDA_STRUCTURE}")
    parser.add_argument("--lambda_content", type=float, default= config.LAMBDA_CONTENT,
                        help="lambda value of content loss. default:"+f"{config.LAMBDA_CONTENT}")
    parser.add_argument("--lambda_variation", type=float, default= config.LAMBDA_VARIATION,
                        help="lambda value of variation loss. default:"+f"{config.LAMBDA_VARIATION}")
    parser.add_argument("--lambda_dog", type=float, default= config.LAMBDA_DOG,
                        help="lambda value of dog loss. default:"+f"{config.LAMBDA_DOG}")
    parser.add_argument("--lambda_like", type=float, default= config.LAMBDA_LIKE,
                        help="lambda value of like loss. default:"+f"{config.LAMBDA_LIKE}")
    parser.add_argument("--device", default= config.DEVICE,
                        help="DEVICE. default:"+f"{config.DEVICE}")
    return parser.parse_args()

def update_config(args, verbose=True):
    config.PROJECT_NAME = args.name
    config.BATCH_SIZE = args.batch_size
    config.NUM_WORKERS = args.num_workers
    config.NUM_EPOCHS = args.epochs
    config.LAMBDA_SURFACE = args.lambda_surface
    config.LAMBDA_TEXTURE = args.lambda_texture
    config.LAMBDA_STRUCTURE = args.lambda_structure
    config.LAMBDA_CONTENT = args.lambda_content
    config.LAMBDA_VARIATION = args.lambda_variation
    config.LAMBDA_DOG = args.lambda_dog
    config.LAMBDA_LIKE = args.lambda_like
    config.SAVE_MODEL_FREQ = args.save_model_freq
    config.SAVE_IMG_FREQ = args.save_img_freq
    config.DEVICE = args.device
    
    config.CHECKPOINT_FOLDER = os.path.join("checkpoints", config.PROJECT_NAME)
    config.RESULT_TRAIN_DIR = os.path.join("results", config.PROJECT_NAME, "train")
    config.RESULT_VAL_DIR = os.path.join("results", config.PROJECT_NAME, "val")
    config.RESULT_TEST_DIR = os.path.join("results", config.PROJECT_NAME, "test")

    if(verbose):
        print("="*80)
        print("=> Input config:")
        print("Using Device: " + config.DEVICE)
        print(f'PROJECT_NAME: {config.PROJECT_NAME}')
        print(f'BATCH_SIZE: {config.BATCH_SIZE}')
        print(f'NUM_WORKERS: {config.NUM_WORKERS}')
        print(f'NUM_EPOCHS: {config.NUM_EPOCHS}')
        print(f'LAMBDA_SURFACE: {config.LAMBDA_SURFACE}')
        print(f'LAMBDA_TEXTURE: {config.LAMBDA_TEXTURE}')
        print(f'LAMBDA_STRUCTURE: {config.LAMBDA_STRUCTURE}')
        print(f'LAMBDA_CONTENT: {config.LAMBDA_CONTENT}')
        print(f'LAMBDA_VARIATION: {config.LAMBDA_VARIATION}')
        print(f'LAMBDA_LIKE: {config.LAMBDA_LIKE}')
        print(f'SAVE_MODEL_FREQ: {config.SAVE_MODEL_FREQ}')
        print(f'SAVE_IMG_FREQ: {config.SAVE_IMG_FREQ}')
        print("="*80)

def initialization_phase(gen, loader, opt_gen, l1_loss, VGG, pretrain_epochs):
    for epoch in range(pretrain_epochs):
        loop = tqdm(loader, leave=True)
        losses = []

        for idx, (sample_photo, _) in enumerate(loop):
            sample_photo = sample_photo.to(config.DEVICE)
            reconstructed = gen(sample_photo)

            sample_photo_feature = VGG(sample_photo)
            reconstructed_feature = VGG(reconstructed)
            reconstruction_loss = l1_loss(reconstructed_feature, sample_photo_feature.detach()) * 255
            
            losses.append(reconstruction_loss.item())

            opt_gen.zero_grad()
            
            reconstruction_loss.backward()
            opt_gen.step()

            loop.set_postfix(epoch=epoch)

        print('[%d/%d] - Recon loss: %.8f' % ((epoch + 1), pretrain_epochs, torch.mean(torch.FloatTensor(losses))))
        
        save_training_images(torch.cat((sample_photo*0.5+0.5,reconstructed*0.5+0.5), axis=3),
                                                epoch=epoch, step=0, dest_folder=config.RESULT_TRAIN_DIR, suffix_filename="initial_io")
    
    if config.SAVE_MODEL:
        save_checkpoint(gen, opt_gen, 'i', folder=config.CHECKPOINT_FOLDER, filename=config.CHECKPOINT_GEN)
    
def train_fn(kernel_sigma, kappa, disc, gen, loader, opt_disc, opt_gen, l1_loss, mse, VGG, var_loss, val_loader, clip_label=False):
    nonzero_soft_weight_threshold = 1e-3 # kyoma: should be set in config, but default in CcGAN is 1e-3
    step = 0
    
    for epoch in range(config.NUM_EPOCHS):
        loop = tqdm(loader, leave=True)

        # Training
        for idx, (sample_photo, sample_cartoon, sample_labels) in enumerate(loop):
            sample_photo = sample_photo.to(config.DEVICE)
            sample_cartoon = sample_cartoon.to(config.DEVICE)
            sample_labels = sample_labels.to(config.DEVICE)

            unique_train_labels = np.sort(np.array(list(set(sample_labels))))
            batch_target_labels_in_dataset = np.random.choice(unique_train_labels, size=config.BATCH_SIZE, replace=True)
            batch_epsilons = np.random.normal(0, kernel_sigma, config.BATCH_SIZE)
            batch_target_labels_with_epsilon = batch_target_labels_in_dataset + batch_epsilons
            if clip_label:
                batch_target_labels_with_epsilon = np.clip(batch_target_labels_with_epsilon, 0.0, 1.0)

            batch_target_labels = batch_target_labels_with_epsilon[0:config.BATCH_SIZE]

            ## find index of real images with labels in the vicinity of batch_target_labels
            ## generate labels for fake image generation; these labels are also in the vicinity of batch_target_labels
            batch_real_indx = np.zeros(config.BATCH_SIZE, dtype=int) #index of images in the datata; the labels of these images are in the vicinity
            batch_fake_labels = np.zeros(config.BATCH_SIZE)

            for j in range(config.BATCH_SIZE):
                ## index for real images
                # reverse the weight function for SVDL
                indx_real_in_vicinity = np.where((sample_labels-batch_target_labels[j])**2 <= -np.log(nonzero_soft_weight_threshold)/kappa)[0]

                ## if the max gap between two consecutive ordered unique labels is large, it is possible that len(indx_real_in_vicinity)<1
                while len(indx_real_in_vicinity) < 1:
                    batch_epsilons_j = np.random.normal(0, kernel_sigma, 1)
                    batch_target_labels[j] = batch_target_labels_in_dataset[j] + batch_epsilons_j
                    if clip_label:
                        batch_target_labels = np.clip(batch_target_labels, 0.0, 1.0)
                    ## index for real images
                    # reverse the weight function for SVDL
                    indx_real_in_vicinity = np.where((sample_labels-batch_target_labels[j])**2 <= -np.log(nonzero_soft_weight_threshold)/kappa)[0]

                assert len(indx_real_in_vicinity)>=1

                batch_real_indx[j] = np.random.choice(indx_real_in_vicinity, size=1)[0]
                ## labels for fake images generation
                lb = batch_target_labels[j] - np.sqrt(-np.log(nonzero_soft_weight_threshold)/kappa)
                ub = batch_target_labels[j] + np.sqrt(-np.log(nonzero_soft_weight_threshold)/kappa)
                lb = max(0.0, lb); ub = min(ub, 1.0)
                assert lb<=ub
                assert lb>=0 and ub>=0
                assert lb<=1 and ub<=1
                batch_fake_labels[j] = np.random.uniform(lb, ub, size=1)[0]
            
            ## draw the real image batch from the training set


            # Train Discriminator
            fake_cartoon = gen(sample_photo)
            # output_photo = extract_surface.process(sample_photo, fake_cartoon, r=1)


            opt_disc.zero_grad()
            d_loss_total.backward()
            opt_disc.step()
            
            #===============================================================================

            # Train Generator
            fake_cartoon = gen(sample_photo)
            output_photo = extract_surface.process(sample_photo, fake_cartoon, r=1)

            # Guided Filter
            blur_fake = extract_surface.process(output_photo, output_photo, r=5, eps=2e-1)
            D_blur_fake = disc_surface(blur_fake)
            g_loss_surface = config.LAMBDA_SURFACE * mse(D_blur_fake, torch.ones_like(D_blur_fake))

            # Color Shift
            gray_fake, = extract_texture.process(output_photo)
            D_gray_fake = disc_texture(gray_fake)
            g_loss_texture = config.LAMBDA_TEXTURE * mse(D_gray_fake, torch.ones_like(D_gray_fake))

            # Dog
            dog_fake, = dog.process(output_photo*0.5 + 0.5)
            D_dog_fake = disc_dog(dog_fake)
            g_loss_dog = config.LAMBDA_DOG * mse(D_dog_fake, torch.ones_like(D_dog_fake))

            # SuperPixel
            input_superpixel = extract_structure.process(output_photo.detach())
            vgg_output = VGG(output_photo)
            _, c, h, w = vgg_output.shape
            vgg_superpixel = VGG(input_superpixel)
            superpixel_loss = config.LAMBDA_STRUCTURE * l1_loss(vgg_superpixel, vgg_output)*255 / (c*h*w)
            #^ Original author used CaffeVGG model which took (0-255)BGR images as input,
            # while we used PyTorch model which takes (0-1)BGB images as input. Therefore we multply the l1 with 255.

            # Content Loss
            vgg_photo = VGG(sample_photo)
            content_loss = config.LAMBDA_CONTENT * l1_loss(vgg_photo, vgg_output)*255 / (c*h*w)
            #^ Original author used CaffeVGG model which took (0-255)BGR images as input,
            # while we used PyTorchVGG model which takes (0-1)BGB images as input. Therefore we multply the l1 with 255.

            # Like loss
            D_like_fake = disc_like(output_photo)
            like_loss = config.LAMBDA_LIKE * mse(D_like_fake, torch.ones_like(D_like_fake))

            # Variation Loss
            tv_loss = config.LAMBDA_VARIATION * var_loss(fake_cartoon)
            
            #NOTE Equation 6 in the paper
            g_loss_total = g_loss_surface + g_loss_texture + superpixel_loss + content_loss + tv_loss + g_loss_dog + like_loss 

            opt_gen.zero_grad()
            g_loss_total.backward()
            opt_gen.step()

            #===============================================================================
            if step % config.SAVE_IMG_FREQ == 0:
                save_training_images(torch.cat((blur_fake*0.5+0.5,gray_fake*0.5+0.5,dog_cartoon*0.5+0.5,dog_fake*0.5+0.5), axis=3), epoch=epoch, step=step, dest_folder=config.RESULT_TRAIN_DIR, suffix_filename="photo_rep")
                save_training_images(torch.cat((sample_photo*0.5+0.5,fake_cartoon*0.5+0.5,output_photo*0.5+0.5), axis=3),
                                                epoch=epoch, step=step, dest_folder=config.RESULT_TRAIN_DIR, suffix_filename="io")

                save_val_examples(gen=gen, val_loader=val_loader, 
                                  epoch=epoch, step=step, dest_folder=config.RESULT_VAL_DIR, num_samples=5, concat_image=True, post_processing=True)

                sw.add_scalar("D Surface loss", d_loss_surface.item(), step)
                sw.add_scalar("D Texture loss", d_loss_texture.item(), step)
                sw.add_scalar("D DoG loss", d_loss_dog.item(), step)
                sw.add_scalar("D like loss", d_loss_like.item(), step)

                sw.add_scalar("G Surface loss", g_loss_surface.item(), step)
                sw.add_scalar("G Texture loss", g_loss_texture.item(), step)
                sw.add_scalar("G DoG loss", g_loss_dog.item(), step)
                sw.add_scalar("G Structure loss", superpixel_loss.item(), step)
                sw.add_scalar("G Content loss", content_loss.item(), step)
                sw.add_scalar("G like loss", like_loss.item(), step)
                sw.add_scalar("G Variation loss", tv_loss.item(), step)

                print('[Epoch: %d| Step: %d] - D Surface loss: %.12f' % ((epoch + 1), (step+1), d_loss_surface.item()))
                print('[Epoch: %d| Step: %d] - D Texture loss: %.12f' % ((epoch + 1), (step+1), d_loss_texture.item()))
                print('[Epoch: %d| Step: %d] - D DoG loss: %.12f' % ((epoch + 1), (step+1), d_loss_dog.item()))
                print('[Epoch: %d| Step: %d] - D like loss: %.12f' % ((epoch + 1), (step+1), d_loss_like.item()))
                print("")
                print('[Epoch: %d| Step: %d] - G Surface loss: %.12f' % ((epoch + 1), (step+1), g_loss_surface.item()))
                print('[Epoch: %d| Step: %d] - G Texture loss: %.12f' % ((epoch + 1), (step+1), g_loss_texture.item()))
                print('[Epoch: %d| Step: %d] - G DoG loss: %.12f' % ((epoch + 1), (step+1), g_loss_dog.item()))
                print('[Epoch: %d| Step: %d] - G Structure loss: %.12f' % ((epoch + 1), (step+1), superpixel_loss.item()))
                print('[Epoch: %d| Step: %d] - G Content loss: %.12f' % ((epoch + 1), (step+1), content_loss.item()))
                print('[Epoch: %d| Step: %d] - G like loss: %.12f' % ((epoch + 1), (step+1), like_loss.item()))
                print('[Epoch: %d| Step: %d] - G Variation loss: %.12f' % ((epoch + 1), (step+1), tv_loss.item()))

            step += 1

            loop.set_postfix(step=step, epoch=epoch+1)

        if config.SAVE_MODEL and epoch % config.SAVE_MODEL_FREQ == 0:
            save_checkpoint(gen, opt_gen, epoch, folder=config.CHECKPOINT_FOLDER, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc_texture, opt_disc, epoch, folder=config.CHECKPOINT_FOLDER, filename="texture_" + config.CHECKPOINT_DISC)
            save_checkpoint(disc_surface, opt_disc, epoch, folder=config.CHECKPOINT_FOLDER, filename="surface_" + config.CHECKPOINT_DISC)
            save_checkpoint(disc_dog, opt_disc, epoch, folder=config.CHECKPOINT_FOLDER, filename="dog_" + config.CHECKPOINT_DISC)
            save_checkpoint(disc_like, opt_disc, epoch, folder=config.CHECKPOINT_FOLDER, filename="like_" + config.CHECKPOINT_DISC)

    if config.SAVE_MODEL:
        save_checkpoint(gen, opt_gen, epoch, folder=config.CHECKPOINT_FOLDER, filename="last_"+config.CHECKPOINT_GEN)
        save_checkpoint(disc_texture, opt_disc, epoch, folder=config.CHECKPOINT_FOLDER, filename="last_texture_"+config.CHECKPOINT_DISC)
        save_checkpoint(disc_surface, opt_disc, epoch, folder=config.CHECKPOINT_FOLDER, filename="last_surface_" + config.CHECKPOINT_DISC)
        save_checkpoint(disc_dog, opt_disc, epoch, folder=config.CHECKPOINT_FOLDER, filename="last_dog_" + config.CHECKPOINT_DISC)
        save_checkpoint(disc_like, opt_disc, epoch, folder=config.CHECKPOINT_FOLDER, filename="last_like_" + config.CHECKPOINT_DISC)

def main():
    disc_texture = Discriminator(in_channels=3).to(config.DEVICE)
    disc_surface = Discriminator(in_channels=3).to(config.DEVICE)
    disc_dog = Discriminator(in_channels=3).to(config.DEVICE)
    disc_like = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(img_channels=3).to(config.DEVICE)

    opt_disc = optim.Adam(itertools.chain(disc_surface.parameters(),disc_texture.parameters(),disc_dog.parameters(), disc_like.parameters()), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    VGG19 = VGGNet(in_channels=3, VGGtype="VGG19", init_weights=config.VGG_WEIGHTS, batch_norm=False, feature_mode=True)
    VGG19 = VGG19.to(config.DEVICE)
    VGG19.eval()

    extract_structure = SuperPixel(config.DEVICE, mode='simple')
    extract_texture = ColorShift(config.DEVICE, mode='uniform', image_format='rgb')
    extract_surface = GuidedFilter()
    dog = DoG(config.DEVICE, mode='uniform', image_format='rgb')

    #BCE_Loss = nn.BCELoss()
    L1_Loss = nn.L1Loss()
    MSE_Loss = nn.MSELoss() # went through the author's code and found him using LSGAN, LSGAN should gives better training
    var_loss = VariationLoss(1)
    
    train_dataset = MyDataset(config.TRAIN_PHOTO_DIR, config.TRAIN_CARTOON_DIR)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    val_dataset = MyTestDataset(config.VAL_PHOTO_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS)

    if config.LOAD_MODEL:
        is_gen_loaded = load_checkpoint(
            gen, opt_gen, config.LEARNING_RATE, path=os.path.join(config.CHECKPOINT_FOLDER, config.LOAD_CHECKPOINT_GEN)
        )

    # Initialization Phase
    if not(is_gen_loaded):
        print("="*80)
        print("=> Initialization Phase")
        initialization_phase(gen, train_loader, opt_gen, L1_Loss, VGG19, config.PRETRAIN_EPOCHS)
        print("Finished Initialization Phase")
        print("="*80)

    # Do the training
    print("=> Start Training")
    train_fn(disc_texture, disc_surface, disc_dog, disc_like, gen, train_loader, opt_disc, opt_gen, L1_Loss, MSE_Loss, 
            VGG19, dog, extract_structure, extract_texture, extract_surface, var_loss, val_loader)  
    print("=> Training finished")


if __name__ == "__main__":
    args = parser()
    update_config(args)
    main()
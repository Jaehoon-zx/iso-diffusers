import os
import argparse
import torch
import numpy as np
import pickle
from datasets import load_dataset
import random
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate.state import AcceleratorState
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from tqdm import tqdm
# from stylegan3.dataset import ImageFolderDataset
from dnnlib.util import open_url
import torch_utils

def get_activations(dl, model, batch_size, device, max_samples, dl_include_step=True):
    pred_arr = []
    total_processed = 0

    print('Starting to sample.')
    if dl_include_step:
        for step, batch in tqdm(enumerate(dl)):
            batch = batch["pixel_values"].to(torch.float16)
            # ignore labels
            if isinstance(batch, list):
                batch = batch[0]

            batch = batch.to(device)
            if batch.shape[1] == 1:  # if image is gray scale
                batch = batch.repeat(1, 3, 1, 1)
            elif len(batch.shape) == 3:  # if image is gray scale
                batch = batch.unsqueeze(1).repeat(1, 3, 1, 1)

            with torch.no_grad():
                batch = (batch / 2. + .5).clip(0., 1.)
                batch = (batch * 255.).to(torch.uint8)
                pred = model(batch, return_features=True).unsqueeze(-1).unsqueeze(-1)

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr.append(pred)
            total_processed += pred.shape[0]
            if max_samples is not None and total_processed > max_samples:
                print('Max of %d samples reached.' % max_samples)
                break

        pred_arr = np.concatenate(pred_arr, axis=0)
        if max_samples is not None:
            pred_arr = pred_arr[:max_samples]
    else:
        for batch in dl:
            # ignore labels
            if isinstance(batch, list):
                batch = batch[0]

            batch = batch.to(device)
            if batch.shape[1] == 1:  # if image is gray scale
                batch = batch.repeat(1, 3, 1, 1)
            elif len(batch.shape) == 3:  # if image is gray scale
                batch = batch.unsqueeze(1).repeat(1, 3, 1, 1)
            with torch.no_grad():
                pred = model(batch, return_features=True).unsqueeze(-1).unsqueeze(-1)

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr.append(pred)
            total_processed += pred.shape[0]
            if max_samples is not None and total_processed > max_samples:
                print('Max of %d samples reached.' % max_samples)
                break

        pred_arr = np.concatenate(pred_arr, axis=0)
        if max_samples is not None:
            pred_arr = pred_arr[:max_samples]

    return pred_arr


def main(args):
    pre_dataset = load_dataset(
            args.path,
            None,
            cache_dir=None,
        )
    if not os.path.exists(args.fid_dir):
        os.makedirs(args.fid_dir)
    column_names = pre_dataset[args.split].column_names # "train"

    DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
    }

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.path, None)
    
    image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    
    caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    
    # tokenizer = CLIPTokenizer.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="tokenizer", revision=None
    # )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)

        return examples

    def preprocess_train_ddpm(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]

        return examples


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # dataset = ImageFolderDataset(args.path)
    dataset = pre_dataset[args.split].with_transform(preprocess_train_ddpm)
    # dataset = pre_dataset[args.split].with_transform(preprocess_train)
    # queue = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, pin_memory=True, num_workers=0)
    
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    def collate_fn_ddpm(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        # input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values}
    
    queue = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        collate_fn=collate_fn_ddpm,
        batch_size=args.batch_size,
        num_workers=0,
    )
    with open_url('https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl') as f:
        model = pickle.load(f).to(device)
        model.eval()
    # accelerator_project_config = ProjectConfiguration(project_dir="output/sd2-ffhq512-8", logging_dir="logs")
    # accelerator = Accelerator(
    #     gradient_accumulation_steps=1,
    #     mixed_precision="fp16",
    #     project_config=accelerator_project_config,
    # )
    # queue = accelerator.prepare(queue)
    act = get_activations(queue, model, batch_size=args.batch_size, device=device, max_samples=args.max_samples)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    file_path = os.path.join(args.fid_dir, args.file)
    np.savez(file_path, mu=mu, sigma=sigma)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="mattymchen/celeba-hq")
    parser.add_argument('--pretrained_model_name_or_path', type=str, default="google/ddpm-ema-celebahq-256")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--center_crop', type=bool, default=True)
    parser.add_argument('--random_flip', type=bool, default=True)
    parser.add_argument('--fid_dir', type=str, default='./')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--file', type=str, default="assets/stats/celebahq_256.npz")
    parser.add_argument('--max_samples', type=int, default=None)
    
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    main(args)

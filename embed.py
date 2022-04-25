from facenet_pytorch import MTCNN, InceptionResnetV1
from tqdm import tqdm
from PIL import Image
import argparse
import torch
import os
import json

parser = argparse.ArgumentParser(description='TECLARS Embedding Generator')
parser.add_argument('-i', '--id_path', type=str, default='./data/id.json',
                    help='Path of ID data JSON file')
parser.add_argument('-p', '--img_dir', type=str, default='./data/images',
                    help='Directory of where image data is stored')
parser.add_argument('-o', '--out_path', type=str, default='./data/embeddings.pt',
                    help='Path of ID data JSON file')
parser.add_argument('-d', '--device', type=str, default='auto',
                    help='Device to compute algorithm on')

args = parser.parse_args()

with open(args.id_path, 'r') as f:
    ID = json.load(f)

if args.device == "auto":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device(args.device)
print('Running on device: {}'.format(device))

print("Aligning for faces from data")

mtcnn = MTCNN(image_size=160, device=device)

aligned = []
for idx, person in tqdm(enumerate(ID)):
    fpath = os.path.join(args.img_dir, person['image'])

    with Image.open(fpath) as x:
        x_aligned, prob = mtcnn(x, return_prob=True)

        if x_aligned is not None:
            aligned.append(x_aligned)
        else:
            print(f"Error: No image of {person['first']} {person['last']} found in {person['image']}", end='')
            print("You might wanna crop it and reduce the dimensions to around 700px on the sorter side.")

print("Loading face recognition model")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

print("Computing face embeddings")
aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()
torch.save({'embedding': embeddings}, args.out_path)

print("\n[All done]")

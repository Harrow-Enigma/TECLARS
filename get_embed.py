from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import os
import json

with open('data\id.json', 'r') as f:
    ID = json.load(f)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

print("Aligning for faces from data")

mtcnn = MTCNN(image_size=160, device=device)

aligned = []
for idx, person in enumerate(ID):
    fpath = os.path.join('./data/images', person['image'])

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
torch.save({'embedding': embeddings}, 'data/embeddings.pt')

print("\n[All done]")

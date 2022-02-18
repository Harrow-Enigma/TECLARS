from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torch.nn as nn
from PIL import Image
import cv2

import os
import json
import time
import argparse

def euclidean_distance(out, refs):
    return (out - refs).norm(dim=1)

def cosine_distance(out, refs):
    # matrix multiplication between n x 512 reference embeddings and 512 output embedding =
    # dot products between each of the n reference embeddings and output embedding
    dot_prods = refs @ out
    refs_norm = refs.norm(dim=1)
    norms = out.norm() * refs_norm
    return dot_prods / norms

def find_match(output_embedding, all_embeddings, temperature, mode):
    with torch.no_grad():
        dist_func, arg_func, val_func = {
            'euclidean': (euclidean_distance, torch.argmin, torch.min),
            'cosine': (cosine_distance, torch.argmax, torch.max)
        }[mode]
        softmax = nn.Softmax(dim=0)

        distances = dist_func(output_embedding, all_embeddings)
        probs = softmax(distances * temperature)

    return arg_func(probs), val_func(probs)

def identify_faces(pil_image, probability_threshold=0.4, temperature=2, mode='cosine'):
    faces = mtcnn(pil_image)
    if faces is not None:
        bounding_boxes, _ = mtcnn.detect(pil_image)
        preds = resnet(faces)

        matches = []
        for y, box in zip(preds, bounding_boxes):
            idx, confidence = find_match(y, embeddings, temperature, mode)
            if confidence > probability_threshold:
                entity = ID[idx]
                matches.append((entity, confidence.item(), box))
        
        return matches
    return None

def video():
    print("Starting video capture\n")
    video_capture = cv2.VideoCapture(0)

    registered_students = []
    while True:
        _, frame = video_capture.read()

        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        matches = identify_faces(im, args.threshold, args.temp, args.mode)

        if matches is None:
            print("> No faces found")
        elif len(matches) == 0:
            print("> Faces found but none recognised")
        else:
            for entity, confidence, box in matches:
                bounds = box.astype(int)
                text = f"{entity['first']} {entity['last']}: {round(confidence*100, 2)}%"
                cv2.putText(frame, text, (bounds[0], bounds[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (bounds[0], bounds[1]), (bounds[2], bounds[3]), (0, 0, 255), 2)

                if entity not in registered_students:
                    print(f"{entity['first']} {entity['last']} registered")
                    registered_students.append(entity)

        cv2.imshow('TECLARS Mainframe', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    print()
    video_capture.release()
    cv2.destroyAllWindows()

    time.sleep(2)

    print("\n\nRegistered students:")
    print("\n".join([
        f"{e}) {entity['first']} {entity['last']}"
        for e, entity in enumerate(registered_students)
    ]))

def test():
    print("Beginning testing")
    for filename in os.listdir(args.test_path):
        with Image.open(os.path.join(args.test_path, filename)) as im:
            matches = identify_faces(im, args.threshold, args.temp, args.mode)

            if matches is None:
                print(f"\nNo faces found in {filename}")
            elif len(matches) == 0:
                print(f"\nFaces found but none recognised in {filename}")
            else:
                print(f'\nFaces in {filename}')
                print('\n'.join([
                    f"{entity['first']} {entity['last']}, confidence: {round(confidence*100, 2)}%, bounds: {box.astype(int).tolist()}"
                    for entity, confidence, box in matches
                ]))


parser = argparse.ArgumentParser(description='TECLARS: Team Enigma CMC Lab Auto-Registration System')
parser.set_defaults(which='video')
subparsers = parser.add_subparsers(help='TECLARS subcommands (run without any subcommands to run default system)')

parser.add_argument('--threshold', type=float, default=0.4,
                    help='Probability above which a face will be considered recognised')
parser.add_argument('--temp', type=float, default=2,
                    help='Temperature: higher temperature creates higher probabilities for a recognised face')
parser.add_argument('--mode', type=str, choices=['cosine', 'euclidean'], default='cosine',
                    help='Distance function for evaluating the similarity between face embeddings')

test_parser = subparsers.add_parser('test', help='Test system performance on a set of images in a given directory')
test_parser.set_defaults(which='test')
test_parser.add_argument('--test_path', type=str, default='./data/test',
                         help='Path to directory with test images')

args = parser.parse_args()

if __name__ == "__main__":
    with open('data\id.json', 'r') as f:
        ID = json.load(f)

    bundle = torch.load('data/embeddings.pt')
    embeddings = bundle['embedding']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(image_size=160, keep_all=True, device=device)

    print("Loading face recognition model")
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    if args.which == 'video':
        video()
    elif args.which == 'test':
        test()

    print("\n\n[TECLARS terminated]")

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torch.nn as nn
from PIL import Image
import cv2

import os
import json
import time
import argparse
from datetime import datetime

from counter import IDCounter
from interface import Interface
from vidcap import BufferlessVideoCapture
from localparams import LOCALPARAMS


def euclidean_distance(out, refs):
    return (out - refs).norm(dim=1)

def cosine_distance(out, refs):
    # matrix multiplication between n x 512 reference embeddings and 512 output embedding =
    # dot products between each of the n reference embeddings and output embedding
    dot_prods = refs @ out
    refs_norm = refs.norm(dim=1)
    norms = out.norm() * refs_norm
    return dot_prods / norms

def validate_prediction(idx, confidence, distances):
    margins = confidence - torch.cat((distances[:idx], distances[idx+1:]))
    min_margin = torch.min(margins)
    return confidence > args.threshold and min_margin > args.margin

def find_match(output_embedding, all_embeddings):
    with torch.no_grad():
        dist_func, arg_func, val_func = {
            'euclidean': (euclidean_distance, torch.argmin, torch.min),
            'cosine': (cosine_distance, torch.argmax, torch.max)
        }[args.mode]
        # softmax = nn.Softmax(dim=0)

        distances = dist_func(output_embedding, all_embeddings)
        # probs = softmax(distances * temperature)

        # print(distances, probs, arg_func(probs), val_func(probs))

        index, confidence = arg_func(distances), val_func(distances)
        isvalid = validate_prediction(index, confidence, distances)

        if args.dev and isvalid:
            devinfo.append({
                'prediction_index': arg_func(distances).item(),
                'prediction_confidence': val_func(distances).item(),
                'distances': distances.numpy().tolist(),
                'output_embeddings': output_embedding.numpy().tolist()
            })
            with open("logs.json", 'w') as f:
                json.dump(devinfo, f)

    return index, confidence, isvalid

def identify_faces(pil_image):
    faces = mtcnn(pil_image)
    if faces is not None:
        bounding_boxes, _ = mtcnn.detect(pil_image)
        preds = resnet(faces)

        matches = []
        for y, box in zip(preds, bounding_boxes):
            if (box[2] - box[0]) < args.ignore:
                continue

            idx, confidence, isvalid = find_match(y, embeddings)
            if isvalid:
                entity = ID[idx]
                matches.append((entity, confidence.item(), box))
            elif args.show_unrecognised and args.which != 'test':
                matches.append((None, None, box))
        
        return matches
    return None

def video():
    if args.session_id is None:
        sess_id = datetime.strftime(datetime.now(), '%Y-%m-%d_%a')
    else:
        sess_id = args.session_id
    
    counter = IDCounter(args.number, ID, sess_id, args.output_dir)

    print("Starting wio interface\n")
    itf = Interface(args.port)

    print("Starting video capture\n")
    cv2_capture = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cv2_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
    cv2_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
    cv2_capture.set(cv2.CAP_PROP_FPS, 2)
    cv2_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    video_capture = BufferlessVideoCapture(cv2_capture)

    registered_students = []
    while True:
        frame = video_capture.read()

        im = frame[:, :, [2, 1, 0]]

        matches = identify_faces(im)

        if args.ignore != 0:
            cv2.line(frame, (0, 2), (args.ignore, 2), (255, 0, 0), 5)

        summary_text = []
        summary_color = 0

        if matches is None:
            summary_color = 3
            summary_text.append("No face found")
        
        elif len(matches) == 0:
            summary_color = 3
            if args.show_unrecognised:
                summary_text.append("No face found")
            else:
                summary_text.append("Face(s) unrecognised")
        
        else:
            for entity, confidence, box in matches:
                bounds = box.astype(int)

                if args.show_unrecognised and entity is None:
                    summary_text.append("Face(s) unrecognised")
                    summary_color = 3
                    
                    if not args.blind:
                        cv2.rectangle(frame, (bounds[0], bounds[1]), (bounds[2], bounds[3]), (0, 0, 255), 2)
                
                else:
                    name = f"{entity['first']} {entity['last']}"
                    nick = entity['nick'] if entity['nick'] is not None else entity['first']

                    if counter.update(name):
                        if entity not in registered_students:
                            registered_students.append(entity)
                        if summary_color <= 1: summary_color = 1
                        color = (0, 255, 0)
                    else:
                        if summary_color <= 2: summary_color = 2
                        color = (0, 210, 255)
                    
                    text = f"{nick} [{round(confidence*100, 2)}%]"
                    summary_text.append(text)

                    if not args.blind:
                        cv2.putText(frame, text, (bounds[0], bounds[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.rectangle(frame, (bounds[0], bounds[1]), (bounds[2], bounds[3]), color, 2)

        summary_text = ', '.join(summary_text)
        print(summary_text)
        button_pressed = itf.exchange(summary_text, summary_color)
        if button_pressed:
            counter.save_unk_img(im)

        if not args.blind:
            cv2.imshow('TECLARS Main UI', frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
            
            if cv2.getWindowProperty('TECLARS Main UI', cv2.WND_PROP_VISIBLE) < 1:
                break
    
    video_capture.release()
    cv2.destroyAllWindows()

    print()

    time.sleep(1)

    print("\nRegistered students:")
    print("\n".join([
        f"{e}) {entity['first']} {entity['last']}"
        for e, entity in enumerate(registered_students)
    ]))

def test():
    print("Beginning testing")
    for filename in os.listdir(args.test_path):
        with Image.open(os.path.join(args.test_path, filename)) as im:
            matches = identify_faces(im)

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

def show():
    print("Peple:")
    for e, i in enumerate(ID):
        print(f"{e}) {i['first']} {i['last']} ({i['image']})")

def check():
    print("List of available compute devices [`name`: type]")
    print("`cpu`: CPU")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"`cuda{i}`: {torch.cuda.get_device_name(0)}")
    
    print("\nChecking camera stream.")
    print("Enter integer index of camera to check; then press `esc` to proceed to next check.")

    while True:
        cam = int(input("Enter camera index to check (-1 to exit): "))
        if cam == -1:
            break

        else:
            video_capture = cv2.VideoCapture(cam, cv2.CAP_V4L2)
            video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
            video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
            video_capture.set(cv2.CAP_PROP_FPS, 2)

            if video_capture is None or not video_capture.isOpened():
                print(f"No device available at camera {cam}")
            
            else:
                while True:
                    _, frame = video_capture.read()
                    cv2.imshow(f'TECLARS Cam Check: Camera {cam}', frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                
                video_capture.release()
                cv2.destroyAllWindows()

parser = argparse.ArgumentParser(description='TECLARS: Team Enigma CMC Lab Auto-Registration System')
parser.set_defaults(which='video')
subparsers = parser.add_subparsers(help='TECLARS subcommands (run without any subcommands to execute default system)')

parser.add_argument('-r', '--threshold', type=float, default=0.8,
                    help='Probability above which a face will be considered recognised')
parser.add_argument('-n', '--number', type=int, default=3,
                    help='Minimum number of frames above which a face will be considered recognised')
parser.add_argument('-g', '--margin', type=float, default=0.1,
                    help='Minimum probability margin above next likely face for the face to be considered recognised')
parser.add_argument('-t', '--temp', type=float, default=2,
                    help='Temperature: higher temperature creates higher probabilities for a recognised face')
parser.add_argument('-c', '--camera', type=int, default=0,
                    help='Index for camera to stream from')
parser.add_argument('-d', '--device', type=str, default='auto',
                    help='Device to compute algorithm on')
parser.add_argument('-m', '--mode', type=str, choices=['cosine', 'euclidean'], default='cosine',
                    help='Distance function for evaluating the similarity between face embeddings')
parser.add_argument('-u', '--show_unrecognised', action="store_false",
                    help='Remove bounding boxes around unrecognised faces')
parser.add_argument('-i', '--ignore', type=int, default=70,
                    help='Ignore faraway faces with a width smaller than this value (set 0 to include all faces)')
parser.add_argument('-s', '--session_id', type=str, default=None,
                    help="Session ID")
parser.add_argument('-l', '--load_dir', type=str, default=LOCALPARAMS['LOADDIR'],
                    help="Directory to `id.json` and `embeddings.pt`")                    
parser.add_argument('-o', '--output_dir', type=str, default=LOCALPARAMS['OUTDIR'],
                    help="Directory to output reports")
parser.add_argument('-p', '--port', type=str, default=LOCALPARAMS['PORT'],
                    help="Port to connect to Wio terminal")
parser.add_argument('-b', '--blind', action='store_true',
                    help="Do not display video")
parser.add_argument('-x', '--dev', action='store_true',
                    help="Enable developer options")

test_parser = subparsers.add_parser('test', help='Test system performance on a set of images in a given directory')
test_parser.set_defaults(which='test')
test_parser.add_argument('--test_path', type=str, default='./data/test',
                         help='Path to directory with test images')

test_parser = subparsers.add_parser('check', help='Checks system for available compute devices and camera streams')
test_parser.set_defaults(which='check')

test_parser = subparsers.add_parser('show', help='Shows list of students and their indices in the TECLARS system')
test_parser.set_defaults(which='show')

args = parser.parse_args()

try:
    if __name__ == "__main__":
        with open(os.path.join(args.load_dir, 'id.json'), 'r') as f:
            ID = json.load(f)
        
        if args.dev:
            devinfo = []
            print("Developer options enabled")
        
        if args.which == "check":
            check()
        elif args.which == "show":
            show()
        else:
            bundle = torch.load(os.path.join(args.load_dir, 'embeddings.pt'))
            embeddings = bundle['embedding']

            if args.device == "auto":
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device(args.device)
            print('Running on device: {}'.format(device))

            mtcnn = MTCNN(image_size=160, keep_all=True, device=device)

            print("Loading face recognition model")
            resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

            if args.which == 'video':
                video()
            elif args.which == 'test':
                test()

        print("\n[TECLARS terminated]")

except KeyboardInterrupt:
    print("\n[TECLARS session temporarily terminated by user]")

# TECLARS
*Team Enigma CMC Lab Auto-Registration System*

A simple-to-use end-to-end facial recognition registraion system powered by [`facenet-pytorch`](https://github.com/timesler/facenet-pytorch). Licensed under the The Commons Clause license, on top of the GNU GENERAL PUBLIC LICENSE Version 3.

## Table of Contents

* [Features](#features)

* [Quick Start](#quick-start)

## Features

All key facial recognition features have been implemented:

* one-shot classification by comparing embeddings of faces, so new faces can be easily added and removed without any re-training

* concurrent detection and recogntion of multiple faces in the same frame

* reasonable framerate even on CPUs (although supports GPU too, via PyTorch)

* automatically filters out faces with low-confidence, reducing the number of miss-recognitions

* real-time facial detection and recognition on webcam stream, visualised with annoated bounding boxes via OpenCV2

* supports both euclidean distance and cosine angle as measures of embedding similarities (cosine works better in practice)

Features that are planned to be implemented:

* automatic downloading of students' images via iSAMS API

* automatic registration of students via iSAMS API

## Quick Start

1. Clone this repository:
    ```shell
    git clone https://github.com/Harrow-Enigma/TECLARS.git
    ```

2. Install the following Python packages in your environment of choice. E.g.
    ```shell
    pip install facenet-pytorch opencv-python
    ```

3. Prepare the data.

    1. In the directory `data`, create two subdirectories named `images` and `test`.

    2. All reference face images should be stored under `images` and in either `JPG`, `JPEG` or `PNG` format. If you don't have any data on you, go ahead and copy over the images in `data/sample_images` into your newly created `data/images` directory.

    3. Place any image you would like to test the system on in `data/test`. Again, all images must be in either `JPG`, `JPEG` or `PNG` format. You can also copy the images in `data/sample_test` for a quick start.

    4. Create a file called `id.json` in the `data` directory. This will store all the information about the people the system will learn to recognise. It is a list of entires, each of which follows this format:
        ```json5
        {
            "first": "Someone",       // First name of person
            "last":  "Cool",          // Last name of person
            "house": null,            // House of person (null if N/A)
            "year":  null,            // Year group of person (null if N/A)
            "email": null,            // Email of person (null if N/A)
            "image": "some_image.png" // Filename of the person's image (no need to write out full path or directory)
        }
        ```
        If in doubt, reference the example in `data/id.example.json`. If you used the sample images in the above steps, then you can just copy across the content in `data/id.example.json` into your new `data/id.json`.

4. Generate the one-shot embeddings:
    ```shell
    python get_embed.py
    ```
    This step is crucial as it provides the model with a reference embedding with which it recognises individuals. **Remember to do this every time a new face is added!** It's quick and easy, and has nothing to do with retraining.

5. Run TECLARS on your webcam!
    ```shell
    python run.py
    ```
    Press `esc` to exit. Note that TECLARS will only put a bounding box around a face that it recognises. So you'll have to put in some images of yourself for this to work!

    * Additionally, you can also specify a few facial detection and recognition hyperparameters:

        ```
        usage: run.py [-h] [-x THRESHOLD] [-t TEMP] [-c CAMERA] [-d DEVICE] [-m {cosine,euclidean}] [-u] [-i IGNORE] {test,check} ...

        TECLARS: Team Enigma CMC Lab Auto-Registration System

        positional arguments:
        {test,check}          TECLARS subcommands (run without any subcommands to execute default system)
            test                Test system performance on a set of images in a given directory
            check               Checks system for available compute devices and camera streams

        optional arguments:
        -h, --help            show this help message and exit
        -x THRESHOLD, --threshold THRESHOLD
                                Probability above which a face will be considered recognised
        -t TEMP, --temp TEMP  Temperature: higher temperature creates higher probabilities for a recognised face
        -c CAMERA, --camera CAMERA
                                Index for camera to stream from
        -d DEVICE, --device DEVICE
                                Device to compute algorithm on
        -m {cosine,euclidean}, --mode {cosine,euclidean}
                                Distance function for evaluating the similarity between face embeddings
        -u, --show_unrecognised
                                Remove bounding boxes around unrecognised faces
        -i IGNORE, --ignore IGNORE
                                Ignore faraway faces with a width smaller than this value (set 0 to include all faces)
        ```

    * Use `python run.py test` to test TECLARS on your test images.
    
    * Use `python run.py check` to find the available options for your computing device (`--device`) and camera stream (`--camera`).

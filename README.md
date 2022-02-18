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
            "first": "Someone",       # First name of person
            "last":  "Cool",          # Last name of person
            "house": null,            # House of person (null of N/A)
            "year":  null,            # Year group of person (null of N/A)
            "email": null,            # Email of person (null of N/A)
            "image": "some_image.png" # Filename of the person's image (no need to write out full path or directory)
        }
        ```
        If in doubt, reference the example in `data/id.example.json`. If you used the sample images in the above steps, then you can just copy across the content in `data/id.example.json` into your new `data/id.json`.

4. Generate the one-shot embeddings:
    ```shell
    python get_embed.py
    ```

5. Run TECLARS on your webcam!
    ```shell
    python run.py
    ```
    Press `esc` to exit.

    * Additionally, you can also specify a few facial detection and recognition hyperparameters:

        ```
        usage: run.py [-h] [--threshold THRESHOLD] [--temp TEMP] [--mode {cosine,euclidean}] {test} ...

        TECLARS: Team Enigma CMC Lab Auto-Registration System

        positional arguments:
        {test}                TECLARS subcommands (run without any subcommands to run default system)
            test                Test system performance on a set of images in a given directory

        optional arguments:
        -h, --help            show this help message and exit
        --threshold THRESHOLD
                                Probability above which a face will be considered recognised
        --temp TEMP           Temperature: higher temperature creates higher probabilities for a recognised face
        --mode {cosine,euclidean}
                                Distance function for evaluating the similarity between face embeddings
        ```

    * Use `python run.py test` to test TECLARS on your test images.

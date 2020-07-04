#!/bin/bash

python test.py examples/petter/petter?.png --checkpoint_dir model_logs/celebs
python test.py examples/petter/skane?.png --checkpoint_dir model_logs/places

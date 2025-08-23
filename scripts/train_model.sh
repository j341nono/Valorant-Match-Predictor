#!/bin/bash

uv run src/data_processing/data_processing.py
uv run src/models/train_diff.py
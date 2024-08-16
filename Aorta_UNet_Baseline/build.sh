#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t aorta2024_unet_hust:v6 "$SCRIPTPATH" #--no-cache
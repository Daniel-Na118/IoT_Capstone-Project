#!/bin/bash

set -e

YEAR=${2:-2014}
if [ -z "$1" ]; then
  echo "usage download_mscoco.sh [data dir] (2014|2017)"
  exit
fi

if [ "$(uname)" == "Darwin" ]; then
  UNZIP="tar -xf"
else
  UNZIP="unzip -nq"
fi

# Output Directory
OUTPUT_DIR="${1%/}"
mkdir -p "${OUTPUT_DIR}"

function download_and_unzip() {
  local BASE_URL=${1}
  local FILENAME=${2}

  if [ ! -f "${FILENAME}" ]; then
    echo "Downloading ${FILENAME} to $(pwd)"
    wget -nd -c "${BASE_URL}/${FILENAME}"
  else
    echo "Skipping download of ${FILENAME}"
  fi
  echo "Unzipping ${FILENAME}"
  ${UNZIP} "${FILENAME}"
  rm "${FILENAME}"
}

cd "${OUTPUT_DIR}"

# Download the images
BASE_IMAGE_URL="http://images.cocodataset.org/zips"

TRAIN_IMAGE_FILE="train${YEAR}.zip"
download_and_unzip ${BASE_IMAGE_URL} "${TRAIN_IMAGE_FILE}"
mv "train${YEAR}" "Images"

VAL_IMAGE_FILE="val${YEAR}.zip"
download_and_unzip ${BASE_IMAGE_URL} "${VAL_IMAGE_FILE}"
mv "val${YEAR}" "Images_val"

# Download the annotations
BASE_INSTANCES_URL="http://images.cocodataset.org/annotations"
INSTANCES_FILE="annotations_trainval${YEAR}.zip"
download_and_unzip ${BASE_INSTANCES_URL} "${INSTANCES_FILE}"
mkdir annotations
cp "annotations_trainval${YEAR}/instances_train${YEAR}.json" "annotations/train.json"
mv "annotations_trainval${YEAR}/instances_val${YEAR}.json" "annotations/val.json"
rm -rf "annotations_trainval${YEAR}"
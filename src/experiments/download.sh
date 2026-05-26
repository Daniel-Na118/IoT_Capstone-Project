# 1. Navigate to your dataset folder
cd ~/coco_dataset

# 2. Define the variables and the function
YEAR=2017
UNZIP="unzip -nq"

download_and_unzip() {
  local BASE_URL=${1}
  local FILENAME=${2}
  if [ ! -f "${FILENAME}" ]; then
    echo "Downloading ${FILENAME}..."
    wget -nd -c "${BASE_URL}/${FILENAME}"
  else
    echo "Skipping download, file exists."
  fi
  echo "Unzipping ${FILENAME}..."
  ${UNZIP} "${FILENAME}"
  rm "${FILENAME}"
}

# 3. Run the specific annotation logic (with the fix for the folder name)
BASE_INSTANCES_URL="http://images.cocodataset.org/annotations"
INSTANCES_FILE="annotations_trainval${YEAR}.zip"

download_and_unzip ${BASE_INSTANCES_URL} "${INSTANCES_FILE}"

# NOTE: The zip usually extracts into a folder named 'annotations' 
# If the folder 'annotations_trainval2017' doesn't exist, we use 'annotations'
if [ -d "annotations_trainval${YEAR}" ]; then
    mv "annotations_trainval${YEAR}/instances_train${YEAR}.json" "annotations/train.json"
    mv "annotations_trainval${YEAR}/instances_val${YEAR}.json" "annotations/val.json"
    rm -rf "annotations_trainval${YEAR}"
else
    # This handles the case where it unzipped directly into /annotations
    mv "annotations/instances_train${YEAR}.json" "annotations/train.json"
    mv "annotations/instances_val${YEAR}.json" "annotations/val.json"
fi
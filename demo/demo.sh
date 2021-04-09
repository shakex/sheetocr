#! /bin/sh
###
 # @Des: demo.sh
 # @Date: 2020-12-17 13:58:04
 # @@Author: kxie
### 


if [ $# -eq 0 ] ;then
    echo
    echo "OCR demo v1.0 in python."
    echo "Usage: `basename $0` [image_path] [csv_path] [is_plot](0: not plot, 1: plot)"
    echo
    exit 1 
fi

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

# env
# python path
# TODO

# requirements
# TODO

# params
IMAGE_PATH=$1
CSV_PATH=$2
IS_PLOT=$3


# reqired python files
# linux or macOS
INFERENCE_FILE="$(dirname "$BASE_DIR")/inference.py"
VIS_FILE="$(dirname "$BASE_DIR")/export/xml_vis.py"
XML2CSV_FILE="$(dirname "$BASE_DIR")/export/xml2csv.py"
LOG_FILE="$BASE_DIR/log/$(date +%Y%m%d%H%M%S)_$(echo $(basename $IMAGE_PATH) | cut -d . -f1).log"

XML_DIR="$BASE_DIR/xml"
XML_PATH="$XML_DIR/$(echo $(basename $IMAGE_PATH) | cut -d . -f1).xml"
VIS_PATH="$BASE_DIR/vis/$(echo $(basename $IMAGE_PATH) | cut -d . -f1)_ocr.png"


main(){
    # inference
    cd $(dirname "$BASE_DIR")
    echo "---------"                                                >>$LOG_FILE
    echo "inference"                                                >>$LOG_FILE
    echo "---------"                                                >>$LOG_FILE
    python $INFERENCE_FILE -i $IMAGE_PATH -o $XML_DIR               >>$LOG_FILE

    # plot result
    cd "$(dirname "$BASE_DIR")/export"
    if [ $IS_PLOT == '1' ]; then
        echo                                                        >>$LOG_FILE
        echo "-----------"                                          >>$LOG_FILE
        echo "plot result"                                          >>$LOG_FILE
        echo "-----------"                                          >>$LOG_FILE
        python $VIS_FILE -i $IMAGE_PATH -x $XML_PATH -o $VIS_PATH   >>$LOG_FILE
        echo "plot image saved in $VIS_FILE."
    fi

    # xml2csv
    cd "$(dirname "$BASE_DIR")/export"
    echo                                                            >>$LOG_FILE
    echo "---------------"                                          >>$LOG_FILE
    echo "export csv file"                                          >>$LOG_FILE
    echo "---------------"                                          >>$LOG_FILE
    python $XML2CSV_FILE -x $XML_PATH -o $CSV_PATH                  >>$LOG_FILE

    echo "Success. csv file saved in '$CSV_PATH', see '$LOG_FILE' for more details."
}

main
exit 0




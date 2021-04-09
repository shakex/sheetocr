#! /bin/sh
###
 # @Des: Start ocr service
 # @Date: 2020-12-21
 # @@Author: kxie
### 

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$BASE_DIR/log/$(date +%Y%m%d%H%M%S)_serve.log"
OCR_SERVE_FILE="$BASE_DIR/ocr_serve.py"

echo "%Y-%m-%d %H:%M:%S - [INFO] - OCR service Begin to start."    >>$LOG_FILE &
nohup python $OCR_SERVE_FILE                                       >>$LOG_FILE &

sleep 2
echo "OCR SERVICE STARTS."
echo
echo "--------begin--------"
ps -ef | grep ocr_serve
echo "---------end---------"
echo
echo "tail -f $LOG_FILE to see log."
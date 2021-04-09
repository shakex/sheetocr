#! /bin/sh
###
 # @Des: Stop ocr service
 # @Date: 2020-12-21
 # @@Author: kxie
### 

typeset appid;
appid=`ps -ef|grep ocr_serve|grep -v grep|grep -v ps|awk '{print $2}'`

for loop in $appid
do
    if [ -n "$loop" ] && [ $loop -gt "0" ]; then
        kill -9 $loop
        echo ""
    else
        echo ""
    fi
done

echo
echo "OCR SERVICE STOPED."
ps -ef | grep ocr_serve

echo
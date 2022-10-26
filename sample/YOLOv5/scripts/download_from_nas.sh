#!/bin/bash

echo -e "===============================================================================================\n"
 
command="grep"
 
# judge platform, on macOS we need use ggrep command
case "$OSTYPE" in
  solaris*) echo -e " OSType: SOLARIS! Aborting.\n"; exit 1 ;;
  darwin*)  echo -e " OSType: MACOSX!\n"; command="ggrep" ;;
  linux*)   echo -e " OSType: LINUX!\n" ;;
  bsd*)     echo -e " OSType: BSD! Aborting.\n"; exit 1 ;;
  msys*)    echo -e " OSType: WINDOWS! Aborting.\n"; exit 1 ;;
  *)        echo -e " OSType: unknown: $OSTYPE\n"; exit 1 ;;
esac
 
# judge if grep/ggrep exist
type $command >/dev/null 2>&1 || { echo >&2 "Using brew to install GUN grep first.  Aborting."; exit 1; }
 
if [ $# -eq 0 ];then
  echo -e "Usage: $0 sharing_file_path [save_path].\n"
  exit -1
fi
 
web_prefix="http://219.142.246.77:65000/fsdownload/"
file_url=$1

# id=echo $file_url | ggrep -Po '(?<=sharing/).*(?=/)'
id=`echo $file_url | cut -d "/" -f 5`
sid=`curl -i $file_url | $command  -Po '(?<=sid=).*(?=;path)'`
v=`curl -i $file_url | $command -Po '(?<=none&quot;&v=).*(?=">)'`
file_name=`curl -b "sharing_sid=${sid}" -i "http://219.142.246.77:65000/sharing/webapi/entry.cgi?api=SYNO.Core.Sharing.Session&version=1&method=get&sharing_id=%22${id}%22&sharing_status=%22none%22&v=${v}" | $command -Po '(?<="filename" : ").*(?=")'`

if [ $# -eq 2 ];then
  save_path=$2
else
  save_path=$file_name
fi

echo -e "\ndownload with sid=$sid\n"
curl -o $save_path -b "sharing_sid=${sid}" "${web_prefix}${id}/${file_name}"
echo -e "\nDone! Saved to $save_path.\n"

echo -e "===============================================================================================\n"
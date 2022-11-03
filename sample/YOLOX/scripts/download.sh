function download_nas()
{
    if [ $# -eq 0 ];then
        judge_ret 1 "download_nas"
    fi
    command="grep"

    case "$OSTYPE" in
    solaris*) echo -e " OSType: SOLARIS! Aborting.\n"; exit 1 ;;
    darwin*)  echo -e " OSType: MACOSX!\n"; command="ggrep" ;;
    linux*)   echo -e " OSType: LINUX!\n" ;;
    bsd*)     echo -e " OSType: BSD! Aborting.\n"; exit 1 ;;
    msys*)    echo -e " OSType: WINDOWS! Aborting.\n"; exit 1 ;;
    *)        echo -e " OSType: unknown: $OSTYPE\n"; exit 1 ;;
    esac
    # judge platform, on macOS we need use ggrep command

    type $command >/dev/null 2>&1 || { echo >&2 "Using brew to install GUN grep first.  Aborting."; exit 1; }
    
    web_prefix="http://219.142.246.77:65000/fsdownload/"
    file_url=$1
    
    # id=echo $file_url | ggrep -Po '(?<=sharing/).*(?=/)'
    id=`echo $file_url | cut -d "/" -f 5`
    sid=`curl -i $file_url | $command  -Po '(?<=sid=).*(?=;path)'`
    v=`curl -i $file_url | $command -Po '(?<=none"&v=).*(?=">)'`
    file_name=`curl -b "sharing_sid=${sid}" -i "http://219.142.246.77:65000/sharing/webapi/entry.cgi?api=SYNO.Core.Sharing.Session&version=1&method=get&sharing_id=%22${id}%22&sharing_status=%22none%22&v=${v}" | $command -Po '(?<="filename" : ").*(?=")'`
    
    if [ $# -eq 2 ];then
        save_path=$2
    else
        save_path=$file_name
    fi
    curl -o $save_path -b "sharing_sid=${sid}" "${web_prefix}${id}/${file_name}"
    
    line_0=`cat $save_path|awk -F "\"" '{print $1}'`
    line_1=`cat $save_path|awk -F "\"" '{print $2}'`
    # if [ $line_0 = "{" ];then
    #     if [ $line_1 = "error" ];then
    #         judge_ret 1 "download "$file_url
    #     fi
    # fi
}
url=http://219.142.246.77:65000/sharing/DhuW74zyD
download_nas $url data.zip
unzip -d $(pwd)/.. ./data.zip
rm data.zip
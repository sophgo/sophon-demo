#!/bin/bash
script_dir=$(dirname $(readlink -f "$0"))
project_dir=${script_dir}/..
project_name=sophon-demo

set -e

pushd ${project_dir}

commit_id=$(git log -1 | awk 'NR==1 {print substr($2,0,8)}')
times=`date +%Y%m%d`

VERSION_PATH=$project_dir/git_version
echo $VERSION_PATH
line=$(cat $VERSION_PATH)
SOPHON_DEMO_VERSION=$line

dst_file_name="${project_name}_v${SOPHON_DEMO_VERSION}_${commit_id}_${times}"
echo "save name: ${dst_file_name}"

dst_dir=$project_dir/release
sophon_demo_dir=${dst_dir}/${dst_file_name}

rm -rf ${dst_dir}
mkdir ${dst_dir}
mkdir ${sophon_demo_dir}

cp -r `ls ${project_dir} -A | grep -v "release"` ${sophon_demo_dir}/
rm -rf ${sophon_demo_dir}/.git

pushd ${dst_dir}
tar -cvzf ${dst_file_name}.tar.gz ${dst_file_name}
rm -rf ${sophon_demo_dir}
popd

popd

echo "saved: ${dst_dir}/${dst_file_name}"
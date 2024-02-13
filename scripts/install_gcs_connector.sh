#!/usr/bin/env bash
set -ex

# option for user site packages
if [[ "$1" == "--user" ]]; then
    echo "installing to user site packages"
    site_packages=$(python -c 'import site; print(site.getusersitepackages())')
else
    site_packages=$(python -c 'import site; print(site.getsitepackages()[0])')
fi

lib_dir="${site_packages}/pyspark/jars"
mkdir -p "${lib_dir}"
cd "${lib_dir}"
wget https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar

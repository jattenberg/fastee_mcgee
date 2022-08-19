#!/bin/bash

set -e

if ! command -v virtualenv &> /dev/null
then
    echo "virtualenv could not be found"
    pip install pipx;
else
  venv_path=$(which virtualenv);
  echo "using venv located at $venv_path"
fi

venv="venv"
echo "making virtualenv: $venv"

virtualenv $venv

$venv/bin/pip install --upgrade pip pytest
$venv/bin/pip install pip-tools
$venv/bin/pip install -r requirements.txt
$venv/bin/jupyter nbextension install https://github.com/drillan/jupyter-black/archive/master.zip
$venv/bin/jupyter nbextension enable jupyter-black-master/jupyter-black

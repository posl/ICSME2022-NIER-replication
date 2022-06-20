#!/bin/bash

readonly SCRIPT_NAME=${0##*/}
readonly SCRIPT_DIR=$(cd $(dirname $0); pwd)


print_help()
{
  cat << END
Usage: $SCRIPT_NAME PARAM [-h]

Launch jupyter-lab on docker.

Param:
  cpu     Launch cpu mode.
  gpu     Launch gpu mode. (require gpu environment.)

Options:
  -h      Display help message and exit.

Example:
  $SCRIPT_NAME cpu
  $SCRIPT_NAME gpu
  $SCRIPT_NAME -h
END
}


print_error()
{
  cat << END 1>&2
$SCRIPT_NAME: $1
Try -h option for more information.
END
}


# parse arguments
for OPT in "$@"
do
  echo $OPT
  case "$OPT" in
    '-h'|'--help')
      print_help
      exit 0
      ;;
    -*)
      print_error "unrecognized option -- '$OPT'"
      exit 1
      ;;
    *)
      if [[ ! -z "$1" ]] && [[ ! "$1" =~ ^-+ ]]; then
        param+=( "$1" )
        shift 1
      fi
      ;;
  esac
done


# check param
if [[ ${param[0]} != "cpu" ]] && [[ ${param[0]} != "gpu" ]]; then
  print_error "Param is only 'cpu' or 'gpu'."
  exit 1
fi

readonly launch_mode=${param[0]}

gpu_option=
dockerfile=

if [[ $launch_mode = "cpu" ]]; then
  dockerfile="Dockerfile"
fi

if [[ $launch_mode = "gpu" ]]; then
  gpu_option="--gpus all"
  dockerfile="dockerfile_gpu"
fi


# docker image build
readonly image_name="jupyter-lab"
readonly image_version="latest"
readonly build_context="${SCRIPT_DIR}"
docker build -f ${build_context}/${dockerfile} ${build_context} -t ${image_name}:${image_version}


# docker run
readonly host_dir="$(pwd)"
readonly container_dir="/work"
readonly container_name="icsme-nier-replica"
readonly image_id=$(docker images | grep ${image_name} | awk '{print $3}')
docker run --name ${container_name}  --rm ${gpu_option} -v ${host_dir}:${container_dir} -p 9999:9999 $image_id

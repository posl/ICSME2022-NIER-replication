#!/bin/bash

docker exec -it icsme-nier-replica /bin/bash -c "cd src/ && python build_datasets.py"

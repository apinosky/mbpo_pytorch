#!/bin/sh

# hearbeat to wait for some process to finish before continuing (can check parent id with ps -f <pid>)
# pid=4678
# while ps -p $pid; do sleep 60; done

# main loop
for env in 'Hopper-v2' 'HalfCheetah-v2' #
do
    for i in $(seq 13 100 950)
    do
        echo $env $i
        python3 main_mbpo.py --seed $i --env_name $env
    done
done

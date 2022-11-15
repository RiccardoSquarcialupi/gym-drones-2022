#!/bin/bash

# Provide number of environments to be generated

if [ -z $1 ]
then
  N=10
else
  N=$1
fi

#OLD EASY are 80
#GOOD PARAMS ARE: 140 sphere = 0.3
if [ -z $2 ]
then
  DENSITY_MULTIPLIER=1
else
  DENSITY_MULTIPLIER=$2
fi

# Provide starting number for env to generate.
if [ -z $3 ]
then
  STARTING_POINT=0
else
  STARTING_POINT=$3
fi

#rm -rf generated_envs

END=$(($STARTING_POINT + $N))

for i in $(eval echo {$STARTING_POINT..$END})
do
  dirname="generated_envs/environment_""$i"
  mkdir -p $dirname
  python3 obstacle_generator.py "$RANDOM" $DENSITY_MULTIPLIER
  mv static_obstacles.csv $dirname
done

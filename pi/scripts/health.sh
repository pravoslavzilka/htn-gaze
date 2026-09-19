#!/bin/sh
while true; do
  echo "$(date +%T) temp=$(cat /dev/thermal) $(pidin info | grep -o "FreeMem:[0-9]*MB")" >> /data/home/qnxuser/health.log
  sleep 2
done

#!/bin/bash

echo "Starting 8 clients in the background..."

for i in {1..8}
do
   echo "Starting client $i..."
   # Set CLIENT_ID environment variable and run client.py in the background (&)
   CLIENT_ID=$i python client.py &
done

# 'wait' will pause the script until all background jobs (clients) finish
wait
echo "All clients have finished."
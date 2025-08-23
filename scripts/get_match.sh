#!/bin/bash

mkdir -p data

for ((from=1; from<=30; from+=10)); do
    to=$((from+9))
    echo "Fetching pages ${from} to ${to}..."
    curl -s "https://vlrggapi.vercel.app/match?q=results&from_page=${from}&to_page=${to}&timeout=120" \
         -o "data/data_match_${from}-${to}.json"
    sleep 1
done

REGION="ap"
TIMESPAN="all"

curl "https://vlrggapi.vercel.app/stats?region=${REGION}&timespan=${TIMESPAN}" -o data/data_ap_all.json
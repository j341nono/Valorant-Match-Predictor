
NUM_PAGES=7

curl "https://vlrggapi.vercel.app/match?q=results&num_pages=${NUM_PAGES}" -o data/data_match.json
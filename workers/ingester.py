from datetime import datetime

import requests
from apscheduler.schedulers.blocking import BlockingScheduler

INTERVAL_HOURS = 23


def query_operation():
    url = "http://127.0.0.1:8000/ingest"
    headers = {"Content-Type": "application/json"}
    data = {"days": 1}
    response = requests.post(url, json=data, headers=headers)
    print(f"Current time: {datetime.now()}")
    print(f"- Response: {response.status_code}, {response.json()}")


scheduler = BlockingScheduler()
scheduler.add_job(query_operation, "interval", hours=INTERVAL_HOURS)

print(f"Starting scheduler to run /ingest every {INTERVAL_HOURS} hours...")
query_operation()
scheduler.start()

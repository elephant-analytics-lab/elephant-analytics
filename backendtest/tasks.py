# tasks.py
from time import sleep
from celery_app import celery_app

@celery_app.task(name="tasks.add")
def add(x: int, y: int) -> int:
    # Simulate a slow calculation
    sleep(5)
    return x + y

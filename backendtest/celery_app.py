# celery_app.py
from celery import Celery

# Create Celery instance
celery_app = Celery(
    "my_celery_app",
    broker="redis://localhost:6379/0",      # Redis broker
    backend="redis://localhost:6379/0",     # Redis backend (for results)
)

# Optional: Celery configuration
celery_app.conf.update(
    task_routes={
        "tasks.add": {"queue": "math"},
    },
)

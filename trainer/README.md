# Run locally
To run a model locally, adjust main.py, navigate to the root of the project and then run:

gcloud ml-engine local train --module-name trainer.main --package-path ./trainer

# Run Gcloud
Should be: \
gcloud ml-engine jobs submit training test12323 --stream-logs --runtime-version 1.4 --job-dir gs://mlip/test12323/ --package-path ./trainer --module-name trainer.main --region europe-west1 --config ./trainer/config.yaml -- --GCP


# Run locally
To run a model locally, adjust main.py, navigate to the root of the project and then run:

gcloud ml-engine local train --module-name GCP.main --package-path ./ -- --GCP 0 --job-dir ./

# Run Gcloud
Should be: \
gcloud ml-engine jobs submit training test12315 --stream-logs --runtime-version 1.4 --job-dir gs://mlip-test/test12315/ --package-path ./ --module-name GCP.main --region europe-west1 --config ./GCP/trainer/config
.yaml -- --GCP 0

# Task Orchestrator

This repository provides a small FastAPI service that manages `volcal_baseline` tasks.  It
handles file uploads through [tusd](https://github.com/tus/tusd), stores task metadata in
PostgreSQL and keeps result files in MinIO.  Heavy processing is executed in a background
thread so that the API remains responsive.

## Features

- **Asynchronous uploads** via the TUS protocol (handled by `tusd`).
- **Background worker** that runs the `volcal_baseline` pipeline after all files are uploaded.
- **PostgreSQL** for task information and status tracking.
- **MinIO / S3** for storing processed assets with presigned download links.

## Getting Started

The easiest way to run the service is with Docker Compose:

```bash
# clone the repository
git clone <this repo>
cd volcal_baseline_server

# start all services
docker compose up
```

The compose file starts the API server, a PostgreSQL container, `tusd` for uploads and a
local MinIO instance.  After the containers are ready the API will be available on
`http://localhost:8000`.

### Environment Variables

The application relies on the following variables (defaults are provided in the compose
file):

- `DATABASE_URL` – PostgreSQL connection string
- `TUS_ENDPOINT` – URL of the tusd server
- `MINIO_ENDPOINT` – MinIO/S3 endpoint used for results
- `MINIO_ACCESS_KEY` and `MINIO_SECRET_KEY`
- `AWS_S3_ADDRESSING_STYLE` – usually `path`

When running with Docker Compose these values are set automatically.

## API Overview

### `POST /create_task`
Creates a new task and returns four upload URLs (`pre`, `post`, `shp`, `shx`).  Each URL is a
TUS endpoint where the corresponding file can be uploaded.

### `POST /start_task/{task_id}`
Marks a task ready for processing once all uploads are finished.  If files are missing the
response lists which ones are still required.

### `GET /query_task/{task_id}`
Returns the current status of a task.  When processing is complete the response contains the
result structure with presigned URLs to any generated images.

### `POST /tusd_hook`
Internal endpoint called by `tusd` after an upload succeeds.  It updates the task and will
start processing automatically when every file has been uploaded.

## Task Statuses

Tasks move through several states:

1. `waiting` – task created, waiting for file uploads
2. `pending` – all files uploaded, queued for worker
3. `running` – worker is processing
4. `done` – processing finished successfully
5. `error` – processing failed, see `results` field for details
6. `killed` – worker was terminated before finishing

## Example Workflow

1. Call `/create_task` to obtain the upload URLs and `task_id`.
2. Upload each file to its URL using any TUS client.
3. Once all uploads complete, call `/start_task/{task_id}`.
4. Poll `/query_task/{task_id}` until the status becomes `done`.
5. Download result images using the provided presigned URLs.

## License

This project is released under the MIT License.

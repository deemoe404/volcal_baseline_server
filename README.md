# Task Orchestrator

This repository provides a small FastAPI service that manages `volcal_baseline` tasks. 
This is useful when porting the pipeline to some edge embedded devices like DJI remotes or 
mobile phones. It handles file uploads through [tusd](https://github.com/tus/tusd), stores 
task metadata in PostgreSQL and keeps result files in MinIO.

This project relies on the core system specifications defined in [volcal_baseline](https://github.com/deemoe404/volcal_baseline).

## Features

- **Asynchronous uploads** via the TUS protocol (handled by `tusd`).
- **Background worker** that runs the `volcal_baseline` pipeline after all files are uploaded.
- **PostgreSQL** for task information and status tracking.
- **MinIO / S3** for storing processed assets with presigned download links.

## Getting Started

> Before getting started, please review the environment variables in `docker-compose.yml`,
> particularly for the `app` container. Ensure that any private IP addresses are updated
> to reflect your actual server address.

The easiest way to run the service is with Docker Compose:

```bash
# clone the repository
git clone --recurse-submodules https://github.com/deemoe404/volcal_baseline_server.git
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
- **`INPUT`**
  - `None` This endpoint does not require any input.
- **`OUTPUT`**
  - `200 OK` The task has been successfully created and is ready to receive file uploads.
    ```json
    {
      "id": "<task_id>",
      "upload_urls": {
        "pre":  "<URL>", "post": "<URL>", "shp":  "<URL>", "shx":  "<URL>"
      }
    }
    ```
    - `id` Task UUID. You can use this ID to query task info later.
    - `upload_urls` A dictionary of TUS upload endpoints:
      - `pre` **Reference** point cloud (unchanged scan).
      - `post` **Target** point cloud (changed scan).
      - `shp` **Stable Area** shapefile (.shp).
      - `shx` **Stable Area** index file (.shx).

### `POST /start_task/{task_id}`
>This endpoint exists to support potential future features.

Marks a task as ready for processing once all uploads are complete.  If any files are missing, the response will list the ones that are still required.
Normally, you won't need to call this API manually, as the task starts automatically once all four files are successfully uploaded.
- **`INPUT`**
  - `task_id` The UUID of the task to start.
- **`OUTPUT`**
  - `200 OK` The task successfully started or already pending/running.
    ```json
    { "started": true }
    ```
  - `400 Bad Request` Some required files are still missing.
    ```json
    { "missing": "<lable1>", "<lable2>", "..." }
    ```
  - `404 Not Found` Invalid task ID.
    ```json
    { "detail": "Task not found" }
    ```

### `GET /query_task/{task_id}`
Returns the current status of a task.  When processing is complete the response contains the
result structure with presigned URLs to any generated images.
- **`INPUT`**
  - `task_id` The UUID of the task to query.
- **`OUTPUT`**
  - `200 OK` The task exists and statue is returned.
    ```json
    {
      "status": "done",
      "results": {
        "hulls": [
          {
            "score": 0.94,
            "image_url": "<presigned_url>",
            "..." : "..."
          },
          "..."
        ]
      }
    }
    ```
    - `status` Task status (`waiting`, `pending`, `running`, `done`, or `error`. See [Task Statuses](#task-statuses) for more details.)
    - `results` Present only when `status` is `done`. Contains details for each detected hull, along with corresponding presigned image URLs.
  - `404 Not Found` Invalid task ID.

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

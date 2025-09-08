## Build and upload MaxText + JetStream + Pathways Server image

These instructions are to build the MaxText + JetStream + Pathways Server image, which calls an entrypoint script that invokes the [JetStream](https://github.com/AI-Hypercomputer/JetStream) inference server with the MaxText framework. 

```
docker build -t jetstream-pathways .
docker tag jetstream-pathways us-docker.pkg.dev/${PROJECT_ID}/jetstream/jetstream-pathways:latest
docker push us-docker.pkg.dev/${PROJECT_ID}/jetstream/jetstream-pathways:latest
```

If you would like to change the version of MaxText or JetStream the image is built off of, change the `MAXTEXT_VERSION` / `JETSTREAM_VERSION` environment variable:
```
ENV MAXTEXT_VERSION=<your desired commit hash, release, or tag>
ENV JETSTREAM_VERSION=<your desired commit hash, release, or tag>
```
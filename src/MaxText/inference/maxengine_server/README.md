## Build and upload Maxengine Server image

These instructions are to build the Maxengine Server image, which calls an entrypoint script that invokes the [JetStream](https://github.com/AI-Hypercomputer/JetStream) inference server with the MaxText framework. 

```
docker build -t maxengine-server .
docker tag maxengine-server us-docker.pkg.dev/${PROJECT_ID}/jetstream/maxengine-server:latest
docker push us-docker.pkg.dev/${PROJECT_ID}/jetstream/maxengine-server:latest
```

If you would like to change the version of MaxText or JetStream the image is built off of, change the `MAXTEXT_VERSION` / `JETSTREAM_VERSION` environment variable:
```
ENV MAXTEXT_VERSION=<your desired commit hash, release, or tag>
ENV JETSTREAM_VERSION=<your desired commit hash, release, or tag>
```
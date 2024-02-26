# Use Python 3.10 as the base image
FROM python:3.10-slim-bullseye


# Copy all files from local workspace into docker container
COPY . .
RUN ls .


WORKDIR /app

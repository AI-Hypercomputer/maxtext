ARG BASEIMAGE=maxtext_base_image
FROM $BASEIMAGE


RUN apt-get update && apt-get install -y python3-pip


WORKDIR /app
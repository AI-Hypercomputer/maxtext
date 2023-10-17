ARG BASEIMAGE=maxtext_base_image
FROM $BASEIMAGE

#FROM maxtext_base_image

# Set the working directory in the container
WORKDIR /app

# Copy all files from local workspace into docker container
COPY . .

WORKDIR /app

# Copy the Bash script into the image
# COPY save_xaot.sh /tmp/
# RUN chmod +x /tmp/save_xaot.sh
# # Set the default command to run the Bash script
# RUN /tmp/save_xaot.sh
apt-get install docker-compose-plugin
docker compose version
gcloud auth configure-docker us-docker.pkg.dev --quiet

cd utils_pathways
docker compose down
sleep 2
docker compose up &
sleep 5
docker compose ps

export JAX_PLATFORMS="proxy"
export JAX_BACKEND_TARGET="grpc://localhost:29000"
echo "Setup for Pathways containers complete!"
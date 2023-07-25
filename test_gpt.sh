COMMAND="true"

# Run the command until it succeeds
while true; do
    # Attempt to run the command
    if $COMMAND; then
        echo "Command succeeded!"
        break
    else
        echo "Command failed! Retrying..."
        sleep 1  # Adjust the sleep duration as needed
    fi
done
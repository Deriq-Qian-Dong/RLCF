message=$1
time=$(date "+%Y-%m-%d %H:%M:%S")
if [ -z "$message" ]
then
    message="Update"
fi
message="$message at $time"
git add .
git commit -m "$message"
git push 

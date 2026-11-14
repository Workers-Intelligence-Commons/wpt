#!/bin/sh
# Starts nginx with the depot's HTTPS server enabled once its certificate
# exists. Until then nginx serves port 80 only, which lets certbot get the
# first certificate, and this script looks for it every minute. Every 12
# hours nginx reloads, which picks up a renewed certificate.
set -eu
conf=/etc/nginx/depot/nginx.conf
cert=/etc/letsencrypt/live/depot.workers-intelligence-commons.org/fullchain.pem
https=/run/nginx-https/depot.conf

mkdir -p /run/nginx-https
if [ -f "$cert" ]; then
    cp /etc/nginx/depot/https.conf "$https"
fi

(
    minutes=0
    while sleep 60; do
        minutes=$((minutes + 1))
        if [ ! -f "$https" ] && [ -f "$cert" ]; then
            cp /etc/nginx/depot/https.conf "$https"
            nginx -c "$conf" -s reload
            minutes=0
        elif [ "$minutes" -ge 720 ]; then
            nginx -c "$conf" -s reload
            minutes=0
        fi
    done
) &

exec nginx -c "$conf" -g "daemon off;"

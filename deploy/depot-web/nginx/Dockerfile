# The public depot web app's reverse proxy (compose.yaml beside the nginx
# folder): the official nginx image with this folder's configuration, so a
# change to it reaches the server with `docker compose up -d --build`.
# The image is pinned by digest: a new version is a change to its line, and
# its license is checked again (docs/licenses.md).
FROM nginx:1.30-trixie@sha256:42c68eab6f7e22611786a9fe1d1c019e275c50185bd54b8f75dc2b9184322703
COPY nginx.conf https.conf start.sh /etc/nginx/depot/
CMD ["sh", "/etc/nginx/depot/start.sh"]

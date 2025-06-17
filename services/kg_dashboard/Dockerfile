# NOTE: This Dockerfile was partially generated using AI assistance.

# Simple Dockerfile to serve pre-built static files
FROM nginx:alpine

# Copy built files from local build directory
COPY ./build /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"] 
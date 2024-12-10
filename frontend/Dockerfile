FROM nginx:alpine

# Copy files to Nginx directory
COPY ./ /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Start NGINX
CMD ["nginx", "-g", "daemon off;"]


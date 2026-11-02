FROM node:22-slim
WORKDIR /app
COPY package*.json ./
RUN npm install --production
COPY server.js ./
EXPOSE 3001
CMD ["node", "server.js"]

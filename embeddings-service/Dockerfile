FROM node:20-alpine

WORKDIR /app

COPY embeddings-service/package.json ./
RUN npm install

COPY embeddings-service ./embeddings-service
WORKDIR /app/embeddings-service

EXPOSE 8001

CMD ["node", "src/index.js"]

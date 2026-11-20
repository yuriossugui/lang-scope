import express from 'express';
import { generateEmbeddings, health } from './embeddings.js';

const app = express();
app.use(express.json({ limit: '10mb' }));

app.get('/health', health);
app.post('/embed', generateEmbeddings);

const PORT = parseInt(process.env.PORT || '8001', 10);
app.listen(PORT, () => {
  console.log(`Embeddings service listening on port ${PORT}`);
});

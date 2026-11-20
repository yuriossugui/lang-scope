import { pipeline } from '@xenova/transformers';

let extractor = null;

async function getExtractor() {
  if (!extractor) {
    extractor = await pipeline(
      'feature-extraction',
      'Xenova/all-MiniLM-L6-v2'
    );
  }
  return extractor;
}

export async function health(req, res) {
  res.json({ status: 'ok', model: 'Xenova/all-MiniLM-L6-v2' });
}

export async function generateEmbeddings(req, res) {
  try {
    const { texts, model } = req.body;

    if (!Array.isArray(texts) || texts.length === 0) {
      return res
        .status(400)
        .json({ error: 'texts must be a non-empty array' });
    }

    const extractorInstance = await getExtractor();
    const output = await extractorInstance(texts, {
      pooling: 'mean',
      normalize: true,
    });

    const dimension = output.dims[1];
    const data = output.data;
    const embeddings = [];
    for (let i = 0; i < data.length; i += dimension) {
      embeddings.push(Array.from(data.slice(i, i + dimension)).map(Number));
    }

    res.json({
      embeddings,
      model: model || 'Xenova/all-MiniLM-L6-v2',
      dimension,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({
      error: error.message || 'Embedding generation failed',
    });
  }
}

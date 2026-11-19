import {GoogleGenAI, ThinkingLevel, Type} from '@google/genai';
import express from 'express';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer as createViteServer} from 'vite';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT) || 3000;

interface BoxPayload {
  id: number;
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
}

interface ClassifyRequestBody {
  imageBase64?: string;
  mimeType?: string;
  boxes?: BoxPayload[];
}

function createGeminiClient(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) return null;
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

async function startServer() {
  const app = express();
  app.use(express.json({limit: '6mb'}));

  app.post('/api/classify-gender', async (req, res) => {
    const ai = createGeminiClient();
    if (!ai) {
      res.status(503).json({
        error:
          'GEMINI_API_KEY is not configured. Add it in Settings > Secrets to enable live camera gender classification.',
      });
      return;
    }

    const body = (req.body ?? {}) as ClassifyRequestBody;
    const boxes = Array.isArray(body.boxes) ? body.boxes.slice(0, 12) : [];
    if (!body.imageBase64 || boxes.length === 0) {
      res.status(400).json({error: 'Expected imageBase64 and non-empty boxes array.'});
      return;
    }

    const mimeType = body.mimeType || 'image/jpeg';
    const boxDescriptions = boxes
      .map(
        (b) =>
          `id=${b.id}: xmin=${b.xmin.toFixed(3)}, ymin=${b.ymin.toFixed(3)}, xmax=${b.xmax.toFixed(3)}, ymax=${b.ymax.toFixed(3)}`,
      )
      .join('\n');

    const prompt = `Classify the perceived gender presentation ("female", "male", or "unknown") of each person inside the following normalized [0..1] bounding boxes (origin top-left) in this camera frame:
${boxDescriptions}

Use "unknown" if the person is too blurry, occluded, facing away with no clear cues, or ambiguous. Return one entry per requested id.`;

    try {
      const response = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: {
          parts: [
            {
              inlineData: {
                mimeType,
                data: body.imageBase64,
              },
            },
            {text: prompt},
          ],
        },
        config: {
          thinkingConfig: {thinkingLevel: ThinkingLevel.LOW},
          systemInstruction:
            'You are a fast, precise computer-vision classifier for privacy filtering. Classify each numbered person bounding box as female, male, or unknown.',
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            required: ['classifications'],
            properties: {
              classifications: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  required: ['id', 'gender', 'confidence'],
                  properties: {
                    id: {type: Type.INTEGER},
                    gender: {type: Type.STRING, enum: ['female', 'male', 'unknown']},
                    confidence: {type: Type.NUMBER},
                  },
                },
              },
            },
          },
        },
      });

      const rawText = (response.text ?? '').trim();
      const parsed = rawText ? JSON.parse(rawText) : {classifications: []};
      res.json(parsed);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const status =
        message.includes('429') || message.includes('RESOURCE_EXHAUSTED')
          ? 429
          : message.includes('403') || message.includes('PERMISSION_DENIED')
            ? 403
            : 500;
      res.status(status).json({error: message});
    }
  });

  if (process.env.NODE_ENV === 'production') {
    const distDir = path.join(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('/{*splat}', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  } else {
    const vite = await createViteServer({
      server: {middlewareMode: true},
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Street Privacy Lens server listening on http://0.0.0.0:${PORT}`);
  });
}

void startServer();

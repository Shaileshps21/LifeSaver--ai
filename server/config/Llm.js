// --------------------newer version with more models---------------------------------------
/**
 * Unified LLM Client — Groq (default) + Gemini (fallback)
 * ─────────────────────────────────────────────────────────────────────────────
 * v3 — Enhanced with:
 *   • Cross-provider fallback (Groq → Gemini on failure)
 *   • Exponential backoff retry w/ jitter, honoring provider Retry-After hints
 *   • Per-call token budget control, clamped to each model's real ceiling —
 *     long planning/hierarchy generations no longer silently truncate
 *   • Automatic one-shot continuation at the model's max ceiling when a
 *     response comes back truncated (finish_reason === length / MAX_TOKENS),
 *     instead of handing clipped JSON to the syntax-only repair pass
 *   • Optional native JSON mode (Groq response_format / Gemini responseMimeType)
 *   • Empty/blocked-response detection treated as a retryable failure
 *   • Token/cost capture on every generateText() call
 *   • Prompt version tracking
 *
 * Both providers expose the same unified interface:
 *   clients.pro.generateText(prompt, opts?)   → EnrichedResult
 *   clients.flash.generateText(prompt, opts?) → EnrichedResult
 *   clients.embedding.embed(text)             → number[] | null
 *
 * generateText(prompt, opts):
 *   opts.promptVersion    - string, default 'v1.0.0'
 *   opts.maxOutputTokens  - number, overrides the tier default (clamped to
 *                           the model's documented ceiling — see MODEL_LIMITS)
 *   opts.jsonMode         - boolean, requests the provider's native
 *                           structured-JSON output mode when supported
 *
 * EnrichedResult:
 *   { text, usage: { promptTokens, completionTokens, totalTokens },
 *     provider, model, estimatedCost, promptVersion, truncated }
 *
 * Supported keyTypes: 'gemini' | 'groq'
 */

import { GoogleGenerativeAI } from '@google/generative-ai';
import Groq from 'groq-sdk';
import dotenv from 'dotenv';
dotenv.config();

// ── Model mapping ─────────────────────────────────────────────────────────────
// IMPORTANT — two separate ways a model choice can break this app on Groq:
//   1. Availability: the model catalog varies per account/region — an id
//      not in your account's list 404s as "model_not_found" on every call.
//      Verify with: curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"
//   2. Free-plan TPM (tokens/minute) is often far smaller than the model's
//      documented context window, and a request over that limit 413s as
//      "Request too large" — this app's planning/knowledge prompts commonly
//      run 3-6K tokens, so headroom matters more than raw capability.
//
//   `llama-3.3-70b-versatile` (the previous `pro` default) has been
//   dismantled/deprecated on Groq and now 404s as model_not_found on every
//   call — it has been removed everywhere below. `openai/gpt-oss-120b` is
//   now the `pro` default per explicit instruction, even though it was
//   previously avoided here for exactly this reason: its free-plan TPM cap
//   (~8K, vs. the old default's ~12K) can 413 "Request too large" on the
//   larger knowledge/planning prompts. If that starts happening in practice,
//   it is not a bug — it's this known trade-off; the fix is either a paid
//   Groq tier with higher TPM or swapping `pro` back to a higher-TPM model.
const GROQ_MODELS = {
    pro: 'openai/gpt-oss-120b',   // best quality — planning, prioritization
    flash: 'openai/gpt-oss-20b', // fastest — parsing, quick tasks
}


const GEMINI_MODELS = {
    pro: 'gemini-2.5-pro',
    flash: 'gemini-2.5-flash',
    flashLite: 'gemini-2.5-flash-lite',
    embedding: 'embedding-001',
};

// ── Per-model output/context ceilings (provider-documented) ─────────────────
// Used to clamp caller-supplied `maxOutputTokens` and to pick the retry
// budget when a response comes back truncated. If a provider changes these
// limits, update here — every call site benefits automatically.
const MODEL_LIMITS = {
    'openai/gpt-oss-120b': { maxOutputTokens: 65536, contextWindow: 131072 },
    'openai/gpt-oss-20b': { maxOutputTokens: 65536, contextWindow: 131072 },
    'qwen/qwen3.6-27b': { maxOutputTokens: 131072, contextWindow: 131072 },
    'qwen/qwen3.8-27b': { maxOutputTokens: 131072, contextWindow: 131072 },
    'gemini-2.5-pro': { maxOutputTokens: 65536, contextWindow: 1048576 },
    'gemini-2.5-flash': { maxOutputTokens: 65536, contextWindow: 1048576 },
    'gemini-2.5-flash-lite': { maxOutputTokens: 65536, contextWindow: 1048576 },
};

function modelCeiling(modelName) {
    return MODEL_LIMITS[modelName]?.maxOutputTokens ?? 8192;
}

// ── Human-readable model labels ──────────────────────────────────────────────
// Single source of truth for "what model are we actually using" — the client
// (ApiKeySetup banner) and orchestrator.js SSE messages both read this via
// getModelLabel()/getProviderSummary() instead of hardcoding a model name, so
// changing GROQ_MODELS/GEMINI_MODELS above is enough to update the whole app.
const MODEL_DISPLAY_NAMES = {
    'qwen/qwen3.6-27b': 'Qwen3.6 27B',
    'qwen/qwen3.8-27b': 'Qwen3.8 27B',
    'openai/gpt-oss-20b': 'GPT-OSS 20B',
    'openai/gpt-oss-120b': 'GPT-OSS 120B',
    'gemini-2.5-pro': 'Gemini 2.5 Pro',
    'gemini-3.7-flash': 'Gemini 3.7 Flash',
    'gemini-3.6-flash': 'Gemini 3.6 Flash',
    'gemini-3.5-flash-lite': 'Gemini 3.5 Flash Lite',
    'gemini-2.5-flash-lite': 'Gemini 2.5 Flash Lite',
};

// Fallback for any model id not yet added to MODEL_DISPLAY_NAMES above —
// strips a provider path prefix and title-cases the rest, so a new model
// still gets a readable (if unpolished) label instead of a raw slug.
function humanizeModelId(modelId) {
    const base = modelId.split('/').pop() ?? modelId;
    return base.replace(/[-_]/g, ' ').replace(/\b[a-z]/g, (c) => c.toUpperCase());
}

/**
 * @param {string} modelId
 * @returns {string} human-readable model name, independent of keyType/tier
 */
export function getModelLabelById(modelId) {
    if (!modelId) return '';
    return MODEL_DISPLAY_NAMES[modelId] ?? humanizeModelId(modelId);
}

/**
 * @param {'groq'|'gemini'} keyType
 * @param {'pro'|'flash'} [tier]
 * @returns {string} human-readable model name, e.g. "Llama 3.3 70B"
 */
export function getModelLabel(keyType, tier = 'pro') {
    const modelId = keyType === 'groq' ? GROQ_MODELS[tier] : GEMINI_MODELS[tier];
    if (!modelId) return keyType === 'groq' ? 'Groq' : 'Gemini';
    return getModelLabelById(modelId);
}

/**
 * @param {'groq'|'gemini'} keyType
 * @param {string|null} [modelOverride] - if the user picked a specific model for their key
 * @returns {{ keyType, modelId, modelLabel, flashModelId, flashModelLabel }}
 */
export function getProviderSummary(keyType, modelOverride = null) {
    const models = keyType === 'groq' ? GROQ_MODELS : GEMINI_MODELS;
    const modelId = modelOverride || models.pro;
    return {
        keyType,
        modelId,
        modelLabel: getModelLabelById(modelId),
        flashModelId: models.flash,
        flashModelLabel: getModelLabel(keyType, 'flash'),
    };
}

// ── User-selectable models — what the "Choose a model" dropdown offers ──────
// Deliberately a curated subset of everything Groq/Gemini expose (not every
// model is chat-capable or a sane choice for this app's planning/reasoning
// workload — e.g. embedding-001, whisper, prompt-guard are excluded).
const GROQ_SELECTABLE_MODELS = [
    'qwen/qwen3.6-27b',
    'qwen/qwen3.8-27b',
    'openai/gpt-oss-120b',
    'openai/gpt-oss-20b',
];
const GEMINI_SELECTABLE_MODELS = [
    'gemini-2.5-pro',
    'gemini-3.7-flash',
    'gemini-3.6-flash',
    'gemini-3.5-flash-lite',
    'gemini-2.5-flash-lite',
];

/**
 * @param {'groq'|'gemini'} keyType
 * @returns {Array<{id: string, label: string}>} models the user can pick between
 */
export function getAvailableModels(keyType) {
    const ids = keyType === 'groq' ? GROQ_SELECTABLE_MODELS : GEMINI_SELECTABLE_MODELS;
    return ids.map((id) => ({ id, label: getModelLabelById(id) }));
}

// Per-tier defaults, used when the caller doesn't pass `maxOutputTokens`.
// Bumped up slightly from v3 so planning/hierarchy generations get more
// headroom before ever hitting the truncation/continuation path.
// Deliberately smaller than the model ceiling for the 'flash' tier (quick,
// cheap calls) — the ceiling is still available via the truncation
// auto-continuation path or an explicit override.
const DEFAULT_MAX_TOKENS = { pro: 10240, flash: 5120 };

// ── Cost constants (USD per 1k tokens) ───────────────────────────────────────
// Approximate public pricing as of 2025 — adjust as needed
const COST_PER_1K = {
    'openai/gpt-oss-120b': { input: 0.00015, output: 0.00060 },
    'openai/gpt-oss-20b': { input: 0.000075, output: 0.00030 },
    // qwen/qwen3.6-27b: pricing not yet confirmed — omitted deliberately so
    // estimateCost() logs a warning and reports $0 rather than a fabricated
    // figure. Add real per-1K rates here once confirmed against Groq's pricing page.
    'gemini-2.5-pro': { input: 0.00125, output: 0.00500 },
    'gemini-2.5-flash': { input: 0.000075, output: 0.00030 },
    'gemini-2.5-flash-lite': { input: 0.0000375, output: 0.00015 },
};

function estimateCost(model, promptTokens, completionTokens) {
    const rates = COST_PER_1K[model];
    if (!rates) {
        console.warn(`[LLM] No cost entry for model "${model}" — estimatedCost will read 0. Add it to COST_PER_1K if this model is in regular use.`);
        return 0;
    }
    return (promptTokens / 1000) * rates.input + (completionTokens / 1000) * rates.output;
}

// ── Quota / rate-limit error detection ───────────────────────────────────────
export function isQuotaError(err) {
    const msg = (err?.message || '').toLowerCase();
    const code = err?.status || err?.code || 0;
    return (
        code === 429 ||
        msg.includes('429') ||
        msg.includes('quota') ||
        msg.includes('rate limit') ||
        msg.includes('resource_exhausted') ||
        msg.includes('too many requests') ||
        msg.includes('rate_limit_exceeded')
    );
}

function isAuthError(err) {
    const msg = (err?.message || '').toLowerCase();
    const code = err?.status || err?.code || 0;
    return (
        code === 401 || code === 403 ||
        msg.includes('api key') ||
        msg.includes('invalid_api_key') ||
        msg.includes('authentication') ||
        msg.includes('permission denied') ||
        msg.includes('incorrect api key')
    );
}

/**
 * A misconfigured/unavailable model id (typo, deprecated, or not enabled
 * for this account) fails identically on every attempt — retrying just
 * burns the retry budget and delays surfacing the real problem.
 */
function isInvalidModelError(err) {
    const msg = (err?.message || '').toLowerCase();
    const code = err?.status || err?.code || 0;
    return (
        code === 404 ||
        msg.includes('model_not_found') ||
        (msg.includes('model') && (msg.includes('does not exist') || msg.includes('not found')))
    );
}

/**
 * The request payload itself exceeds the model's per-request/TPM ceiling
 * (Groq 413 "Request too large", or an equivalent size-based 400 from
 * another provider). Retrying the identical payload always fails the same
 * way — unlike a plain rate limit, waiting doesn't help.
 */
function isPayloadTooLargeError(err) {
    const msg = (err?.message || '').toLowerCase();
    const code = err?.status || err?.code || 0;
    return code === 413 || msg.includes('413') || msg.includes('too large') || msg.includes('request too large');
}

/** A response that came back syntactically fine but empty/blocked — worth a retry. */
class EmptyResponseError extends Error {
    constructor(label) {
        super(`${label} returned an empty response (possibly content-filtered)`);
        this.name = 'EmptyResponseError';
    }
}

// ── Retry-After extraction ────────────────────────────────────────────────────
/**
 * Best-effort extraction of a provider-supplied "retry after N ms" hint.
 * Groq/OpenAI-style SDKs surface a `headers` map on the error; Gemini's API
 * sometimes includes a `RetryInfo` with a `retryDelay` like "13s" in
 * `errorDetails`. Falls back to null (caller uses exponential backoff).
 * @param {*} err
 * @returns {number|null} milliseconds to wait, or null if unknown
 */
function getRetryAfterMs(err) {
    try {
        const headerVal = err?.headers?.get?.('retry-after') ?? err?.response?.headers?.get?.('retry-after');
        if (headerVal) {
            const secs = Number(headerVal);
            if (Number.isFinite(secs)) return secs * 1000;
        }
        const retryInfo = err?.errorDetails?.find?.((d) => typeof d?.retryDelay === 'string');
        if (retryInfo?.retryDelay) {
            const secs = parseFloat(retryInfo.retryDelay);
            if (Number.isFinite(secs)) return secs * 1000;
        }
    } catch {
        // best-effort only
    }
    return null;
}

// ── Retry with exponential backoff + jitter ──────────────────────────────────
async function withRetry(fn, { maxAttempts = 3, baseDelayMs = 1000, label = 'LLM' } = {}) {
    let lastErr;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            return await fn(attempt);
        } catch (err) {
            lastErr = err;
            if (isAuthError(err) || isInvalidModelError(err) || isPayloadTooLargeError(err)) throw err; // not retryable — will fail identically every time
            if (attempt < maxAttempts) {
                const retryAfter = getRetryAfterMs(err);
                // Exponential backoff (1s, 2s, 4s...) with +/-25% jitter to avoid
                // synchronized retry storms across concurrent pipeline runs.
                const backoff = baseDelayMs * Math.pow(2, attempt - 1);
                const jittered = backoff * (0.75 + Math.random() * 0.5);
                const delay = retryAfter ?? jittered;
                console.warn(`[${label}] Attempt ${attempt} failed (${err.message?.slice(0, 60)}). Retrying in ${Math.round(delay)}ms...`);
                await new Promise((r) => setTimeout(r, delay));
            }
        }
    }
    throw lastErr;
}

// ── Rate-limit header extraction ─────────────────────────────────────────────
// Groq (OpenAI-compatible) returns these on every chat/completions response —
// see https://console.groq.com/docs/rate-limits. Gemini's Generative Language
// API has no equivalent header, so this is Groq-only.
function extractGroqRateLimit(headers) {
    if (!headers?.get) return null;
    const num = (name) => {
        const v = headers.get(name);
        const n = v === null ? NaN : Number(v);
        return Number.isFinite(n) ? n : null;
    };
    const limitRequests = num('x-ratelimit-limit-requests');
    const remainingRequests = num('x-ratelimit-remaining-requests');
    const limitTokens = num('x-ratelimit-limit-tokens');
    const remainingTokens = num('x-ratelimit-remaining-tokens');
    if (limitRequests === null && limitTokens === null) return null; // nothing usable
    return {
        limitRequests,
        remainingRequests,
        limitTokens,
        remainingTokens,
        resetRequests: headers.get('x-ratelimit-reset-requests') ?? null,
        resetTokens: headers.get('x-ratelimit-reset-tokens') ?? null,
    };
}

// ── Groq text wrapper ─────────────────────────────────────────────────────────
// Exported (alongside wrapGeminiText below) so the Groq→Gemini fallback wiring
// itself — the exact bug class this comment sits next to — can be exercised
// directly in tests without needing real API keys or live network calls.
export function wrapGroqText(groqClient, modelName, temperature = 0.3, defaultMaxTokens = 8192, fallbackFn = null) {
    const ceiling = modelCeiling(modelName);

    return {
        async generateText(prompt, { promptVersion = 'v1.0.0', maxOutputTokens, jsonMode = false } = {}) {
            const budget = Math.min(maxOutputTokens ?? defaultMaxTokens, ceiling);

            const call = async (tokenBudget) => withRetry(async () => {
                const created = groqClient.chat.completions.create({
                    model: modelName,
                    messages: [{ role: 'user', content: prompt }],
                    temperature,
                    max_tokens: tokenBudget,
                    ...(jsonMode ? { response_format: { type: 'json_object' } } : {}),
                });
                // .withResponse() (instead of a plain await) also surfaces the raw
                // Fetch Response so we can read Groq's x-ratelimit-* headers — the
                // parsed body alone never exposes them. Only the real groq-sdk
                // APIPromise has this method; fake clients used in tests return a
                // plain Promise, so fall back to a bare await for those.
                let res, response = null;
                if (typeof created.withResponse === 'function') {
                    ({ data: res, response } = await created.withResponse());
                } else {
                    res = await created;
                }
                const text = res.choices[0]?.message?.content || '';
                if (!text.trim()) throw new EmptyResponseError(`Groq:${modelName}`);
                return { res, text, rateLimit: extractGroqRateLimit(response?.headers) };
            }, { label: `Groq:${modelName}` });

            try {
                let { res, text, rateLimit } = await call(budget);
                let truncated = res.choices[0]?.finish_reason === 'length';

                // One-shot continuation at the model's real ceiling — a
                // truncated response is not "malformed JSON", it's missing
                // data, and no amount of syntax repair recovers that.
                if (truncated && budget < ceiling) {
                    console.warn(`[Groq:${modelName}] Response truncated at ${budget} tokens — retrying once at the ${ceiling}-token ceiling.`);
                    ({ res, text, rateLimit } = await call(ceiling));
                    truncated = res.choices[0]?.finish_reason === 'length';
                }

                const promptTokens = res.usage?.prompt_tokens ?? 0;
                const completionTokens = res.usage?.completion_tokens ?? 0;
                return {
                    text,
                    usage: { promptTokens, completionTokens, totalTokens: promptTokens + completionTokens },
                    provider: 'groq',
                    model: modelName,
                    estimatedCost: estimateCost(modelName, promptTokens, completionTokens),
                    promptVersion,
                    truncated,
                    rateLimit,
                };
            } catch (err) {
                if (fallbackFn && !isAuthError(err)) {
                    console.warn(`[Groq] Falling back to Gemini: ${err.message?.slice(0, 80)}`);
                    // fallbackFn is a wrapGeminiText() wrapper object ({ generateText }),
                    // not a bare function — calling it directly throws "fallbackFn is
                    // not a function" the moment a real fallback is ever needed.
                    return fallbackFn.generateText(prompt, { promptVersion, maxOutputTokens, jsonMode });
                }
                throw err;
            }
        },
    };
}

// ── Gemini text wrapper ───────────────────────────────────────────────────────
export function wrapGeminiText(model, modelName, baseGenerationConfig = {}) {
    const ceiling = modelCeiling(modelName);

    return {
        async generateText(prompt, { promptVersion = 'v1.0.0', maxOutputTokens, jsonMode = false } = {}) {
            const budget = Math.min(maxOutputTokens ?? baseGenerationConfig.maxOutputTokens ?? 8192, ceiling);

            const call = async (tokenBudget) => withRetry(async () => {
                const result = await model.generateContent({
                    contents: [{ role: 'user', parts: [{ text: prompt }] }],
                    generationConfig: {
                        ...baseGenerationConfig,
                        maxOutputTokens: tokenBudget,
                        ...(jsonMode ? { responseMimeType: 'application/json' } : {}),
                    },
                });
                const text = result.response.text();
                if (!text.trim()) throw new EmptyResponseError(`Gemini:${modelName}`);
                return { result, text };
            }, { label: `Gemini:${modelName}` });

            let { result, text } = await call(budget);
            let truncated = result.response.candidates?.[0]?.finishReason === 'MAX_TOKENS';

            if (truncated && budget < ceiling) {
                console.warn(`[Gemini:${modelName}] Response truncated at ${budget} tokens — retrying once at the ${ceiling}-token ceiling.`);
                ({ result, text } = await call(ceiling));
                truncated = result.response.candidates?.[0]?.finishReason === 'MAX_TOKENS';
            }

            const meta = result.response.usageMetadata ?? {};
            const promptTokens = meta.promptTokenCount ?? 0;
            const completionTokens = meta.candidatesTokenCount ?? 0;
            return {
                text,
                usage: { promptTokens, completionTokens, totalTokens: promptTokens + completionTokens },
                provider: 'gemini',
                model: modelName,
                estimatedCost: estimateCost(modelName, promptTokens, completionTokens),
                promptVersion,
                truncated,
            };
        },
    };
}

// ── Gemini embedding wrapper ──────────────────────────────────────────────────
function wrapGeminiEmbedding(model) {
    return {
        async embed(text) {
            const result = await model.embedContent(text);
            return result.embedding.values;
        },
    };
}

// Groq has no embedding API — returns null so RAG gracefully skips
const noEmbedding = {
    async embed() { return null; },
};

// ── Client factory ─────────────────────────────────────────────────────────────
/**
 * Creates unified LLM clients for a given provider + key.
 * @param {'groq'|'gemini'} keyType
 * @param {string} apiKey
 * @param {object} [fallbackKeys] - optional { gemini: key } for cross-provider fallback
 * @param {string|null} [proModelOverride] - user-chosen model for the 'pro' tier
 *   (from getAvailableModels(keyType)); ignored (with a warning) if it isn't
 *   one of the curated selectable ids, so an old/invalid saved choice can
 *   never silently break every call.
 */
export function createClients(keyType, apiKey, fallbackKeys = {}, proModelOverride = null) {
    const selectable = getAvailableModels(keyType).map((m) => m.id);
    if (proModelOverride && !selectable.includes(proModelOverride)) {
        console.warn(`[LLM] Ignoring unknown model override "${proModelOverride}" for ${keyType} — falling back to the default.`);
        proModelOverride = null;
    }
    const proModel = proModelOverride || (keyType === 'groq' ? GROQ_MODELS.pro : GEMINI_MODELS.pro);

    if (keyType === 'groq') {
        const groq = new Groq({ apiKey });

        // Build Gemini fallback clients if a Gemini key is available
        let geminiFallbackPro = null;
        let geminiFallbackFlash = null;
        if (fallbackKeys.gemini) {
            const fallbackGenAI = new GoogleGenerativeAI(fallbackKeys.gemini);
            geminiFallbackPro = wrapGeminiText(
                fallbackGenAI.getGenerativeModel({ model: GEMINI_MODELS.pro }),
                GEMINI_MODELS.pro,
                { temperature: 0.3, topP: 0.8, maxOutputTokens: DEFAULT_MAX_TOKENS.pro },
            );
            geminiFallbackFlash = wrapGeminiText(
                fallbackGenAI.getGenerativeModel({ model: GEMINI_MODELS.flash }),
                GEMINI_MODELS.flash,
                { temperature: 0.1, topP: 0.8, maxOutputTokens: DEFAULT_MAX_TOKENS.flash },
            );
        }

        return {
            keyType: 'groq',
            modelId: proModel,
            modelLabel: getModelLabelById(proModel),
            pro: wrapGroqText(groq, proModel, 0.3, DEFAULT_MAX_TOKENS.pro, geminiFallbackPro),
            flash: wrapGroqText(groq, GROQ_MODELS.flash, 0.1, DEFAULT_MAX_TOKENS.flash, geminiFallbackFlash),
            embedding: noEmbedding,
        };
    }

    // Gemini
    const genAI = new GoogleGenerativeAI(apiKey);
    return {
        keyType: 'gemini',
        modelId: proModel,
        modelLabel: getModelLabelById(proModel),
        pro: wrapGeminiText(
            genAI.getGenerativeModel({ model: proModel }),
            proModel,
            { temperature: 0.3, topP: 0.8, maxOutputTokens: DEFAULT_MAX_TOKENS.pro },
        ),
        flash: wrapGeminiText(
            genAI.getGenerativeModel({ model: GEMINI_MODELS.flash }),
            GEMINI_MODELS.flash,
            { temperature: 0.1, topP: 0.8, maxOutputTokens: DEFAULT_MAX_TOKENS.flash },
        ),
        embedding: wrapGeminiEmbedding(
            genAI.getGenerativeModel({ model: GEMINI_MODELS.embedding })
        ),
    };
}

// ── Server default clients (from .env) ───────────────────────────────────────
// Groq is checked FIRST — it is the preferred default.
function buildDefaultClients() {
    const fallbackKeys = process.env.GEMINI_API_KEY ? { gemini: process.env.GEMINI_API_KEY } : {};
    if (process.env.GROQ_API_KEY) {
        console.log(`[LLM] Default provider: Groq (${GROQ_MODELS.pro})`);
        return createClients('groq', process.env.GROQ_API_KEY, fallbackKeys);
    }
    if (process.env.GEMINI_API_KEY) {
        console.log(`[LLM] Default provider: Gemini (${GEMINI_MODELS.pro})`);
        return createClients('gemini', process.env.GEMINI_API_KEY);
    }
    console.warn('[LLM] ⚠️  No default API key found. Users must supply their own.');
    return null;
}

export const defaultClients = buildDefaultClients();

// ── Live key validation ───────────────────────────────────────────────────────
/**
 * @param {'groq'|'gemini'} keyType
 * @param {string} apiKey
 * @param {string|null} [model] - if the user picked a specific model, also
 *   verify THAT model responds (catches an unavailable/mistyped model id
 *   immediately at save time instead of failing later mid-pipeline).
 */
export async function validateApiKey(keyType, apiKey, model = null) {
    let clients;
    try {
        clients = createClients(keyType, apiKey, {}, model);
    } catch (err) {
        return { valid: false, error: `Connection failed: ${(err.message || '').slice(0, 100)}` };
    }
    // The flash-tier call always uses the fixed flash model for this provider
    // (never the user's `model` override — that only applies to the pro
    // tier), so error messages must attribute a flash failure to THAT model,
    // not to `model` — which is null whenever the user picked "Recommended
    // default", and previously produced a literal `"null" isn't available...`
    // message that misattributed the failure and hid the real cause.
    const flashModelId = keyType === 'groq' ? GROQ_MODELS.flash : GEMINI_MODELS.flash;

    try {
        // Flash call proves the key itself authenticates, regardless of model choice.
        const flashResult = await clients.flash.generateText('Reply with the single word: valid');
        const flashText = typeof flashResult === 'string' ? flashResult : flashResult.text;
        if (!flashText) throw new Error('Empty response');
    } catch (err) {
        return buildValidationError(err, flashModelId);
    }

    if (model) {
        try {
            const proResult = await clients.pro.generateText('Reply with the single word: valid');
            const proText = typeof proResult === 'string' ? proResult : proResult.text;
            if (!proText) throw new Error(`Empty response from ${model}`);
        } catch (err) {
            // clients.modelId is the ACTUALLY resolved pro model — createClients()
            // silently falls back to the provider default if `model` wasn't a
            // recognized selectable id, so this can legitimately differ from
            // the raw `model` argument, and is the more accurate thing to report.
            return buildValidationError(err, clients.modelId ?? model);
        }
    }

    return { valid: true };
}

function buildValidationError(err, failedModelId) {
    const msg = err.message || '';
    if (isInvalidModelError(err)) {
        return { valid: false, error: `"${getModelLabelById(failedModelId) || failedModelId}" isn't available on this account/key. Pick a different model.` };
    }
    if (isQuotaError(err)) return { valid: true }; // quota = key is real, just busy
    if (isAuthError(err)) {
        return { valid: false, error: 'Invalid API key — please check and try again.' };
    }
    return { valid: false, error: `Connection failed: ${msg.slice(0, 100)}` };
}

// ── JSON parse helper ─────────────────────────────────────────────────────────
/**
 * Strips markdown code fences and parses JSON.
 * @param {string} text
 * @returns {object}
 */
export function parseJSON(text) {
    const cleaned = text
        .replace(/```json\n?/gi, '')
        .replace(/```\n?/g, '')
        .trim();
    return JSON.parse(cleaned);
}

/**
 * Parse JSON with an automatic LLM repair fallback.
 * If the first parse fails, sends a repair prompt to the flash model.
 *
 * `wasTruncated` should be set to the `truncated` flag from the
 * EnrichedResult this text came from, when available. Truncated output is
 * missing data, not merely malformed — the repair prompt asks the model to
 * complete the structure instead of just re-punctuating it, which produces
 * far more accurate recoveries than blind syntax repair.
 *
 * @param {string} text            - raw LLM output
 * @param {object} flashClient     - clients.flash (for repair)
 * @param {boolean} [wasTruncated] - true if the source response hit its token ceiling
 * @returns {object}               - parsed JSON
 */
export async function parseJSONWithRepair(text, flashClient, wasTruncated = false) {
    try {
        return parseJSON(text);
    } catch (firstErr) {
        console.warn(`[LLM] JSON parse failed${wasTruncated ? ' (source response was truncated)' : ''}, attempting repair...`);
        try {
            const instruction = wasTruncated
                ? 'This JSON was cut off before it finished generating. Complete it into valid, well-formed JSON that preserves all the data already present — close every open object/array sensibly. Return ONLY the JSON, no markdown, no explanation:'
                : 'Fix this invalid JSON and return ONLY valid JSON with no markdown, no explanation:';
            const repairPrompt = `${instruction}\n\n${text.slice(0, 3000)}`;
            // For a truncated source, ask for as much room as the repair
            // model actually has (generateText clamps this to its real
            // ceiling) — a small default budget here would just truncate
            // the "repair" too, before it even reaches the original cutoff.
            const repairBudget = wasTruncated ? 65536 : 4096;

            let repairResult;
            try {
                // Prefer the provider's native JSON mode when available — it
                // meaningfully cuts down on repair-of-a-repair loops.
                repairResult = await flashClient.generateText(repairPrompt, { maxOutputTokens: repairBudget, jsonMode: true });
            } catch (jsonModeErr) {
                // Not every model/provider combination supports structured
                // output (e.g. some Groq models reject `response_format`) —
                // fall back to plain prompting rather than losing the repair
                // pass entirely.
                console.warn(`[LLM] JSON-mode repair unavailable (${jsonModeErr.message?.slice(0, 80)}), retrying without it...`);
                repairResult = await flashClient.generateText(repairPrompt, { maxOutputTokens: repairBudget });
            }
            const repairText = typeof repairResult === 'string' ? repairResult : repairResult.text;
            return parseJSON(repairText);
        } catch (repairErr) {
            throw new Error(`JSON repair failed: ${firstErr.message} | repair: ${repairErr.message}`);
        }
    }
}

// ── Text extraction helper ────────────────────────────────────────────────────
/**
 * Extract plain text from either a legacy string response or enriched result.
 * Keeps backward compatibility with agents that call clients.flash.generateText
 * and expect a plain string.
 * @param {string|object} result
 * @returns {string}
 */
export function extractText(result) {
    if (typeof result === 'string') return result;
    return result?.text ?? '';
}

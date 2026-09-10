/**
 * Llm.rateLimit.test.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Groq exposes remaining requests/tokens for the key via x-ratelimit-* response
 * headers (https://console.groq.com/docs/rate-limits). wrapGroqText() only sees
 * those headers when it reads the response via .withResponse() instead of a
 * plain await, and must still work against fakes that don't implement it
 * (Llm.fallback.test.js's fakes, and any older/incompatible groq-sdk).
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { wrapGroqText } from './Llm.js';

function makeHeaders(values) {
    return { get: (name) => values[name] ?? null };
}

function makeGroqClientWithHeaders(headerValues) {
    return {
        chat: {
            completions: {
                create: () => ({
                    withResponse: async () => ({
                        data: {
                            choices: [{ message: { content: 'hello' }, finish_reason: 'stop' }],
                            usage: { prompt_tokens: 3, completion_tokens: 2 },
                        },
                        response: { headers: makeHeaders(headerValues) },
                    }),
                }),
            },
        },
    };
}

function makePlainGroqClient() {
    return {
        chat: {
            completions: {
                create: async () => ({
                    choices: [{ message: { content: 'hello' }, finish_reason: 'stop' }],
                    usage: { prompt_tokens: 3, completion_tokens: 2 },
                }),
            },
        },
    };
}

test('wrapGroqText captures remaining requests/tokens from x-ratelimit-* headers', async () => {
    const client = makeGroqClientWithHeaders({
        'x-ratelimit-limit-requests': '50',
        'x-ratelimit-remaining-requests': '42',
        'x-ratelimit-limit-tokens': '12000',
        'x-ratelimit-remaining-tokens': '8234',
        'x-ratelimit-reset-requests': '2s',
        'x-ratelimit-reset-tokens': '1m30s',
    });
    const groq = wrapGroqText(client, 'bogus-model');

    const result = await groq.generateText('a prompt');

    assert.deepEqual(result.rateLimit, {
        limitRequests: 50,
        remainingRequests: 42,
        limitTokens: 12000,
        remainingTokens: 8234,
        resetRequests: '2s',
        resetTokens: '1m30s',
    });
});

test('wrapGroqText reports rateLimit: null when the client exposes no .withResponse()', async () => {
    const groq = wrapGroqText(makePlainGroqClient(), 'bogus-model');
    const result = await groq.generateText('a prompt');
    assert.equal(result.rateLimit, null);
});

test('wrapGroqText tracks the latest quota on `lastRateLimit`, independent of what the caller does with the return value', async () => {
    // This is the mechanism agentRunner.js actually reads — most agents parse
    // the LLM response themselves inside agentFn and return their own domain
    // object, discarding generateText()'s return value entirely. `lastRateLimit`
    // must still reflect the most recent call regardless.
    const client = makeGroqClientWithHeaders({
        'x-ratelimit-limit-requests': '50',
        'x-ratelimit-remaining-requests': '41',
        'x-ratelimit-limit-tokens': '12000',
        'x-ratelimit-remaining-tokens': '8000',
    });
    const groq = wrapGroqText(client, 'bogus-model');

    assert.equal(groq.lastRateLimit, null);
    await groq.generateText('a prompt'); // caller ignores the return value entirely
    assert.equal(groq.lastRateLimit.remainingRequests, 41);
    assert.equal(typeof groq.lastRateLimit.capturedAt, 'number');
});

test('wrapGroqText leaves `lastRateLimit` untouched when a call yields no headers', async () => {
    const groq = wrapGroqText(makePlainGroqClient(), 'bogus-model');
    await groq.generateText('a prompt');
    assert.equal(groq.lastRateLimit, null);
});

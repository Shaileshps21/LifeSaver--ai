/**
 * shared/agentRunner.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Standardized 7-step agent execution wrapper.
 *
 * Every agent in the pipeline goes through this runner:
 *   1. Invoke   — call the agent function → raw LLM response
 *   2. Parse    — extract JSON from response (with repair fallback)
 *   3. Validate — check against registered schema
 *   4. Evaluate — run quality evaluator (if provided); retry if score < 80
 *   5. Reason   — ensure reasoning block is present
 *   6. Log      — record metrics in observability log
 *   7. Write    — store output in PlanningContext namespace
 */

import { parseJSONWithRepair, extractText } from '../../config/Llm.js';
import { validateAgentOutput, validateReasoningBlock } from './validator.js';
import { createAgentLog, completeAgentLog, attachToContext } from './logger.js';
import { writeNamespace } from '../contextManager.js';

/**
 * Run an agent through the standardized pipeline.
 *
 * @param {object} opts
 * @param {string}   opts.agentName      - human label e.g. 'planning_agent'
 * @param {string}   opts.sseAgentName   - name used in SSE events e.g. 'planning'
 * @param {function} opts.agentFn        - async (context, clients) → { output, reasoning? }
 * @param {object}   opts.context        - PlanningContext
 * @param {object}   opts.clients        - LLM clients { pro, flash, embedding }
 * @param {string}   opts.namespace      - context namespace to write output into
 * @param {string}   [opts.schemaVersion] - schema version string e.g. '1.0.0'
 * @param {string}   [opts.promptVersion] - prompt version string e.g. 'v1.0.0'
 * @param {function} [opts.evaluator]    - (output) → { score: number, issues: string[] }
 * @param {object}   [opts.eventBus]     - optional eventBus instance for lifecycle events
 * @param {function} [opts.sseEmit]      - optional (agentName, status, message, data) → void
 * @param {number}   [opts.maxRetries]   - max quality-retry attempts (default: 2)
 * @returns {Promise<object>} the parsed and validated output
 */
export async function runAgent({
    agentName,
    sseAgentName,
    agentFn,
    context,
    clients,
    namespace,
    schemaVersion = '1.0.0',
    promptVersion = 'v1.0.0',
    evaluator = null,
    eventBus = null,
    sseEmit = null,
    maxRetries = 2,
}) {
    const log = createAgentLog(agentName, promptVersion);
    let retries = 0;
    let lastErr;

    // Fire agent:start
    eventBus?.agentStart(agentName, `${agentName} started`);
    sseEmit?.(sseAgentName ?? agentName, 'thinking', `${agentName} is working...`, null);

    for (let attempt = 1; attempt <= maxRetries + 1; attempt++) {
        try {
            // ── Step 1: Invoke ────────────────────────────────────────────────
            const rawResult = await agentFn(context, clients, attempt);

            // ── Step 2: Parse ─────────────────────────────────────────────────
            let parsed;
            if (typeof rawResult === 'object' && rawResult !== null && !rawResult.text) {
                // Agent already returned a parsed object
                parsed = rawResult;
            } else {
                const rawText = extractText(rawResult);
                parsed = await parseJSONWithRepair(rawText, clients.flash);
            }

            // Capture token usage from the enriched result if available
            if (rawResult?.usage) {
                log.tokens = {
                    prompt: rawResult.usage.promptTokens ?? 0,
                    completion: rawResult.usage.completionTokens ?? 0,
                    total: rawResult.usage.totalTokens ?? 0,
                };
                log.estimatedCost = rawResult.estimatedCost ?? 0;
                log.provider = rawResult.provider ?? null;
                log.model = rawResult.model ?? null;
            }
            // Groq-only: surfaces the x-ratelimit-* headers from this call so the
            // live SSE trace can show remaining requests/tokens for the key in use.
            const rateLimit = rawResult?.rateLimit ?? null;

            // ── Step 3: Validate ──────────────────────────────────────────────
            const validation = validateAgentOutput(agentName, schemaVersion, parsed);
            if (!validation.valid) {
                const errMsg = `Schema validation failed for ${agentName}: ${validation.errors.join('; ')}`;
                if (attempt <= maxRetries) {
                    console.warn(`[AgentRunner] ${errMsg} — retrying (attempt ${attempt + 1})`);
                    retries++;
                    continue;
                }
                // Allow soft-fail with warnings on final attempt for non-critical agents
                console.warn(`[AgentRunner] ${errMsg} — proceeding with warnings`);
                if (context.metadata?.warnings) {
                    context.metadata.warnings.push(`${agentName}: schema validation warnings: ${validation.warnings.join('; ')}`);
                }
            }

            // ── Step 4: Evaluate quality ──────────────────────────────────────
            if (evaluator) {
                const { score, issues } = evaluator(parsed);
                if (score < 80 && attempt <= maxRetries) {
                    console.warn(`[AgentRunner] ${agentName} quality score ${score}/100 — retrying with feedback`);
                    // Attach review feedback to context for the next attempt
                    if (!context._agentFeedback) context._agentFeedback = {};
                    context._agentFeedback[agentName] = { score, issues };
                    retries++;
                    continue;
                }
            }

            // ── Step 5: Ensure reasoning block ────────────────────────────────
            if (!parsed.reasoning) {
                parsed.reasoning = {
                    confidence: 0.8,
                    assumptions: [],
                    warnings: [],
                    alternatives: [],
                    promptVersion,
                };
            } else {
                parsed.reasoning.promptVersion = parsed.reasoning.promptVersion ?? promptVersion;
            }

            // ── Step 6: Log ───────────────────────────────────────────────────
            completeAgentLog(log, {
                confidence: parsed.reasoning?.confidence ?? null,
                retries,
                status: retries > 0 ? 'retried' : 'success',
                provider: log.provider,
                model: log.model,
            });
            attachToContext(context, log);

            // ── Step 7: Write to context ──────────────────────────────────────
            writeNamespace(context, namespace, parsed);

            // Clean up temp feedback
            if (context._agentFeedback?.[agentName]) {
                delete context._agentFeedback[agentName];
            }

            // Fire agent:complete
            eventBus?.agentComplete(agentName, `${agentName} completed`, { namespace });
            sseEmit?.(sseAgentName ?? agentName, 'done', `${agentName} completed successfully`, rateLimit ? { rateLimit } : null);

            return parsed;

        } catch (err) {
            lastErr = err;
            if (attempt <= maxRetries) {
                console.warn(`[AgentRunner] ${agentName} attempt ${attempt} error: ${err.message?.slice(0, 80)} — retrying`);
                retries++;
            }
        }
    }

    // All attempts exhausted
    completeAgentLog(log, {
        retries,
        status: 'failed',
        errors: [lastErr?.message ?? 'Unknown error'],
    });
    attachToContext(context, log);

    eventBus?.agentError(agentName, lastErr, `${agentName} failed after ${retries} retries`);
    sseEmit?.(sseAgentName ?? agentName, 'error', `${agentName} failed: ${lastErr?.message?.slice(0, 80)}`, null);

    throw lastErr;
}

import { existsSync, mkdirSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AuthStorage } from "../src/core/auth-storage.js";
import { createAgentSession } from "../src/core/sdk.js";
import { SessionManager } from "../src/core/session-manager.js";
import { SettingsManager } from "../src/core/settings-manager.js";
import type { Skill } from "../src/core/skills.js";
import { createSyntheticSourceInfo } from "../src/core/source-info.js";
import { getCodingAgentFixtureModel } from "./fixture-models.js";
import { createTestResourceLoader } from "./utilities.js";

describe("createAgentSession session manager defaults", () => {
	let tempDir: string;
	let cwd: string;
	let agentDir: string;

	beforeEach(() => {
		tempDir = join(tmpdir(), `pi-sdk-session-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
		cwd = join(tempDir, "project");
		agentDir = join(tempDir, "agent");
		mkdirSync(cwd, { recursive: true });
		mkdirSync(agentDir, { recursive: true });
	});

	afterEach(() => {
		vi.unstubAllGlobals();
		vi.unstubAllEnvs();
		vi.useRealTimers();
		if (tempDir && existsSync(tempDir)) {
			rmSync(tempDir, { recursive: true, force: true });
		}
	});

	it.each(["default persisted path", "explicit manager"])("uses the %s for session storage", async (storage) => {
		const model = getCodingAgentFixtureModel("anthropic", "claude-sonnet-4-5");
		expect(model).toBeTruthy();
		const sessionManager = storage === "explicit manager" ? SessionManager.inMemory(cwd) : undefined;
		const { session } = await createAgentSession({ cwd, agentDir, model, sessionManager });
		try {
			if (sessionManager) {
				expect(session.sessionManager).toBe(sessionManager);
				expect(session.sessionManager.isPersisted()).toBe(false);
			} else {
				const expectedSessionDir = join(agentDir, "sessions");
				expect(session.sessionManager.getSessionDir()).toBe(expectedSessionDir);
				expect(session.sessionManager.getSessionFile()?.startsWith(`${expectedSessionDir}/`)).toBe(true);
			}
		} finally {
			session.dispose();
		}
	});

	it.each([
		{ label: "configured timeout", timeoutMs: undefined, expectedMs: 2000 },
		{ label: "request override", timeoutMs: 1000, expectedMs: 1000 },
	])("#1232 forwards the $label from the SDK to the Codex watchdog", async ({ timeoutMs, expectedMs }) => {
		vi.stubEnv("PI_OFFLINE", "1");
		const model = getCodingAgentFixtureModel("openai-codex", "gpt-5.5");
		const payload = Buffer.from(JSON.stringify({ "https://api.openai.com/auth": { chatgpt_account_id: "test" } }));
		const authStorage = AuthStorage.inMemory();
		authStorage.setRuntimeApiKey(model.provider, `test.${payload.toString("base64url")}.test`);
		const { session } = await createAgentSession({
			cwd,
			agentDir,
			model,
			authStorage,
			settingsManager: SettingsManager.inMemory({ retry: { provider: { timeoutMs: 2000 } } }),
			sessionManager: SessionManager.inMemory(cwd),
			resourceLoader: createTestResourceLoader(),
			noTools: "all",
		});
		const controller = new AbortController();
		const cancel = vi.fn();
		vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"] });
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => new Response(new ReadableStream({ cancel }))),
		);
		let stream: Awaited<ReturnType<typeof session.agent.streamFn>> | undefined;
		try {
			stream = await session.agent.streamFn(
				model,
				{ messages: [] },
				{
					transport: "sse",
					timeoutMs,
					signal: controller.signal,
				},
			);
			// The start event proves response headers arrived and the open body is being consumed.
			expect((await stream[Symbol.asyncIterator]().next()).value?.type).toBe("start");
			await vi.advanceTimersByTimeAsync(expectedMs - 1);
			expect(cancel).not.toHaveBeenCalled();
			await vi.advanceTimersByTimeAsync(1);
			expect(cancel).toHaveBeenCalledOnce();
			const result = await stream.result();
			expect(result.stopReason).toBe("error");
			expect(result.diagnostics).toContainEqual(
				expect.objectContaining({
					type: "provider_stream_timeout",
					details: expect.objectContaining({ transport: "sse", timeoutMs: expectedMs }),
				}),
			);
		} finally {
			controller.abort();
			await stream?.result();
			session.dispose();
		}
	});

	it("derives cwd from an explicit sessionManager when cwd is omitted", async () => {
		const model = getCodingAgentFixtureModel("anthropic", "claude-sonnet-4-5");
		expect(model).toBeTruthy();

		const sessionCwd = join(tempDir, "session-project");
		mkdirSync(sessionCwd, { recursive: true });
		const sessionManager = SessionManager.inMemory(sessionCwd);
		const { session } = await createAgentSession({
			agentDir,
			model: model!,
			sessionManager,
			tools: ["ipython"],
		});

		expect(session.sessionManager).toBe(sessionManager);
		expect(session.systemPrompt).toContain(`Working directory: ${sessionCwd}`);

		const ipythonTool = session.agent.state.tools.find((tool) => tool.name === "ipython");
		expect(ipythonTool).toBeTruthy();
		const result = await ipythonTool!.execute("test", { code: "import os\nprint(os.getcwd())" });
		const output = result.content
			.filter((item): item is { type: "text"; text: string } => item.type === "text")
			.map((item) => item.text)
			.join("");

		expect(realpathSync(output.trim())).toBe(realpathSync(sessionCwd));

		session.dispose();
	}, 120_000);

	describe("createAgentSession skills option wiring", () => {
		it("discovers skills from agentDir by default", async () => {
			const skillDir = join(agentDir, "skills", "test-skill");
			mkdirSync(skillDir, { recursive: true });
			writeFileSync(
				join(skillDir, "SKILL.md"),
				"---\nname: test-skill\ndescription: A test skill for SDK tests.\n---\n\n# Test Skill\n",
			);

			const { session } = await createAgentSession({
				cwd: agentDir,
				agentDir,
				sessionManager: SessionManager.inMemory(),
			});

			expect(session.resourceLoader.getSkills().skills.some((s) => s.name === "test-skill")).toBe(true);
			session.dispose();
		});

		it.each([
			["no skills (--no-skills)", [] as Skill[]],
			[
				"skills supplied by the loader",
				[
					{
						name: "custom-skill",
						description: "A custom skill",
						filePath: "/fake/path/SKILL.md",
						baseDir: "/fake/path",
						sourceInfo: createSyntheticSourceInfo("/fake/path/SKILL.md", { source: "sdk" }),
						disableModelInvocation: false,
						kind: "markdown" as const,
					},
				] as Skill[],
			],
		])("passes through an explicit resource loader with %s", async (_label, skills) => {
			const { session } = await createAgentSession({
				cwd: agentDir,
				agentDir,
				sessionManager: SessionManager.inMemory(),
				resourceLoader: createTestResourceLoader({ skills }),
			});

			expect(session.resourceLoader.getSkills().skills).toEqual(skills);
			expect(session.resourceLoader.getSkills().diagnostics).toEqual([]);
			session.dispose();
		});
	});
});

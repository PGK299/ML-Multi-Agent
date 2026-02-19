# ML-Multi-Agent

Multi-Agent: The Historical Court

An example Multi-Agent system built with Google ADK that simulates a "Historical Court": multiple agents research, debate, and synthesize balanced reports about a historical figure or event.

**System Architecture**

The application follows a 4-step flow using Sequential, Parallel, and Loop processing.

- **Step 1 — The Inquiry (Sequential)**
	- **Agent:** `court_clerk` (root)
	- **Role:** Greets the user and asks for the target historical figure or event.
	- **State:** Saves the user's input to `TOPIC` using `append_to_state`.

- **Step 2 — The Investigation (Parallel)**
	- **Agent:** `investigation_team` (ParallelAgent)
	- **Sub-agents:**
		- `admirer_agent` — augments queries with keywords like "achievements" or "positive impact" and appends findings to `pos_data`.
		- `critic_agent` — augments queries with keywords like "controversy" or "failures" and appends findings to `neg_data`.

- **Step 3 — The Trial & Review (Loop)**
	- **Agent:** `trial_and_review` (LoopAgent)
	- **Judge:** `judge_agent` reviews `pos_data` and `neg_data`.
	- **Behavior:** If data is unbalanced, the judge appends feedback to `JUDGE_FEEDBACK` to guide another investigation iteration. If balanced, the judge calls `exit_loop` to terminate the loop.

- **Step 4 — The Verdict (Output)**
	- **Agent:** `verdict_writer`
	- **Role:** Synthesizes `pos_data` and `neg_data` into an objective report and writes a `.txt` file into the `historical_verdicts/` directory using `write_file`.

**Technical Notes**

- **State keys:** `TOPIC`, `pos_data`, `neg_data`, `JUDGE_FEEDBACK`.
- **Research tools:** Agents use Wikipedia via LangChain integrations; queries are explicitly augmented to ensure viewpoint diversity.
- **Loop control:** Loop termination is performed strictly via the `exit_loop` tool called by the judge.

**Output**

The final verdicts are written as `.txt` files into `historical_verdicts/` by the `verdict_writer` agent.
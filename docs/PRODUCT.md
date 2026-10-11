# Aegis: the memory layer for your working life

> **Never lose your place. Never re-explain yourself. See where your time actually goes.**
> Aegis quietly remembers everything you do on your computer, on your machine, and turns it into answers,
> paste-ready context for any AI, and coaching based on what you actually did.

## The problem: context switching is the tax on every knowledge worker

A typical workday is fragmented across an editor, a terminal, a browser with 30 tabs, chat, email, meetings
and now several AI assistants. That costs time in four recurring ways:

| Time sink | What it looks like | What Aegis does instead |
|---|---|---|
| **Re-orientation** | Back from a meeting: "where was I?" Ten minutes rebuilding mental state | *"What was I doing before the standup?"* returns the file, the error and what you typed |
| **Re-finding** | Hunting for the error message, link, doc or number you saw yesterday | *"Where did I see the Stripe signature error?"* returns the moment, quoted and cited |
| **Re-explaining** | Every new AI chat starts with you retyping your setup and goal | **Copy safe context** gives a sanitized brief of your last 30 min, pasted into Claude/ChatGPT/Cursor |
| **Reporting** | Writing standups, timesheets, handoffs and "what did I do this week" | *"Write my standup"* or *"summarize my afternoon"*, generated from what actually happened |

And one you can't see without data: **fragmentation**. Aegis measures your context switches, focus blocks and
when you focus best, then coaches you from your real numbers.

## Who it's for

| Persona | Their day | The moment Aegis saves time |
|---|---|---|
| **The AI-native builder** (developers, data scientists) | editor ⇄ terminal ⇄ docs ⇄ 3 AI assistants | context packs into any agent; "what did I ask Claude to do?"; resuming after interruptions |
| **The researcher / student** | papers, videos, notes, tabs | "what was that paper about attention I read Tuesday?"; study-time insights |
| **The operator** (PMs, founders, consultants) | meetings, email, Slack, docs | standups and handoffs in one click; "when is the budget review?"; weekly reviews |
| **The easily-interrupted** (anyone whose attention gets fragmented) | constant pings, half-finished tasks | "where was I?" as a first-class button; gentle, factual focus feedback |

The first wedge is the **AI-native builder**: they feel the re-explaining tax many times a day, they already
work alongside agents, and they care most that their code and keys never leave the laptop.

## What "saving time" means, and how we measure it

We define it per use case, and Aegis can measure most of it from its own data:

| Metric | Definition | How Aegis measures it |
|---|---|---|
| Resumption time | time from returning to a task until productive input | episode boundaries before and after an interruption |
| Search time saved | time otherwise spent re-finding information | answers with a cited source vs. manual search sessions |
| Context-setup time | first minutes of each new AI chat spent explaining | number of context packs pasted × typical setup minutes |
| Report time | time spent writing standups and timesheets | generated reports used |
| Fragmentation | context switches per hour, focus ratio, refocus cost | `aegis insights`, every day |

On the synthetic eval day, Aegis answers resumption questions in about **1–2 s**. It found **10 context switches
≈ 20 min of refocus cost** and gave three concrete fixes tied to that day's actual blocks.

## Use cases (all working today)

1. **"Where was I?"** Ask *"what was I doing before the standup?"* or press the hotkey and ask. You get the file,
   the error and the last thing you typed, with clickable sources.
2. **Find anything you saw.** *"Where did I see the database is locked error?"*, *"what did Dana say about the
   cardiac model?"* Hybrid search finds exact strings and paraphrases alike.
3. **Quote what you typed.** *"What did I ask Claude to build?"* returns your prompt verbatim (Aegis keeps the
   final version of what you typed, not 30 drafts).
4. **Safe context for any AI.** **Copy safe context** (or *"get only the safe info to share"*) produces a paste-ready
   brief. Secrets and PII are scrubbed, email/chat/meeting/finance content is withheld, and no raw screen text is
   included. Your next assistant starts where you are, not from zero.
5. **Standups, handoffs, timesheets.** *"Summarize my afternoon"*, *"what did I do this morning?"*,
   *"write my standup"*: in time order, cited.
6. **Time coach.** *"How can I optimize my time?"* gives a measured report (time per category, switches, refocus
   cost, deep-work blocks, your best focus hour) and 3–5 specific changes. Example: *"you had two 27-min deep
   blocks, both before 11:00; schedule your hardest task at 10:00 and batch Slack into two slots."*
7. **Find an agent for the job (Darwin).** *"Find an agent that could help me plan my week"* searches the
   [Darwin](https://darwin.so) agent marketplace with a **generic** description of the need. Your memory never
   leaves the device; you see the exact query that did.
8. **Follow-ups that just work.** *"And before that?"*, *"more detail"*: the agent remembers which time window and
   sources it used last.

## Why this is different

| | Aegis | Microsoft Recall | Rewind / Limitless | Meta Muse | Screenpipe |
|---|---|---|---|---|---|
| Runs on | any Windows PC (macOS/Linux capture paths in code) | Copilot+ PCs only | Mac (screen capture shut down after Meta acquisition) | phone/web, cloud | Mac/Win/Linux |
| Where memory lives | local SQLite | local | was local, then cloud pendant | Meta's cloud | local |
| Answers like an agent (time-aware, follow-ups, tools) | ✅ | search UI | ✅ (was) | ✅ | via plugins/MCP |
| Remembers *what you typed* (final version) | ✅ | ✗ | partial | n/a | raw OCR |
| Privacy firewall for sharing context | ✅ scrub + withhold | ✗ | ✗ | ✗ | ✗ |
| Measures focus / fragmentation and coaches | ✅ | ✗ | ✗ | proactive tips, cloud | ✗ |
| Finds external agents without leaking memory | ✅ Darwin | ✗ | ✗ | own ecosystem | ✗ |

*(Competitor facts as reported publicly in 2025–26; see sources at the end.)*

Meta's Muse is a personal agent that knows you because you **upload** your life to it. Aegis is the opposite
bet: **it knows you because it watches your work, and it keeps what it knows on your machine.** It's not a
chatbot with memory. It's a memory with a chatbot on top, a time coach, and a privacy firewall for everything
you share.

## Product principles

1. **Local by default.** Memory never leaves the device. Hosted models are an explicit opt-in for demos and
   weaker hardware.
2. **Zero-effort capture, one-keystroke recall.** No tagging, no notes, no "save this".
3. **Cite or don't say it.** Every answer points to the moment it came from.
4. **Share the gist, never the secrets.** Everything outbound goes through the privacy firewall.
5. **Measure, then coach.** Advice is built only from your own numbers.

## Business model (proposal)

| Tier | Price | For |
|---|---|---|
| **Free / local** | $0 | full memory, chat, insights, safe packs; bring your own local models |
| **Pro** | ~$8/mo | hosted model credits for vision and chat, weekly review emails, encrypted cross-device sync |
| **Team** | ~$15/user/mo | shared *sanitized* standups and handoffs, team focus analytics (aggregate only, never screens) |

## Roadmap

- **Now:** Windows capture, HUD, terminal chat, insights, safe packs, Darwin search.
- **Next:**
  - an MCP server, so Claude, Cursor and any agent can *ask Aegis* for context with the same privacy firewall;
  - local OCR (Windows.Media.Ocr), so screenshots never leave even in hosted-chat mode;
  - pause and "forget the last 10 minutes" buttons;
  - start at login.
- **Later:**
  - a durable user-facts memory (projects, people, goals) with an editable "what Aegis knows about me" page;
  - phone access over Tailscale;
  - proactive nudges ("you've switched 20× in the last hour; want a focus block?").

## 3-minute demo script

1. *(Working in VS Code, then a Slack ping, then YouTube.)* Hotkey → **"what was I doing before I got distracted?"**
   → the file and error from 12 minutes ago, cited.
2. **"And before that?"** → it walks back in time.
3. **"What did I ask Claude to build?"** → your exact prompt, quoted.
4. **Copy safe context** → paste into a fresh ChatGPT. Show that the email and Slack content was withheld and the
   keys were scrubbed.
5. **"How can I optimize my time?"** → measured switches, refocus cost, best focus hour, three concrete changes.
6. **"Find an agent that could help me plan my week"** → Darwin results; point out that only the generic query left
   the laptop.
7. Close: *"Everything you just saw lives in one SQLite file on this laptop."*

## Sources

- [Personal AI memory in 2026: who owns what you remember (Screenpipe blog)](https://screenpi.pe/blog/personal-ai-memory-2026)
- [Rewind AI alternatives in 2026 (Screenpipe blog)](https://screenpi.pe/blog/rewind-ai-alternative-2026)
- [What is Meta Muse? (Built In)](https://builtin.com/articles/meta-muse)
- [Muse: a do-it-all AI assistant (The Week)](https://theweek.com/culture-life/personal-technology/muse-meta-ai-agent-personal-assistant)
- [Darwin: search engine for the agentic web](https://darwin.so/)

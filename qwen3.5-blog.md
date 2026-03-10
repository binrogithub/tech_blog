# From Zero to a Working Coding Assistant: Deploying Qwen3.5-9B on a Huawei Cloud 910B Notebook

There is a very specific kind of joy in getting a local model to answer its first prompt.

Not because the answer is always brilliant. Sometimes it is. Sometimes it is just "Hello." But that tiny moment matters because it means the stack is alive: the model weights are in place, the runtime is healthy, the accelerator is doing real work, and the notebook has turned into something more interesting than a rented shell with an expensive power bill.

This post is about one such deployment: taking **Qwen3.5-9B** and running it on a **Huawei Cloud ModelArts Notebook with an Ascend 910B card**.

The story matters because this was not a "download one package and you are done" experience. It involved choosing the right model size, thinking carefully about what "strongest single-card model" really means, working with Ascend-specific constraints, and validating the final setup with a lightweight coding workflow through **opencode**.

If you are not deeply technical, that is fine. I will keep the explanations concrete and human. If you *are* technical, there will still be enough detail here to reproduce the workflow.

## Why Qwen3.5-9B?

Before writing a single command, the most important decision is model selection.

When people say "I want the strongest model I can run on one card," they often jump straight to the biggest name they recognize. That is how you end up downloading something enormous, waiting hours, and discovering it does not fit, or worse, it fits only in theory and collapses in practice once the runtime, caches, and framework overhead enter the picture.

For a **single Ascend 910B 64GB Notebook**, the selection logic looked like this:

- Very large models such as 30B+ class checkpoints were not realistic on one card unless they used a highly compatible quantization path.
- Some attractive quantized variants existed, but their support on Ascend was not always mature or predictable.
- We wanted a model that was:
  - reasonably strong,
  - realistic on one 64GB device,
  - deployable without multi-card tensor parallelism,
  - and useful for both normal chat and light coding tasks.

That is where **Qwen3.5-9B** landed.

It is not the largest model in the family. It is not the flashiest benchmark trophy. But for this hardware target, it hits a practical sweet spot: large enough to be genuinely useful, small enough to be deployable without turning the whole exercise into infrastructure archaeology.

If I had to summarize the selection principle in one sentence, it would be this:

> On a single 910B card, the best model is not the biggest model you can name. It is the biggest model you can deploy cleanly, keep stable, and actually use.

## Picking the Download Source

Once the model was chosen, the next question was where to download it from.

That question matters more than people expect.

In cloud notebook environments, download speed and reliability can vary wildly depending on the source. Hugging Face is the obvious default, but in China-region or Huawei Cloud environments, mirrors and alternative model hubs are often much more practical. For this deployment, the model workflow was built around the environment available on the notebook, which already had Python and package tools set up and could install from normal Python indexes.

For Qwen3.5-9B, the key goal was simple:

- use an official or widely trusted model repository,
- keep the path stable,
- and avoid unnecessary conversion steps before first deployment.

The deployment layout used two storage layers:

- persistent storage in `/home/ma-user/work`
- fast local storage in `/cache`

That separation turned out to be important.

The reason is easy to understand:

- `/home/ma-user/work` is where you want things to survive notebook restarts.
- `/cache` is where you want things to run fast.

So the practical pattern becomes:

1. Download the model into `/home/ma-user/work/models/...`
2. Validate that the files are complete
3. Copy the model into `/cache/models/...` for actual serving

It is a small design decision, but it makes the notebook feel much less fragile.

## The Hardware Check: Trust, but Verify

Before installing anything heavy, I verified what the notebook really had.

That included:

- current user,
- Python version,
- storage size,
- and most importantly, visible Ascend NPUs.

The machine reported:

- `Python 3.11.10`
- architecture: `aarch64`
- one visible Ascend 910B device in the earlier single-card environment

That combination matters because it narrows the range of compatible runtime stacks very quickly. Python version affects package availability, architecture affects which wheels exist, and the NPU runtime determines whether a "supported" model is actually supported in practice.

This is one of those moments where deployment gets philosophical.

A machine may look powerful on paper, but if the package ecosystem around it is awkward, your real usable compute can feel much smaller than the headline spec suggests.

## Environment Preparation

To keep the deployment isolated from the base notebook environment, I created a dedicated virtual environment under the working directory.

The rough shape was:

```bash
python3 -m venv /home/ma-user/work/venvs/qwen35
source /home/ma-user/work/venvs/qwen35/bin/activate
python -m pip install -U pip setuptools wheel
```

This is not glamorous work, but it is the difference between a notebook you can reason about and a notebook that slowly turns into a haunted house of package conflicts.

I also created a simple directory layout:

```text
/home/ma-user/work/
  models/
  logs/
  venvs/
  src/
```

The point was to keep:

- model files,
- runtime logs,
- Python environments,
- and any custom scripts

in predictable places.

That predictability becomes a gift later, when you need to debug an import failure at midnight and your future self is not in a forgiving mood.

## Installing the Runtime Stack

For this Qwen3.5-9B deployment, the first successful path did **not** use `vllm-ascend`.

That is an important detail.

Why? Because the practical goal was to get a *working single-card deployment* first. We were not optimizing for the fanciest serving stack. We were optimizing for a model that could reliably answer requests on Ascend hardware.

The final working stack for this model used:

- `transformers`
- `accelerate`
- `torch_npu`

and a lightweight custom server layer.

At one point, the notebook had an older `transformers` version that did not understand the `qwen3_5` model type properly. That created the kind of failure that is almost insulting in its simplicity: the model is there, the weights are there, the hardware is there, and the loader still says "I do not know what this thing is."

The fix was to upgrade the user-space packages to versions new enough to recognize the checkpoint:

```bash
python -m pip install -U transformers accelerate
```

That was one of the biggest practical lessons from the deployment:

> In model serving, "the model does not load" often means "your library is old," not "the model is broken."

## A Minimal but Useful Server

Instead of waiting for a more elaborate serving stack to become perfect, I used a simpler design:

- load tokenizer,
- load model onto `npu:0`,
- expose a small HTTP API,
- and keep the interface compatible enough for downstream tools to consume.

That server eventually provided:

- a basic health endpoint,
- generation,
- and later, an OpenAI-compatible API surface

so that tools like `opencode` could talk to it naturally.

This may sound like a compromise. In practice, it was a smart move.

A small server has two huge advantages:

1. It is easier to reason about when things go wrong.
2. It proves that the hard part, model loading on Ascend, actually works.

Only after you have that proof should you start fantasizing about polished serving infrastructure.

## Getting the Model to Actually Answer

The first successful deployment is rarely elegant.

The model answered. That mattered more than whether the code path looked pretty.

Once the server was up, the key validation steps were:

- check health,
- send a simple text generation request,
- make sure the model produced output,
- confirm NPU memory usage climbed in a sensible way,
- and verify that the process stayed alive after the first request.

This is the moment where deployment goes from theoretical to emotional. The logs stop being abstract. The machine becomes conversational.

It is also where you learn that "runs once" and "deployable" are very different things.

So I made a point of testing with multiple kinds of prompts:

- short generic prompts,
- basic coding prompts,
- and later, tool-driven requests through `opencode`.

## Wiring in opencode

Once the local model service was stable enough, the next step was making it useful.

This is where **opencode** came in.

The idea was simple:

- treat the local model as an OpenAI-compatible endpoint,
- configure `opencode` to point to it,
- and see whether the combination could act like a lightweight coding assistant.

The integration flow looked like this:

1. Install `opencode`
2. Create a local model entry in the config
3. Point that entry to the notebook’s local API endpoint
4. Run both interactive and one-shot prompts through it

That turned a locally served model into something much closer to a usable developer tool.

And honestly, this is the part that makes the whole deployment feel worth it. A local model sitting alone is interesting. A local model plugged into a tool you can *actually use* is much more compelling.

## How Good Was It for Coding?

This was the most interesting part of the experiment.

I tested it in two modes:

1. **question-answer style coding**
2. **actual agent-like project work**

The results were mixed, and that is exactly why they are useful.

### What it did well

For classic coding-assistant tasks, Qwen3.5-9B was solid:

- explaining code,
- fixing simple bugs,
- writing small utility functions,
- solving common interview-style tasks,
- and generating short snippets in a predictable style

It behaved like a competent lightweight coding model.

Not magical. But useful.

### Where it struggled

When asked to act more like an autonomous coding agent, it was much less impressive.

Examples of things it struggled with:

- multi-step project changes,
- actually editing files in a real repo,
- following through from "I will inspect the code" to real modifications,
- and reliably completing a bug-fix workflow end to end

That is an important distinction.

Qwen3.5-9B can be a decent **coding assistant**. It is not, in this setup, a strong **coding agent**.

That sounds like a downgrade, but it is actually clarifying. Once you know what it is good at, you stop asking it to be something else.

## The Human-Friendly Summary

If you are not deep in model deployment, here is the plain-English version:

- We chose a model that could realistically fit on one Ascend 910B card.
- We downloaded it into persistent storage and ran it from fast local storage.
- We fixed the Python package stack until the model could load on Ascend.
- We wrapped it in a lightweight server.
- We connected it to `opencode`.
- We tested it like a normal user would.

And the result was this:

**Qwen3.5-9B is a good local chatbot and a decent lightweight coding assistant on Huawei Cloud 910B.**

It is not the model I would trust as a fully autonomous code-rewriting machine. But it is absolutely good enough to:

- answer technical questions,
- explain code,
- generate small code blocks,
- help with documentation,
- and provide a local private assistant experience.

## What I Would Do Next

If I we turning this into a more polished setup, the next steps would be:

- package the working environment more cleanly,
- make the server a proper managed background service,
- tighten response formatting for coding use,
- and benchmark response speed under different prompt sizes

If I were optimizing specifically for code-heavy workflows, I would also consider pairing this deployment with a stronger coder-oriented model later, while keeping Qwen3.5-9B around as the general-purpose assistant.

That combination makes a lot of sense in practice:

- one local model for broad chat and utility tasks
- another stronger model for real repository work

## Final Thoughts

A lot of AI deployment writing makes the process sound either trivial or mystical.

In reality, it is neither.

It is a sequence of ordinary engineering decisions:

- choose a realistic model,
- respect the hardware,
- keep the environment isolated,
- use stable storage patterns,
- test with real workflows,
- and let the machine prove what it can do.

That is what made this deployment satisfying.

Not that it was perfect. It was not.  
Not that the model was all-powerful. It was not.

It was satisfying because it became *real*.

A model that fits the hardware, answers prompts, survives real requests, and plugs into a useful tool is no longer a demo. It is infrastructure.

And that, in the end, is the moment every deployment is chasing.


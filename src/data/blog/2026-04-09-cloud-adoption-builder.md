---
author: Robin
pubDatetime: 2026-04-09T17:45:00-03:00
title: "Two Days, Not Two Months: Why I Gave Up the CTO Title to Become a Cloud Adoption Builder"
description: "A field report on how AI compresses cloud adoption work from months to days: from requirement analysis and driver debugging to Kafka-to-database benchmarking and customer-ready demo delivery."
tags:
  - cloud-adoption
  - ai-engineering
  - builder
  - kafka
  - databases
  - benchmarking
  - customer-engineering
  - productivity
featured: false
draft: false
---

# Two Days, Not Two Months: Why I Gave Up the CTO Title to Become a Cloud Adoption Builder

## A confession

I used to be a CTO. The title was nice. The work, less so. Most of my days
went into managing people, managing requirements, managing commercials.
I spent a lot of time talking *about* technology and very little time
actually *doing* it. The closer I got to the title, the further I drifted
from the thing that pulled me into this industry in the first place:
turning a good idea into a working system that solves a real problem.

So I gave up the title.

Today my title is **Cloud Adoption Builder**. The "Builder" part is
deliberate. It means I write code, I run benchmarks, I sit in front of
a terminal with the customer's problem on one screen and a working
pipeline on the other. It means when somebody asks "can the cloud do X
for our business," my answer is not a slide deck but a live demo with
real numbers from real infrastructure.

The reason I could make this switch is simple: **AI now compresses the
boring parts of cloud adoption by 10x to 100x**, and the boring parts
were exactly what made this kind of role unscalable for one person before.

What follows is one example of what that looks like in practice. Not
the story that convinced me — just a normal week.

---

## What I had on Monday morning

A customer asked us to prove out a high-throughput event ingestion
pipeline on their cloud environment. The shape of the problem was
familiar: a managed Kafka cluster on the front, a distributed
relational database on the back, and a Python application in between
that needs to consume events, transform them, and persist them at
several thousand operations per second with strict no-data-loss
guarantees.

The ask was: by Wednesday afternoon, demonstrate that this can actually
work end-to-end on the customer's cloud environment. Not a slide deck.
Not a proof-of-concept document. A live, running pipeline they could
see and touch, with real benchmark numbers we could defend in a room
full of skeptics.

The old me would have spent the first two days reading the requirements,
booking calls with infrastructure, drafting an architecture document,
and asking the team for capacity estimates. The first commit would have
landed maybe day five, in a sandbox environment, with one developer
fighting an unfamiliar database driver and another trying to remember
the Kafka producer configuration.

I had two days. So I did something different.

---

## Day one, hour one: read the room

I opened the requirements document next to my AI assistant and asked it
to walk me through every section, flag the parts that were ambiguous,
and produce a candidate Q&A list for the customer. In about ten minutes
I had a structured understanding of:

- What the document explicitly required.
- What the document explicitly left open and needed clarification.
- What the document didn't say but a real production deployment would
  need anyway.

The same exercise normally takes two engineers a full day to do well,
because somebody has to read the whole thing, somebody else has to read
the related background, and then the two of them have to argue. Done in
ten minutes by me alone, with the AI doing the comprehension work and
me doing the judgment work. **That's not faster, that's a different
shape of work.**

---

## Day one, hour two: get on the actual cloud

The customer's environment was a managed Kafka cluster and a distributed
relational database, both already provisioned. I had a username, a
password, and a handful of IP addresses for each.

First connectivity attempt: TCP open, but the standard PostgreSQL Python
driver returned `none of the server's SASL authentication mechanisms are
supported`. The managed database uses a vendor-specific SHA256 dialect
that the upstream open-source driver doesn't speak.

In the old world, that's a half-day rabbit hole: search the vendor
docs, download the right driver bundle, fight with dependency conflicts,
deal with the system Python's libpq mismatch, file a ticket with the
cloud team, wait. In the new world, I described the symptom to the AI,
it told me which vendor driver bundle to download from the public
download URL, and it walked me through extracting the right shared
libraries and the bundled driver for the local Python version.
Twenty minutes later I was running `SELECT version()` against all
three database nodes.

This pattern repeated all day:

- First end-to-end pipeline run → producer worked, consumer hit a
  Kafka consumer-group rebalance bug under load. The AI suggested
  switching from `subscribe` to manual partition assignment with
  `assign`, which I hadn't used in a decade.
- First database insert burst → 500 transactions per second. The AI
  immediately diagnosed `cur.executemany()` as the culprit and rewrote
  it to use `psycopg2.extras.execute_values`, which jumped to 5,000+
  TPS. A few hours later we replaced that with `COPY FROM STDIN` and
  it jumped again to 11,000+ TPS.
- First multi-worker run → throughput went *down* instead of up.
  Diagnosed within two minutes: Python GIL contention. The fix was
  multi-process instead of multi-thread, with one full pipeline
  per process and host rotation across the database nodes.
- Cluster partition limit hit mid-test because old test topics were
  silently consuming the budget. The AI wrote a one-shot cleanup
  function and integrated it into the setup command so it would never
  bite us again.

Every single one of these would have been a one-to-three-day excursion
in the old days, with research, trial-and-error, Stack Overflow
archaeology, and at least one Slack thread asking somebody else who'd
seen it before. With the AI, none of them lasted longer than thirty
minutes.

---

## Day two, morning: build the pipeline

By the time I was building the actual end-to-end pipeline, the pattern
was clear. I would describe what I wanted the next stage to do, the AI
would write the first version, I would run it, we would look at the
output together, and I would ask for the next refinement.

The pipeline had three concurrent stages: a producer that synthesized
events and pushed them to Kafka, a processor that consumed from Kafka
and persisted to the primary table, and a downstream persister that
read from a second Kafka topic and wrote to a secondary table. Plus a
query benchmark, a topic management layer, and a metrics aggregator.
About 1,200 lines of Python and SQL, written in roughly four hours.
I touched maybe a third of the lines myself; the rest was AI-generated
and AI-iterated under my supervision.

The first end-to-end run on 10,000 events finished in under a minute.
End-to-end throughput: 1,463 events per second. Below target, but the
pipeline was *working*, end to end, on the customer's real
infrastructure.

The next four hours were nothing but optimization. Each iteration was
a hypothesis, a fix, a re-run, a measurement. The AI was the pair
programmer for every step:

```
 1,463 events/s   →  baseline
 2,668 events/s   →  + orjson + COPY FROM STDIN
 3,554 events/s   →  + multi-process workers
 6,364 events/s   →  + database CN rotation
10,742 events/s   →  + tuning batch size and partition count
20,982 events/s   →  + single-thread per process,
                     letting the OS scheduler do its job
```

By the end of day two, we were running **20,982 events per second
end-to-end** on a one-million-event test, with **zero data loss**, and
with full Kafka durability mode enabled (acks=all, idempotent producer,
min.insync.replicas=2, unclean leader election disabled). The cost of
turning on full zero-data-loss durability was **0.2%** of throughput.

---

## Day two, evening: prepare for the customer

Now the part the old CTO me would actually have been good at: preparing
the customer-facing materials. A demo script, a command cheat sheet, a
question list, an email to a colleague who was going to drive the demo
while I was on standby.

In the old days I would have spent a full day on this. With AI helping
me draft, structure, and translate between two languages (the customer
is bilingual), it took an hour. Including the talking points for each
section, the fallback commands in case the live demo broke, the
pre-flight checklist, and a one-page email to the colleague who would
deliver the demo cold.

Total elapsed time for the full PoC, from "here is the requirements
document" to "here is a working live demo and a documented playbook":
**two days**.

The old version of this work would have been a two-month project with
a four-person team. I know because I have managed exactly that kind of
project, with exactly that kind of team, more times than I want to
count.

---

## What I think this means

The interesting question is not "how did AI write the code." Plenty of
people have written about that. The interesting question is **what
this changes about the role of a senior technical person at a cloud
provider**.

For me the answer is: it shifts the bottleneck off execution and onto
**judgment**.

Before AI, my job was 80% management because the technical execution
was so expensive that only specialists could deliver it, and someone
had to coordinate the specialists. The CTO title was the right title
for that job.

After AI, the technical execution is cheap enough that one person who
understands the business can deliver an end-to-end demo in two days.
The bottleneck is no longer how fast you can write the code, or how
many engineers you can hire, or how well you can coordinate them. The
bottleneck is **whether you understand the customer's problem deeply
enough to ask the right next question**.

That's not a CTO. That's a builder.

---

## What I'm doing next

Three things:

1. Understanding customer needs faster than they can write them down.
2. Validating those hypotheses on the actual cloud, with real data,
   in days instead of months.
3. Going deep with the customer's application and data teams, not
   their procurement team.

This is the work I would never have made time for as a CTO. As a
Cloud Adoption Builder, it's all I do. And I have not regretted the
switch once.
